"""Feature builder – converts raw parquet into model-ready feature parquet.

Steps
-----
Load 1-minute candle parquet from `data/`.
Resample to 15-minute bars (OHLCV agg) using Polars `group_by_dynamic`.
Compute technical indicators (RSI-14, returns, Bollinger width, ATR).
Join macro factors pulled live from Data-API `/macro/latest` (nearest ts).
Save to `features/<token>.parquet` (overwrites; stateless).
"""
from __future__ import annotations

# Inject project root into sys.path so that `utils` resolves when script is called directly
import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

import argparse
import asyncio
import datetime as dt
from typing import List

import aiohttp
import polars as pl
import os
import sqlite3
from pathlib import Path

from utils import DATA_DIR, FEATURES_DIR, DATA_API_URL

# Use Data-API URL from environment (fallback localhost)
API_URL = DATA_API_URL

# ---------------------------------------------------------------------------
# Database paths (configurable via $ANALYTICS_DB_DIR)
# ---------------------------------------------------------------------------
DATABASES = {
    "ohlcv": "/root/analytics-tool-v2/market_data.db",
    "funding": "/root/data-service/futures_metrics.db",
    "orderbook": "/root/data-service/orderbook.db",
    "macro": "/root/data-service/macro.db",
}

MARKET_DB = Path(DATABASES["ohlcv"])
FUTURES_DB = Path(DATABASES["funding"])
ORDERBOOK_DB = Path(DATABASES["orderbook"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rsi(expr: pl.Expr, window: int = 14) -> pl.Expr:  # noqa: D401
    delta = expr - expr.shift(1)
    gain = pl.when(delta > 0).then(delta).otherwise(0).rolling_mean(window)
    loss = pl.when(delta < 0).then(-delta).otherwise(0).rolling_mean(window)
    rs = gain / loss
    return 100 - (100 / (1 + rs))


async def fetch_macro() -> pl.DataFrame:  # noqa: D401
    async with aiohttp.ClientSession() as sess:
        async with sess.get(f"{API_URL}/macro/latest") as r:
            r.raise_for_status(); js = await r.json()
    return pl.from_dicts([js])


def _read_sqlite(db_path: Path, query: str) -> pl.DataFrame:
    """Return query result as a Polars DataFrame (empty if error)."""
    if not db_path.exists():
        print(f"[ERROR] DB not found: {db_path}")
        return pl.DataFrame()
    try:
        with sqlite3.connect(db_path) as conn:
            return pl.read_database(query, conn)
    except Exception as exc:
        print(f"[warn] sqlite error on {db_path}: {exc}")
        return pl.DataFrame()


def load_funding(symbol: str) -> pl.DataFrame:
    return _read_sqlite(FUTURES_DB, f"SELECT ts, rate AS funding_rate FROM funding_rate WHERE symbol='{symbol}' ORDER BY ts")


def load_oi(symbol: str) -> pl.DataFrame:
    return _read_sqlite(FUTURES_DB, f"SELECT ts, oi AS open_interest FROM open_interest WHERE symbol='{symbol}' ORDER BY ts")


def load_orderbook(symbol: str) -> pl.DataFrame:
    return _read_sqlite(ORDERBOOK_DB, f"SELECT ts, (bidVol - askVol) * 1.0 / (bidVol + askVol) AS imbalance FROM ob_imbalance WHERE symbol='{symbol}' ORDER BY ts")


def load_btc_returns() -> pl.DataFrame:
    df = _read_sqlite(MARKET_DB, "SELECT timestamp AS ts, close FROM candles_900s WHERE symbol='BTCUSDT' ORDER BY ts")
    if df.is_empty():
        return df
    return df.with_columns([
        (pl.col("ts")).alias("ts"),
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("btc_ret")
    ]).drop("close")


def build(token: str) -> None:  # noqa: D401
    src_path = DATA_DIR / f"{token}_1m.parquet"
    if src_path.exists():
        df = pl.read_parquet(src_path)
    else:
        # Fallback: pull from Data-API live endpoint
        print(f"[info] local raw parquet missing for {token}; fetching from Data-API…")
        import requests, pandas as pd
        # Ensure we only append USDT once
        symbol = token if token.endswith("USDT") else f"{token}USDT"
        resp = requests.get(f"{API_URL}/candles/{symbol}/1m", params={"limit": 43200})  # 30 days of 1-min
        resp.raise_for_status()
        js = resp.json()
        # unwrap if server responds with {{..., 'candles': [...]}}
        if isinstance(js, dict) and 'candles' in js:
            js = js['candles']
        if not js:
            # secondary fallback hierarchy: first Binance SPOT klines, then Binance FUTURES klines
            print(f"[info] Data-API empty for {symbol}; trying Binance REST…")
            # try SPOT first
            klines = []
            for base_url in ("https://api.binance.com/api/v3/klines", "https://fapi.binance.com/fapi/v1/klines"):
                b_resp = requests.get(base_url, params={"symbol": symbol, "interval": "1m", "limit": 1000})
                if b_resp.status_code == 200 and b_resp.json():
                    klines = b_resp.json()
                    if klines:
                        break
            if klines:
                js = [{
                    "ts": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5])
                } for k in klines]
            else:
                raise SystemExit("No candle data from Data-API nor Binance (spot or futures)")
        df = pl.from_pandas(pd.DataFrame(js))
        # Standardise expected columns
        rename_map = {"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        overlap = {k: v for k, v in rename_map.items() if k in df.columns}
        if overlap:
            df = df.rename(overlap)

    # Harmonise column names
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename({"vol": "volume"})

    if df.height < 100:
        # Warn but continue; downstream resample may still generate enough rows
        print(f"[warn] small dataset for {token} ({df.height}); proceeding anyway")
        # Do not early-return; attempt to build features even with few rows

    # Ensure datetime column
    if "ts" in df.columns:
        first_ts_val = df["ts"].item(0)
        unit = "ms" if first_ts_val > 1e12 else "s"
        df = df.with_columns(pl.from_epoch("ts", time_unit=unit).alias("dt"))
    elif "timestamp" in df.columns:
        first_ts_val = df["timestamp"].item(0)
        unit = "ms" if first_ts_val > 1e12 else "s"
        df = df.with_columns(pl.from_epoch("timestamp", time_unit=unit).alias("dt"))

    # Resample to 15m OHLCV (handle missing volume gracefully)
    agg_cols = [
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
    ]
    if "volume" in df.columns:
        agg_cols.append(pl.col("volume").sum().alias("volume"))
    else:
        agg_cols.append(pl.lit(0).alias("volume"))  # fallback

    resampled = (
        df.sort("dt")
        .group_by_dynamic("dt", every="15m", closed="right", group_by=[])
        .agg(agg_cols)
        .drop_nulls()
    ).with_columns([
        pl.col("dt"),
        pl.col("dt").cast(pl.Int64).alias("ts")
    ])

    # Enhanced features for AAA-grade performance
    resampled = resampled.with_columns([
        # Volume features
        (pl.col("volume") / pl.col("volume").rolling_mean(24)).alias("volume_spike"),
        (pl.col("volume").rolling_mean(4) / pl.col("volume").rolling_mean(24)).alias("volume_momentum"),
        
        # Price structure features
        ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-9)).alias("buy_pressure"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("high_low_range"),
        
        # Liquidity & regime context (24-hour window)
        ((pl.col("close") * pl.col("volume")).rolling_sum(96)).alias("vol_usd_24h"),
        (pl.col("close").log().diff().rolling_std(96)).alias("volatility_24h"),
        # True range and 20-bar ATR (≈5h)
        (pl.max_horizontal(pl.col("high") - pl.col("low"),
                           (pl.col("high") - pl.col("close").shift(1)).abs(),
                           (pl.col("low") - pl.col("close").shift(1)).abs()).rolling_mean(20)).alias("atr_20"),
        # Multi-timeframe returns
        (pl.col("close") / pl.col("close").shift(4) - 1).alias("ret_1h"),
        (pl.col("close") / pl.col("close").shift(16) - 1).alias("ret_4h"),
        (pl.col("close") / pl.col("close").shift(96) - 1).alias("ret_24h"),
        
        # Volatility features
        pl.col("close").pct_change().rolling_std(12).alias("volatility_3h"),
        pl.col("close").pct_change().rolling_std(48).alias("volatility_12h"),
        
        # Price momentum
        (pl.col("close") / pl.col("close").rolling_mean(12) - 1).alias("price_vs_sma12"),
        (pl.col("close") / pl.col("close").rolling_mean(48) - 1).alias("price_vs_sma48"),
    ])

    # VWAP calculation
    resampled = resampled.with_columns([
        ((pl.col("close") * pl.col("volume")).rolling_sum(20) / pl.col("volume").rolling_sum(20)).alias("vwap_20"),
    ])
    resampled = resampled.with_columns([
        ((pl.col("close") - pl.col("vwap_20")) / pl.col("vwap_20") * 100).alias("price_vs_vwap"),
    ])

    # Technicals (keep existing)
    resampled = resampled.with_columns([
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret_1"),
        rsi(pl.col("close"), 14).alias("rsi14"),
    ])

    # Bollinger Band width using expressions to avoid list dtypes
    resampled = resampled.with_columns([
        (pl.col("close").rolling_std(20) * 2 / pl.col("close").rolling_mean(20)).alias("bb_width"),
    ])

    # ATR 14 – max of three true-range components, then rolling mean
    tr_expr = pl.max_horizontal([
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("close").shift(1)).abs(),
        (pl.col("low") - pl.col("close").shift(1)).abs(),
    ])
    resampled = resampled.with_columns([
        tr_expr.alias("tr"),
        tr_expr.rolling_mean(14).alias("atr14"),
    ])

    if resampled.height < 50:
        print(f"[warn] {token} insufficient resampled data – continuing anyway")

    # Macro factors with enhanced processing
    macro = asyncio.run(fetch_macro())

    # Ensure macro has millisecond 'ts' column for asof join
    if "timestamp" in macro.columns and "ts" not in macro.columns:
        macro = macro.with_columns((pl.col("timestamp") * 1000).alias("ts"))

    merged = resampled
    if "ts" in macro.columns:
        merged = merged.join_asof(macro.drop_nulls("ts"), on="ts", strategy="backward")
    else:
        print("[warn] macro feed missing 'ts'; skipping macro join")

    # Add funding rate, open interest, orderbook imbalance from real DBs
    symbol_map = {"PEPE": "PEPEUSDT"}
    symbol = symbol_map.get(token, f"{token}USDT")
    funding_df = load_funding(symbol)
    oi_df = load_oi(symbol)
    ob_df = load_orderbook(symbol)
    btc_df = load_btc_returns()

    # Ensure proper dtype and sorting
    def _prep(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df
        return df.sort("ts")

    funding_df = _prep(funding_df)
    oi_df = _prep(oi_df)
    ob_df = _prep(ob_df)
    btc_df = _prep(btc_df)

    # Ensure millis ts and proper dtype
    for _df in (funding_df, oi_df, ob_df, btc_df):
        if not _df.is_empty():
            _df = _df.with_columns([
                pl.from_epoch("ts", time_unit="s").alias("dt") if _df["ts"].dtype == pl.Int64 else pl.col("ts"),
            ])

    # Conditional joins for market metrics
    if not funding_df.is_empty():
        merged = merged.join_asof(funding_df, on="ts", strategy="backward", suffix="_funding")
    if not oi_df.is_empty():
        merged = merged.join_asof(oi_df, on="ts", strategy="backward", suffix="_oi")
    if not ob_df.is_empty():
        merged = merged.join_asof(ob_df, on="ts", strategy="backward", suffix="_ob")
    if not btc_df.is_empty():
        merged = merged.join_asof(btc_df, on="ts", strategy="backward")

    # Build derived features only if source columns exist
    feature_exprs = []
    if "funding_rate" in merged.columns:
        feature_exprs += [
            (pl.col("funding_rate") - pl.col("funding_rate").rolling_mean(32)).alias("funding_rate_z"),
            pl.col("funding_rate").rolling_mean(8).alias("funding_8h_avg"),
        ]
    if "open_interest" in merged.columns:
        feature_exprs.append(pl.col("open_interest").pct_change(4).alias("oi_change_1h"))
    if "imbalance" in merged.columns:
        feature_exprs.append(pl.col("imbalance").fill_null(0).alias("ob_imbalance"))
    if "btc_ret" in merged.columns:
        feature_exprs += [
            pl.col("btc_ret").alias("btc_return"),
            (pl.col("ret_1") * pl.col("btc_ret")).rolling_mean(12).alias("btc_corr_proxy"),
        ]

    if feature_exprs:
        merged = merged.with_columns(feature_exprs)

    merged = merged.drop("ts")
    # Drop any non-scalar columns (List / Struct) that break model training
    merged = merged.select(pl.exclude([pl.List, pl.Struct]))

    out_path = FEATURES_DIR / f"{token}.parquet"
    merged.write_parquet(out_path, compression="zstd")
    print(f"[+] features → {out_path.relative_to(FEATURES_DIR.parent)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("crypto-models-feature-builder")
    p.add_argument("--token", required=True)
    args = p.parse_args()
    build(args.token)
