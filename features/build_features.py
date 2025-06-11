"""Feature builder – converts raw parquet into model-ready feature parquet.

Steps
-----
1. Load 1-minute candle parquet from `data/`.
2. Resample to 15-minute bars (OHLCV agg) using Polars `group_by_dynamic`.
3. Compute technical indicators (RSI-14, returns, Bollinger width, ATR).
4. Join macro factors pulled live from Data-API `/macro/latest` (nearest ts).
5. Save to `features/<token>.parquet` (overwrites; stateless).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
from typing import List

import aiohttp
import polars as pl

from utils import DATA_DIR, FEATURES_DIR, DATA_API_URL

# Use Data-API URL from environment (fallback localhost)
API_URL = DATA_API_URL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rsi(series: pl.Series, window: int = 14) -> pl.Series:  # noqa: D401
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling_mean(window)
    loss = (-delta.clip(upper=0)).rolling_mean(window)
    rs = gain / loss
    return 100 - (100 / (1 + rs))


async def fetch_macro() -> pl.DataFrame:  # noqa: D401
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{API_URL}/macro/latest") as resp:
            resp.raise_for_status()
            js = await resp.json()
    return pl.from_dicts([js])


def build(token: str) -> None:  # noqa: D401
    src_path = DATA_DIR / f"{token}_1m.parquet"
    if not src_path.exists():
        raise SystemExit(f"Raw parquet not found: {src_path}")

    df = pl.read_parquet(src_path)
    # Ensure datetime column
    if "ts" in df.columns:
        df = df.with_columns(pl.col("ts").cast(pl.Int64).alias("ts"))
        df = df.with_columns((pl.col("ts") * 1_000).cast(pl.Int64).alias("ts_ms"))
        df = df.with_columns(pl.from_epoch("ts", time_unit="s").alias("dt"))
    else:
        df = df.with_columns(pl.from_epoch("timestamp", time_unit="s").alias("dt"))

    # Resample to 15m OHLCV
    resampled = (
        df.sort("dt")
        .group_by_dynamic("dt", every="15m", closed="right", by=[])
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .drop_nulls()
    )

    # Technicals
    resampled = resampled.with_columns([
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret_1"),
        rsi(pl.col("close"), 14).alias("rsi14"),
    ])

    # Bollinger width 20
    ma20 = resampled["close"].rolling_mean(20)
    std20 = resampled["close"].rolling_std(20)
    resampled = resampled.with_columns((2 * std20 / ma20).alias("bb_width"))

    # ATR 14
    tr = pl.max([
        (resampled["high"] - resampled["low"]),
        (resampled["high"] - resampled["close"].shift(1)).abs(),
        (resampled["low"] - resampled["close"].shift(1)).abs(),
    ])
    resampled = resampled.with_columns(tr.alias("tr"))
    resampled = resampled.with_columns(resampled["tr"].rolling_mean(14).alias("atr14"))

    # Macro factors
    macro = asyncio.run(fetch_macro())
    macro_ts = macro["timestamp"].item()
    macro_dt = dt.datetime.utcfromtimestamp(macro_ts)
    macro = macro.with_columns(pl.lit(macro_dt).alias("dt"))
    merged = resampled.join_asof(macro, on="dt", strategy="backward")

    out_path = FEATURES_DIR / f"{token}.parquet"
    merged.write_parquet(out_path, compression="zstd")
    print(f"[+] features → {out_path.relative_to(FEATURES_DIR.parent)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("crypto-models-feature-builder")
    p.add_argument("--token", required=True)
    args = p.parse_args()
    build(args.token)
