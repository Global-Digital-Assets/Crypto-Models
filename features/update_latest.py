"""Append a fresh feature row for each token.

Runs every minute via systemd timer so that the last row in
`features/<TOKEN>.parquet` always reflects near-real-time market data.

We fetch just enough raw data (last 120 minutes of 1-min candles plus
funding-rate / order-book metrics) from the Data-API, recompute the same
feature columns as `build_features.build`, then append (or overwrite) the
row for the latest 15-min bar.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List

import aiohttp
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from utils import FEATURES_DIR, DATA_API_URL, ROOT_DIR  # noqa: E402

REGISTRY = ROOT_DIR / "registry.yaml"
API = DATA_API_URL.rstrip("/")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    return int(dt.datetime.utcnow().timestamp() * 1000)


def _rsi(series: pl.Series, window: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling_mean(window)
    loss = (-delta.clip(upper=0)).rolling_mean(window)
    rs = gain / (loss + 1e-9)
    return float(100 - 100 / (1 + rs.iloc[-1]))


def _boll_bw(close: pl.Series, window: int = 20) -> float:
    mean = close.rolling_mean(window)
    std = close.rolling_std(window)
    upper = mean + 2 * std
    lower = mean - 2 * std
    return float((upper.iloc[-1] - lower.iloc[-1]) / (mean.iloc[-1] + 1e-9))


def _atr(high: pl.Series, low: pl.Series, close: pl.Series, window: int = 20) -> float:
    prev_close = close.shift()
    tr = pl.Series([
        max(high[i] - low[i], abs(high[i] - prev_close[i]), abs(low[i] - prev_close[i]))
        for i in range(len(high))
    ])
    return float(tr.rolling_mean(window).iloc[-1])


async def fetch_json(session: aiohttp.ClientSession, url: str, params: Dict | None = None):
    retries = 2
    for _ in range(retries):
        try:
            async with session.get(url, params=params, timeout=10) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception:
            await asyncio.sleep(1)
    raise RuntimeError(f"failed fetch {url}")


async def build_token_row(session: aiohttp.ClientSession, token: str) -> pl.DataFrame | None:
    symbol = token if token.endswith("USDT") else f"{token}USDT"
    # 120 mins of 1m candles gives room for indicators
    candles = await fetch_json(session, f"{API}/candles/{symbol}/1m", {"limit": 120})
    if not candles:
        return None
    # candles schema [ts, open, high, low, close, volume]
    df = pl.from_records(candles, schema=["ts", "open", "high", "low", "close", "volume"])
    df = df.with_columns([
        (pl.col("ts") * 1000).alias("dt"),
        pl.all().exclude("ts")
    ]).drop("ts")
    # Convert ms to s for grouping
    df = df.with_columns([(pl.col("dt") / 1000).cast(pl.Int64).alias("sec")])
    # Resample to 15-min bar ending now
    end_sec = df["sec"].max()
    start_window = end_sec - 15 * 60 + 1
    slice_15 = df.filter(pl.col("sec") >= start_window)
    if slice_15.is_empty():
        return None
    close_series = slice_15["close"]
    high_series = slice_15["high"]
    low_series = slice_15["low"]

    row = {
        "dt": _now_ms(),
        "close": float(close_series[-1]),
        "rsi14": _rsi(close_series),
        "boll_bw": _boll_bw(close_series),
        "atr_20": _atr(high_series, low_series, close_series),
    }

    # Funding rate delta 1h
    funding_js = await fetch_json(session, f"{API}/funding/{symbol}")
    if funding_js:
        row["funding_rate"] = funding_js.get("funding_rate", 0.0)
    # Orderbook imbalance
    ob_js = await fetch_json(session, f"{API}/orderbook/{symbol}")
    if ob_js:
        row["ob_imb"] = ob_js.get("imbalance", 0.0)

    return pl.DataFrame([row])


async def main() -> None:
    import yaml

    tokens: List[str] = []
    if REGISTRY.exists():
        with open(REGISTRY) as fp:
            reg = yaml.safe_load(fp) or {}
        for entry in reg.get("models", []):
            tokens.append(entry.get("token"))
    # Fallback: any *.parquet already present under FEATURES_DIR
    if not tokens:
        tokens = [p.stem for p in FEATURES_DIR.glob("*.parquet")]

    async with aiohttp.ClientSession() as sess:
        tasks = [build_token_row(sess, t) for t in tokens]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for token, res in zip(tokens, results):
        if isinstance(res, Exception) or res is None or res.is_empty():
            continue
        feat_path = FEATURES_DIR / f"{token}.parquet"
        try:
            if feat_path.exists():
                existing = pl.read_parquet(feat_path)
                # skip if we already have a row within last 10 min
                if (existing["dt"].max() > _now_ms() - 10 * 60 * 1000):
                    continue
                updated = pl.concat([existing, res])
                updated.write_parquet(feat_path, compression="zstd")
            else:
                res.write_parquet(feat_path, compression="zstd")
        except Exception as exc:
            print(f"[warn] failed to write features for {token}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
