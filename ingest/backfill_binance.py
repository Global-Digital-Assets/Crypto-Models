"""Backfill historical OHLCV data directly from Binance REST API.

This script bypasses the local Data-API when older candles are missing.
It downloads raw candles *client-side* and persists them under
`/data/<SYMBOL>_<TF>.parquet` so the rest of the pipeline can proceed
unmodified.

Usage (examples)
----------------
python ingest/backfill_binance.py --symbols BTCUSDT,ETHUSDT --tf 15m --days 180
python ingest/backfill_binance.py --symbols BTCUSDT,ETHUSDT --tf 1m  --days 30

It respects the 1 500 row / request limit enforced by the Binance public
endpoint and fetches data in reverse-chronological slices to avoid
missing most-recent candles.

Environment
-----------
BINANCE_API_URL   Override base URL (default: https://api.binance.com)
OMP_NUM_THREADS   Thread caps inherited from global env (8-core tuning)
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import aiohttp
import polars as pl

# Local project utils (DATA_DIR constant)
from utils import DATA_DIR  # type: ignore

BINANCE_API_URL = os.getenv("BINANCE_API_URL", "https://api.binance.com")
MAX_ROWS_PER_CALL = 1500  # Binance hard limit
INTERVAL_TO_MS = {
    "1m": 60_000,
    "15m": 15 * 60_000,
}


async def req(session, symbol, interval, start_ms, end_ms):
    url = f"{BINANCE_API_URL}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": MAX_ROWS_PER_CALL,
    }
    while True:
        async with session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                # light pacing to stay under 6k weight/min (20 req * 0.15s = 3s)
                await asyncio.sleep(0.15)
                return data
            if resp.status == 429:
                # rate-limit hit – wait and retry
                await asyncio.sleep(65)
                continue
            raise RuntimeError(f"HTTP {resp.status}: {await resp.text()}")


async def fetch_symbol(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    days: int,
) -> pl.DataFrame:
    """Full backfill for a symbol/interval covering *days* trailing days."""
    step_ms = INTERVAL_TO_MS[interval] * MAX_ROWS_PER_CALL
    end_ts = datetime.now(tz=timezone.utc)
    start_ts = end_ts - timedelta(days=days)

    end_ms = int(end_ts.timestamp() * 1000)
    start_ms = int(start_ts.timestamp() * 1000)

    tasks = []
    cur_end = end_ms
    while cur_end > start_ms:
        cur_start = max(start_ms, cur_end - step_ms)
        tasks.append(req(session, symbol, interval, cur_start, cur_end - 1))
        cur_end = cur_start

    # Run fetches with limited concurrency
    results: list[list[list]] = []
    sem = asyncio.Semaphore(4)

    async def worker(idx: int, coro):
        async with sem:
            data = await coro
            results.append(data)
            if (idx + 1) % 50 == 0:
                print(f"[{symbol}-{interval}] fetched {idx+1}/{len(tasks)} chunks")

    await asyncio.gather(*(worker(i, t) for i, t in enumerate(tasks)))

    # Flatten & sort oldest -> newest
    all_rows: list[list] = [row for chunk in reversed(results) for row in chunk]

    if not all_rows:
        raise RuntimeError(f"No data fetched for {symbol} {interval}")

    # Map to schema
    df = pl.DataFrame(
        {
            "ts": [r[0] for r in all_rows],
            "open": [float(r[1]) for r in all_rows],
            "high": [float(r[2]) for r in all_rows],
            "low": [float(r[3]) for r in all_rows],
            "close": [float(r[4]) for r in all_rows],
            "vol": [float(r[5]) for r in all_rows],
        }
    )
    return df


async def main(symbols: List[str], days: int, tf: str) -> None:
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        dfs = []
        for sym in symbols:
            print(f"=== Fetching {sym} {tf} ===")
            df = await fetch_symbol(session, sym, tf, days)
            dfs.append(df)
            out_path = DATA_DIR / f"{sym}_{tf}.parquet"
            df.write_parquet(out_path, compression="zstd")
            span = len(df) / (24*60 if tf=="1m" else 24*4)
            print(f"[+] {sym} {tf}: {len(df)} rows ≈ {span:.1f} days -> {out_path}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("crypto-models-binance-backfill")
    parser.add_argument("--symbols", required=True, help="comma-separated e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--tf", choices=["1m", "15m"], default="15m")
    args = parser.parse_args()

    asyncio.run(main(args.symbols.split(","), days=args.days, tf=args.tf))
