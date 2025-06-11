"""Download raw market data from the local Data-API and persist as Parquet.

The script follows the REST blueprint provided by the Data-Service.
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib
from typing import List
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import aiohttp
import polars as pl

from utils import DATA_DIR, DATA_API_URL

API_URL = DATA_API_URL
ONE_DAY_1m = 1440
ONE_DAY_15m = 96 * 2  # 192 entries


async def fetch_candles(session: aiohttp.ClientSession, symbol: str, tf: str, days: int) -> pl.DataFrame:  # noqa: D401
    limit = ONE_DAY_1m if tf == "1m" else ONE_DAY_15m
    url = f"{API_URL}/candles/{symbol}/{tf}?limit={limit}"
    acc = []
    while len(acc) < days * limit:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Data-API error {resp.status} → {url}")
            js = await resp.json()
            acc.extend(js["candles"])
            # next page (paginated backwards)
            url = f"{API_URL}/candles/{symbol}/{tf}?limit={limit}&before_ts={js['candles'][0]['ts']-1}"
    return pl.DataFrame(acc)


async def main(symbols: List[str], days: int = 30, tf: str = "1m") -> None:  # noqa: D401
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        dfs = await asyncio.gather(*(fetch_candles(session, sym, tf, days) for sym in symbols))

    for sym, df in zip(symbols, dfs):
        out_path = DATA_DIR / f"{sym}_{tf}.parquet"
        df.write_parquet(out_path, compression="zstd")
        print(f"[+] wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("crypto-models-ingest")
    parser.add_argument("--symbols", required=True, help="comma-separated symbols e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--tf", choices=["1m", "15m"], default="1m")
    args = parser.parse_args()

    asyncio.run(main(args.symbols.split(","), days=args.days, tf=args.tf))
