"""Nightly job: compute liquidity/vol buckets & update bucket_mapping.csv.

Reads 30d candles + volume via Data-API.
"""
from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import List, Tuple

import aiohttp
import polars as pl
import yaml

from utils import BUCKET_MAPPING_FILE, ROOT_DIR

CONFIG = yaml.safe_load((ROOT_DIR / "bucket_config.yaml").read_text())
V_BREAKS = CONFIG["vol_breaks"]
VOL_BREAK_1, VOL_BREAK_2 = V_BREAKS
Q_BREAKS = CONFIG["volume_breaks"]
VOLU_BREAK_1, VOLU_BREAK_2 = Q_BREAKS
API_URL = "http://127.0.0.1:8001"


async def fetch_metrics(session: aiohttp.ClientSession, symbol: str) -> Tuple[float, float]:  # noqa: D401
    url = f"{API_URL}/candles/{symbol}/1d?limit=30"
    js = await (await session.get(url)).json()
    df = pl.from_dicts(js["candles"])
    turnover = (df["close"] * df["volume"]).mean()
    vol = (df["close"].pct_change(1).std())
    return float(turnover), float(vol)


async def main(symbols: List[str]):  # noqa: D401
    if symbols == ["ALL"] and BUCKET_MAPPING_FILE.exists():
        symbols = pl.read_csv(BUCKET_MAPPING_FILE)["symbol"].unique().to_list()  # type: ignore
    async with aiohttp.ClientSession() as s:
        metrics = await asyncio.gather(*(fetch_metrics(s, sym) for sym in symbols))

    rows = []
    for sym, (turnover, vol) in zip(symbols, metrics):
        liq = "H" if turnover > VOLU_BREAK_1 else "M" if turnover > VOLU_BREAK_2 else "L"
        vol_cat = "L" if vol < VOL_BREAK_1 else "M" if vol < VOL_BREAK_2 else "H"
        bucket = f"{liq}-{vol_cat}"
        rows.append((sym, bucket, turnover, vol))

    with BUCKET_MAPPING_FILE.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["symbol", "bucket", "turnover", "vol"])
        w.writerows(rows)
    print(f"[+] bucket mapping updated → {BUCKET_MAPPING_FILE}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--symbols", required=True, help="comma-separated symbols or ALL")
    args = p.parse_args()
    sym_list = args.symbols.split(",") if args.symbols != "ALL" else ["ALL"]
    asyncio.run(main(sym_list))
