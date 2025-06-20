#!/usr/bin/env python3
"""Crypto-Models – append the latest feature row for every token.
Designed to run once per minute via systemd timer.  
Dependencies: pandas, numpy, requests, polars.
The script avoids Polars Series arithmetic – all indicator maths is NumPy.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT_DIR / "features"
REGISTRY_PATH = ROOT_DIR / "registry.yaml"
DATA_API = os.environ.get("DATA_API_URL", "http://localhost:8000").rstrip("/")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("feature_refresh")

# ---------------------------------------------------------------------------
# Indicator helpers (pure-numpy) -------------------------------------------
# ---------------------------------------------------------------------------

def rsi14(close: np.ndarray) -> float:
    if close.size < 15:
        return 50.0
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0)
    loss = np.where(diff < 0, -diff, 0)
    avg_gain = gain[-14:].mean()
    avg_loss = loss[-14:].mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))

def boll_bw(close: np.ndarray) -> float:
    if close.size < 20:
        return 0.0
    window = close[-20:]
    mean = window.mean()
    std = window.std(ddof=0)
    if mean == 0:
        return 0.0
    return float((4 * std) / mean)  # (upper-lower)/mean, width = 4*std

def atr14(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
    if close.size < 2:
        return 0.0
    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1]),
    ])
    if tr.size < 14:
        return float(tr.mean())
    return float(tr[-14:].mean())

# ---------------------------------------------------------------------------
# Core functions ------------------------------------------------------------
# ---------------------------------------------------------------------------

def fetch_json(path: str, params: Dict | None = None, *, retries: int = 3, base_delay: float = 0.2) -> Dict | None:
    """GET a JSON endpoint with primitive exponential-back-off.

    Args:
        path: endpoint path part (joined to DATA_API).
        params: optional query params dict.
        retries: number of attempts before giving up.
        base_delay: starting back-off delay in seconds.

    Returns:
        Parsed JSON dict or ``None`` on failure after retries.
    """
    import random
    url = f"{DATA_API}/{path.lstrip('/')}"
    attempt = 0
    while attempt < retries:
        try:
            resp = requests.get(url, params=params, timeout=8)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            attempt += 1
            if attempt >= retries:
                log.warning("API %s failed after %d tries: %s", url, attempt, exc)
                return None
            # jittered exponential back-off:  base_delay * 2**(attempt-1) ± 25 %
            delay = base_delay * (2 ** (attempt - 1))
            delay *= random.uniform(0.75, 1.25)
            time.sleep(delay)

def build_row(token: str) -> pl.DataFrame | None:
    symbol = token if token.endswith("USDT") else f"{token}USDT"

    candles = fetch_json(f"candles/{symbol}/1m", params={"limit": 120})
    if (not candles) or ("candles" not in candles) or (not candles["candles"]):
        log.error("%s – no candle data", symbol)
        return None
    
    # Convert list of dicts to numpy arrays
    candle_list = candles["candles"]
    if len(candle_list) < 20:
        log.warning("%s – insufficient candles: %d", symbol, len(candle_list))
        return None
    
    # Extract fields from dict format
    ts = np.array([c["ts"] for c in candle_list], dtype=np.int64)
    open_ = np.array([c["open"] for c in candle_list], dtype=float)
    high = np.array([c["high"] for c in candle_list], dtype=float)
    low = np.array([c["low"] for c in candle_list], dtype=float)
    close = np.array([c["close"] for c in candle_list], dtype=float)
    vol = np.array([c["vol"] for c in candle_list], dtype=float)

    # limit to last 15-minute window
    end_ms = ts[-1]
    start_ms = end_ms - 15*60*1000
    mask = ts >= start_ms
    close_win = close[mask]
    high_win  = high[mask]
    low_win   = low[mask]
    vol_win   = vol[mask]

    row = {
        "dt": dt.datetime.fromtimestamp(ts[-1] / 1000, tz=dt.timezone.utc).replace(tzinfo=None),
        "close": float(close_win[-1]),
        "open": float(open_[mask][0]),
        "high": float(high_win.max()),
        "low":  float(low_win.min()),
        "volume": float(vol_win.sum()),
        # indicators
        "rsi14": rsi14(close),
        "bb_width": boll_bw(close),
        "atr14": atr14(high, low, close),
    }

    # optional feeds --------------------------------------------------------
    funding = fetch_json(f"funding/{symbol}")
    if funding and "funding_rate" in funding:
        row["funding_rate"] = funding["funding_rate"]
    else:
        row["funding_rate"] = 0.0

    ob = fetch_json(f"orderbook/{symbol}")
    row["ob_imb"] = ob.get("imbalance", 0.0) if ob else 0.0

    return pl.DataFrame([row])


def align_and_append(token: str, new_df: pl.DataFrame) -> bool:
    try:
        path = FEATURES_DIR / f"{token}.parquet"
        if path.exists():
            existing = pl.read_parquet(path)
            
            # Cast datetime to match existing schema
            if "dt" in new_df.columns and "dt" in existing.columns:
                existing_dt_type = existing.schema["dt"]
                new_df = new_df.with_columns(pl.col("dt").cast(existing_dt_type))
            
            # Align schemas - handle null/float mismatches
            for c in existing.columns:
                if c not in new_df.columns:
                    req_dtype = existing.schema[c]
                    default = pl.lit(None)
                    if req_dtype == pl.Float64:
                        default = default.cast(pl.Float64)
                    new_df = new_df.with_columns(default.alias(c))
                elif c in new_df.columns:
                    # Cast to match existing type, handling Null -> Float conversions
                    existing_type = existing.schema[c]
                    new_type = new_df.schema[c]
                    if existing_type != new_type:
                        try:
                            new_df = new_df.with_columns(pl.col(c).cast(existing_type))
                        except:
                            # If cast fails, use compatible type
                            if str(existing_type) == "Null":
                                # Existing is all nulls, keep new type
                                existing = existing.with_columns(pl.col(c).cast(new_type))
                            else:
                                # Force to float64 as safe fallback
                                new_df = new_df.with_columns(pl.col(c).cast(pl.Float64))
                                existing = existing.with_columns(pl.col(c).cast(pl.Float64))
            
            for c in new_df.columns:
                if c not in existing.columns:
                    req_dtype = new_df.schema[c]
                    default = pl.lit(None)
                    if req_dtype == pl.Float64:
                        default = default.cast(pl.Float64)
                    existing = existing.with_columns(default.alias(c))
            
            new_df = new_df.select(existing.columns)
            combined = pl.concat([existing, new_df])
            # drop duplicate timestamps (keep the newest row)
            combined = combined.unique(subset=["dt"], keep="last")
        else:
            combined = new_df
        # ensure duplicate timestamps are removed even for new files
        combined = combined.unique(subset=["dt"], keep="last")
        combined.write_parquet(path, compression="zstd")
        return True
    except Exception as e:
        log.error("%s – append failed: %s", token, e)
        return False


def token_list() -> List[str]:
    if REGISTRY_PATH.exists():
        import yaml
        with open(REGISTRY_PATH) as fp:
            reg = yaml.safe_load(fp) or {}
        return [m["token"] for m in reg.get("models", [])]
    return [p.stem for p in FEATURES_DIR.glob("*.parquet")]

# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    single = sys.argv[1] if len(sys.argv) > 1 else None
    tokens = [single] if single else token_list()
    if not tokens:
        log.error("No tokens to process – aborting.")
        sys.exit(1)

    ok = fail = 0
    for tkn in tokens:
        df = build_row(tkn)
        if df is None or df.is_empty():
            fail += 1
            continue
        if align_and_append(tkn, df):
            ok += 1
    log.info("Refresh complete – %d success, %d fail", ok, fail)
    sys.exit(0 if ok else 1)  # any success => systemd "green"
