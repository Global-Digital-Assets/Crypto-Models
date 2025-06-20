"""Periodic inference script – iterates over registry.yaml and writes JSON signals."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import polars as pl
import yaml
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from utils import (
    SIGNALS_DIR,
    timestamp,
    dump_json,
    ROOT_DIR,
    BUCKET_MAPPING_FILE,
    parse_duration,
    DATA_API_URL,
    PUSHGATEWAY_URL,
    http_session,
)

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "registry.yaml"

# HTTP session with retry/backoff
SESSION = http_session()

registry = CollectorRegistry()
latency_g = Gauge("cm_infer_latency_ms", "Inference latency", ["model_id"], registry=registry)
signal_g = Gauge("cm_signal_count", "Signal counter", ["model_id", "signal"], registry=registry)


def parse_args() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser("crypto-models-infer")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--horizon", default="4h", help="prediction horizon e.g. 4h")
    return p.parse_args()


def load_registry() -> Dict:
    return yaml.safe_load(REGISTRY_PATH.read_text())


def rsi(expr: pl.Expr, window: int = 14) -> pl.Expr:  # noqa: D401
    """Vectorised RSI implemented with Polars expressions (stable for both Series & Expr)."""
    delta = expr - expr.shift(1)
    gain = pl.when(delta > 0).then(delta).otherwise(0).rolling_mean(window)
    loss = pl.when(delta < 0).then(-delta).otherwise(0).rolling_mean(window)
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def fetch_candles(symbol: str, tf: str = "15m", limit: int = 96) -> pl.DataFrame:  # noqa: D401
    url = f"{DATA_API_URL}/candles/{symbol}/{tf}?limit={limit}"
    js = SESSION.get(url, timeout=15).json()
    return pl.from_dicts(js["candles"])


def fetch_macro() -> pl.DataFrame:  # noqa: D401
    js = SESSION.get(f"{DATA_API_URL}/macro/latest", timeout=15).json()
    # Flatten nested macro structure: {"DXY": {"ts": ..., "value": ...}, "VIX": {...}}
    flattened = {}
    for key, data in js.items():
        if isinstance(data, dict) and 'ts' in data and 'value' in data:
            flattened[key.lower()] = data['value']
            if 'ts' not in flattened:  # use first timestamp found
                flattened['ts'] = data['ts']
    
    if not flattened:  # fallback if no valid data
        flattened = {'ts': int(time.time() * 1000), 'dxy': 100.0, 'vix': 20.0}
        
    return pl.DataFrame([flattened])


def _pick_ts_col(df: pl.DataFrame, candidates: tuple[str, ...] = ("ts", "timestamp", "time", "open_time", "openTime")) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("no timestamp column found")


def build_live_features(token: str, feature_cols: list[str]) -> Tuple[pl.DataFrame, int]:
    df = fetch_candles(token)
    ts_col = _pick_ts_col(df)
    if ts_col != "ts":
        df = df.rename({ts_col: "ts"})
    df = df.sort("ts")
    # Auto-detect timestamp unit (ms vs s)
    first_ts = df["ts"].item(0)
    time_unit = "ms" if first_ts > 1e12 else "s"
    df = df.with_columns(pl.from_epoch("ts", time_unit=time_unit).alias("dt"))
    df = df.with_columns(pl.col("dt").dt.cast_time_unit("ms"))

    # technicals (mirror training)
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret_1"),
        rsi(pl.col("close"), 14).alias("rsi14"),
    ])
    df = df.with_columns([
        (2 * pl.col("close").rolling_std(20) / pl.col("close").rolling_mean(20)).alias("bb_width")
    ])
    tr_expr = pl.max_horizontal([
        (pl.col("high") - pl.col("low")),
        (pl.col("high") - pl.col("close").shift(1)).abs(),
        (pl.col("low")  - pl.col("close").shift(1)).abs(),
    ])
    df = df.with_columns([
        tr_expr.alias("tr"),
        tr_expr.rolling_mean(14).alias("atr14"),
    ])

    macro = fetch_macro()
    m_ts_col = None
    for cand in ("ts", "timestamp", "time"):
        if cand in macro.columns:
            m_ts_col = cand
            break
    if m_ts_col is None:
        macro = macro.with_columns(pl.lit(time.time()).alias("ts"))
        m_ts_col = "ts"
    # Auto-detect timestamp unit for macro too
    first_macro_ts = macro[m_ts_col].item(0)
    macro_unit = "ms" if first_macro_ts > 1e12 else "s"
    macro = macro.with_columns(pl.from_epoch(m_ts_col, time_unit=macro_unit).alias("dt"))
    macro = macro.with_columns(pl.col("dt").dt.cast_time_unit("ms"))
    merged = df.join_asof(macro, on="dt", strategy="backward")

    latest_row = merged.tail(1)
     
     # ensure all feature_cols present; missing values filled with 0
    missing = [c for c in feature_cols if c not in latest_row.columns]
    for m in missing:
        latest_row = latest_row.with_columns(pl.lit(0).alias(m))

    latest_ts_ms = latest_row["dt"].cast(pl.Int64).item()
    return latest_row.select(feature_cols), latest_ts_ms


def main() -> None:  # noqa: D401
    args = parse_args()
    reg = load_registry()

    for entry in reg["models"]:
        token = entry["token"]
        bucket_df = pl.read_csv(BUCKET_MAPPING_FILE)
        bucket = bucket_df.filter(pl.col("symbol") == token)["bucket"].item()

        for mode in entry["modes"]:
            model_dir = ROOT_DIR / "models" / f"{token}_{mode}"
            if not model_dir.exists():
                model_dir = ROOT_DIR / "models" / f"{bucket}_{mode}"

            model_path = model_dir / "model.pkl"
            meta_path = model_dir / "metadata.json"
            if not model_path.exists() or not meta_path.exists():
                print(f"[warn] model artifacts missing: {model_dir}")
                continue

            with open(meta_path) as fp:
                meta = json.load(fp)
            feature_cols = meta.get("feature_cols")
            if not feature_cols:
                print(f"[warn] feature_cols missing in metadata for {model_dir}")
                continue

            try:
                model = joblib.load(model_path)
                live_feats, latest_ts_ms = build_live_features(token, feature_cols)
                if live_feats.is_empty():
                    print(f"[warn] no features for {token}")
                    continue
                    
                try:
                    pred_proba = model.predict_proba(live_feats.to_numpy())
                    if hasattr(pred_proba, 'ndim') and pred_proba.ndim == 2 and pred_proba.shape[1] > 1:
                        proba = float(pred_proba[0, 1])  # binary classification
                    else:
                        proba = float(pred_proba[0])  # single probability
                except Exception as pred_err:
                    print(f"[debug] {token}_{mode} pred error: {pred_err}")
                    print(f"[debug] pred_proba type: {type(pred_proba) if 'pred_proba' in locals() else 'undefined'}")
                    print(f"[debug] pred_proba value: {pred_proba if 'pred_proba' in locals() else 'undefined'}")
                    continue
                    
                try:
                    infer_latency = int(timestamp()*1000 - latest_ts_ms)
                except Exception as lat_err:
                    print(f"[debug] {token}_{mode} latency error: {lat_err}, latest_ts_ms={latest_ts_ms}, type={type(latest_ts_ms)}")
                    infer_latency = 0

                latency_g.labels(model_id=f"{token}_{mode}").set(infer_latency)

                signal_type = (
                    "buy" if (mode == "long" and proba > 0.6) else
                    "sell" if (mode == "short" and proba > 0.6) else
                    "neutral"
                )

                signal = {
                    "timestamp": int(timestamp()),
                    "model_id": f"{token}_{mode}_v{str(meta['trained_at'])[:10]}",
                    "signal": signal_type,
                    "probability": proba,
                    "expires_at": int(timestamp() + 3600),
                }
                out_path = SIGNALS_DIR / f"{token}_{mode}.json"
                dump_json(signal, out_path)
                signal_g.labels(model_id=signal["model_id"], signal=signal_type).inc()
                print(f"[+] {signal_type} {token}({mode}) → {proba:.2f}")
            except Exception as e:
                print(f"[error] {token}_{mode}: {e}")
                continue

    if not args.dry_run:
        push_to_gateway(PUSHGATEWAY_URL, job="crypto-models-infer", registry=registry)


if __name__ == "__main__":
    main()
