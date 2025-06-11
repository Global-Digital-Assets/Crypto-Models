"""Periodic inference script – iterates over registry.yaml and writes JSON signals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

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


def rsi(series: pl.Series, window: int = 14) -> pl.Series:  # noqa: D401
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling_mean(window)
    loss = (-delta.clip(upper=0)).rolling_mean(window)
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def fetch_candles(symbol: str, tf: str = "15m", limit: int = 96) -> pl.DataFrame:  # noqa: D401
    url = f"{DATA_API_URL}/candles/{symbol}/{tf}?limit={limit}"
    js = SESSION.get(url, timeout=15).json()
    return pl.from_dicts(js["candles"])


def fetch_macro() -> pl.DataFrame:  # noqa: D401
    js = SESSION.get(f"{DATA_API_URL}/macro/latest", timeout=15).json()
    return pl.from_dicts([js])


def build_live_features(token: str, feature_cols: list[str]) -> pl.DataFrame:
    df = fetch_candles(token)
    df = df.sort("ts")
    df = df.with_columns(pl.from_epoch("ts", time_unit="s").alias("dt"))

    # technicals (mirror training)
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret_1"),
        rsi(pl.col("close"), 14).alias("rsi14"),
    ])
    ma20 = df["close"].rolling_mean(20)
    std20 = df["close"].rolling_std(20)
    df = df.with_columns((2 * std20 / ma20).alias("bb_width"))
    tr = pl.max([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ])
    df = df.with_columns(tr.alias("tr"))
    df = df.with_columns(df["tr"].rolling_mean(14).alias("atr14"))

    macro = fetch_macro()
    macro_dt = pl.from_epoch(macro["timestamp"], time_unit="s")[0]
    macro = macro.with_columns(pl.lit(macro_dt).alias("dt"))
    merged = df.join_asof(macro, on="dt", strategy="backward")

    latest_row = merged.tail(1)
    # ensure all feature_cols present; missing values filled with 0
    missing = [c for c in feature_cols if c not in latest_row.columns]
    for m in missing:
        latest_row = latest_row.with_columns(pl.lit(0).alias(m))

    return latest_row.select(feature_cols)


def main() -> None:  # noqa: D401
    args = parse_args()
    reg = load_registry()

    for entry in reg["models"]:
        token = entry["token"]
        bucket_df = pl.read_csv(BUCKET_MAPPING_FILE)
        bucket = bucket_df.filter(pl.col("symbol") == token)["bucket"].item()

        model_dir = ROOT_DIR / "models" / f"{token}_{entry['mode']}"
        if not model_dir.exists():
            model_dir = ROOT_DIR / "models" / f"{bucket}_{entry['mode']}"

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

        model = joblib.load(model_path)
        live_feats = build_live_features(token, feature_cols)
        if live_feats.is_empty():
            print(f"[warn] no features for {token}")
            continue
        proba = float(model.predict_proba(live_feats.to_numpy())[:, 1][0])
        infer_latency = (timestamp() - live_feats["dt"].to_numpy()[-1].astype(int)) * 1000
        latency_g.labels(model_id=f"{token}_{entry['mode']}").set(infer_latency)

        signal_type = (
            "buy" if (entry['mode'] == "long" and proba > 0.6) else
            "sell" if (entry['mode'] == "short" and proba > 0.6) else
            "neutral"
        )

        horizon_sec = parse_duration(args.horizon)
        edge_pct = meta.get("threshold_pct", 2.0)
        signal = {
            "timestamp": timestamp(),
            "model_id": f"{token}_{entry['mode']}_v{timestamp()//86400}",
            "signal": signal_type,
            "probability": round(proba, 3),
            "confidence": round(proba, 3),
            "edge": edge_pct,
            "bucket": bucket,
            "horizon": args.horizon,
            "target_price": round(float(live_feats.to_numpy()[0][feature_cols.index("close")] * (1 + edge_pct/100 if signal_type == "buy" else 1 - edge_pct/100)), 2) if "close" in feature_cols else None,
            "stop_loss": round(float(live_feats.to_numpy()[0][feature_cols.index("close")] * (1 - edge_pct/100 if signal_type == "buy" else 1 + edge_pct/100)), 2) if "close" in feature_cols else None,
            "expires_at": timestamp() + horizon_sec,
        }

        print(json.dumps(signal))

        if not args.dry_run:
            out_path = SIGNALS_DIR / f"{signal['model_id']}.json"
            dump_json(signal, out_path)
            signal_g.labels(model_id=signal["model_id"], signal=signal_type).inc()
            print(f"[+] wrote {out_path.relative_to(Path.cwd())}")

    if not args.dry_run:
        push_to_gateway(PUSHGATEWAY_URL, job="crypto-models-infer", registry=registry)


if __name__ == "__main__":
    main()
