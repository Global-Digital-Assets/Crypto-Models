"""Quick percentile backtest for a single token model.

Usage:
    python backtest.py --model PEPE_short --percentiles 90,95,97,99 --tp 6.0 --sl 1.5

Assumptions:
* `models/<model>/metadata.json` contains horizon_min & feature_cols.
* Features parquet exists under `features/<token>.parquet` (same path used by train).
* Short model: positive class = price decrease; Long model: price increase.
* Profit calculation: percentage move over `horizon_min`, capped by TP/SL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from utils import FEATURES_DIR, MODELS_DIR


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("crypto-models-backtest")
    p.add_argument("--model", required=True, help="model name e.g. PEPE_short (folder under models/)")
    p.add_argument("--percentiles", default="90,95,97,99", help="comma-separated percentile cutoffs")
    p.add_argument("--tp", type=float, default=6.0, help="take-profit in % (absolute)")
    p.add_argument("--sl", type=float, default=1.5, help="stop-loss in % (absolute)")
    p.add_argument("--window", type=int, default=30, help="backtest window in days (0 = all)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def load_metadata(model_dir: Path) -> dict:
    md_path = model_dir / "metadata.json"
    if not md_path.exists():
        raise SystemExit(f"metadata not found: {md_path}")
    return json.loads(md_path.read_text())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    args = parse_args()
    model_dir = MODELS_DIR / args.model
    if not model_dir.exists():
        raise SystemExit(f"model dir not found: {model_dir}")

    meta = load_metadata(model_dir)
    token: str = meta["token"]
    mode: str = meta["mode"]
    horizon_min: int = int(meta.get("horizon_min", 60))
    feature_cols: List[str] = meta["feature_cols"]

    # Load data and build future return for the horizon
    df = pl.read_parquet(FEATURES_DIR / f"{token}.parquet").sort("dt")
    if args.window:
        cutoff = df["dt"].max() - pl.duration(days=args.window)
        df = df.filter(pl.col("dt") >= cutoff)

    shift_steps = max(1, horizon_min // 15)
    df = df.with_columns(
        ((pl.col("close").shift(-shift_steps) / pl.col("close") - 1)).alias("future_ret")
    )
    df = df.drop_nulls(subset=["future_ret"])

    # Prepare features
    features_df = df.select(feature_cols).with_columns([
        pl.col(c).cast(pl.Float64, strict=False) for c in feature_cols
    ])
    X = features_df.to_numpy()

    # Load model & predict probabilities
    model = joblib.load(model_dir / "model.pkl")
    proba = model.predict_proba(X)[:, 1]

    # Calculate percentile thresholds
    perc_values = [int(p) for p in args.percentiles.split(",")]
    results = []

    for p in perc_values:
        thresh = np.percentile(proba, p)
        mask = proba >= thresh
        if mask.sum() == 0:
            avg_pnl = 0.0
        else:
            fut = df["future_ret"].to_numpy()[mask]
            raw_pnl = (-fut) if mode == "short" else fut
            pnl = clamp(raw_pnl, -args.sl / 100, args.tp / 100)
            avg_pnl = pnl.mean() * 100  # to percentage
        results.append((p, mask.sum(), avg_pnl))

    # AUC (optional sanity-check)
    y = (df["future_ret"] < 0).to_numpy().astype(int) if mode == "short" else (df["future_ret"] > 0).to_numpy().astype(int)
    auc = roc_auc_score(y, proba)

    # Output
    print(f"Backtest — {args.model} | window={args.window}d | horizon={horizon_min}m | AUC={auc:.4f}\n")
    print("Pctile  Trades  Avg_PnL_%")
    for p, n_trades, avg in results:
        print(f"{p:>6}  {n_trades:>6}  {avg:+9.4f}")

    # Simple deployment rule
    p99 = next((avg for pct, _, avg in results if pct == 99), 0.0)
    if p99 > 0.35:
        print("\n✅ 99th percentile meets profit target → ready to deploy.")
        (model_dir / "DEPLOY_READY").touch()
    else:
        print("\n❌ Profit target not met.")


if __name__ == "__main__":
    main()
