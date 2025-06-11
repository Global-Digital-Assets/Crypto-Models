"""Entry-point for training a single token model using REAL feature parquet.

Usage
-----
python train.py BTCUSDT --mode long --window 30
"""
from __future__ import annotations

import argparse
import joblib
from pathlib import Path

import polars as pl
import numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from utils import FEATURES_DIR, MODELS_DIR, timestamp, dump_json


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser("crypto-models-train")
    p.add_argument("token", help="token symbol e.g. BTCUSDT or bucket id e.g. H-H")
    p.add_argument("--mode", choices=["long", "short"], default="long")
    p.add_argument("--window", type=int, default=30, help="training window in days")
    p.add_argument("--tokens", help="comma-separated member symbols when training a bucket model")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_features(token: str) -> pl.DataFrame:
    fp = FEATURES_DIR / f"{token}.parquet"
    if not fp.exists():
        raise SystemExit(f"Features parquet not found: {fp}. Run features/build_features.py first.")
    return pl.read_parquet(fp)


def build_target(df: pl.DataFrame, mode: str) -> pl.DataFrame:
    """Add binary target column based on **future** 15-min return to avoid leakage."""
    future_ret = (df["close"].shift(-1) / df["close"] - 1).alias("future_ret")
    df = df.with_columns(future_ret)
    if mode == "long":
        target = (pl.col("future_ret") > 0).cast(int).alias("y")
    else:  # short model
        target = (pl.col("future_ret") < 0).cast(int).alias("y")
    df = df.with_columns(target)
    return df.drop_nulls(subset=["y"])


def train_lightgbm(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    args = parse_args()

    df_list = []
    if args.tokens:
        symbols = args.tokens.split(",")
        for sym in symbols:
            df_list.append(load_features(sym))
        df = pl.concat(df_list, how="vertical")
    else:
        df = load_features(args.token)
    df = build_target(df, args.mode)
    df = df.sort("dt")

    # Keep last *window* days
    if args.window:
        cutoff = df["dt"].max() - pl.duration(days=args.window)
        df = df.filter(pl.col("dt") >= cutoff)

    # Split train/val (chronological) 80/20
    n = len(df)
    split_idx = int(n * 0.8)
    train_df, val_df = df[:split_idx], df[split_idx:]

    feature_cols = [c for c in df.columns if c not in {"dt", "future_ret", "y"}]
    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["y"].to_numpy()
    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df["y"].to_numpy()

    model = train_lightgbm(X_train, y_train)
    val_pred = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, val_pred)

    # Persist
    out_dir = MODELS_DIR / f"{args.token}_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pkl"
    joblib.dump(model, model_path)

    meta = {
        "trained_at": timestamp(),
        "token": args.token,
        "mode": args.mode,
        "feature_cols": feature_cols,
        "roc_auc": round(float(roc_auc), 4),
        "model_path": str(model_path),
    }
    dump_json(meta, out_dir / "metadata.json")
    print(f"[+] saved model → {model_path} | ROC-AUC={meta['roc_auc']}")


if __name__ == "__main__":
    main()
