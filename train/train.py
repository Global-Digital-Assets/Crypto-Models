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
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

# Ensure project root is on sys.path so `utils` can be imported when this
# script is launched from a sub-directory subprocess.
from pathlib import Path
import sys as _sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in _sys.path:
    _sys.path.append(str(ROOT_DIR))
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
    p.add_argument("--horizon", type=int, default=60, help="forecast horizon in minutes (e.g. 60)")
    p.add_argument("--target-pct", type=float, default=1.0, help="absolute percentage move defining a positive label, e.g. 1.0 for 1%")
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced", help="LightGBM class_weight parameter")
    p.add_argument("--force", action="store_true", help="train even if positive samples below threshold")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_features(token: str) -> pl.DataFrame:
    fp = FEATURES_DIR / f"{token}.parquet"
    if not fp.exists():
        raise SystemExit(f"Features parquet not found: {fp}. Run features/build_features.py first.")
    return pl.read_parquet(fp)


def build_target(df: pl.DataFrame, mode: str, horizon: int, target_pct: float) -> pl.DataFrame:
    """Add binary target column based on *future* return over `horizon` minutes.

    Parameters
    ----------
    df : pl.DataFrame
        Feature dataframe sorted by `dt` 15-minute timestamps.
    mode : str
        "long" or "short" model label definition.
    horizon : int
        Forecast horizon in minutes, must be positive. Assumes 15-minute bar spacing.
    target_pct : float
        Threshold move (absolute percent) defining a positive label. e.g. 1.0 = 1 %.
    """
    # Number of 15-minute bars to look ahead
    shift_steps = max(1, int(horizon // 15))
    future_ret = (df["close"].shift(-shift_steps) / df["close"] - 1).alias("future_ret")
    df = df.with_columns(future_ret)
    if mode == "short" and target_pct <= 0:
        # adaptive threshold: 0.5 * ATR percent
        atr_pct = (pl.col("atr_20") / pl.col("close")).fill_null(strategy="forward")
        thresh_series = (atr_pct * 0.5).alias("dyn_thresh")
        df = df.with_columns(thresh_series)
        target = (pl.col("future_ret") <= -pl.col("dyn_thresh")).cast(int).alias("y")
    else:
        thresh = target_pct / 100
        if mode == "long":
            target = (pl.col("future_ret") >= thresh).cast(int).alias("y")
        else:
            target = (pl.col("future_ret") <= -thresh).cast(int).alias("y")
    df = df.with_columns(target)
    return df.drop_nulls(subset=["y"])


def train_lightgbm(X: np.ndarray, y: np.ndarray, class_weight: str) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.08,
        max_depth=8,
        num_leaves=127,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=8,
        force_row_wise=True,
        class_weight=(class_weight if class_weight == "balanced" else None),
        verbose=-1,
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
    df = build_target(df, args.mode, args.horizon, args.target_pct)
    df = df.sort("dt")

    # Keep last *window* days
    if args.window:
        cutoff = df["dt"].max() - pl.duration(days=args.window)
        df = df.filter(pl.col("dt") >= cutoff)

    # Split train/val (chronological) 80/20
    n = len(df)
    split_idx = int(n * 0.8)
    train_df, val_df = df[:split_idx], df[split_idx:]

    # --- Positive sample quality gate ---
    pos_count = int(train_df["y"].sum())
    pos_ratio = pos_count / len(train_df) if len(train_df) else 0
    if (not args.force) and (pos_count < 100 and pos_ratio < 0.01):
        print(
            f"[warn] {args.token} positives={pos_count} (ratio={pos_ratio:.2%}) below 100 or 1% – skipping model."
        )
        return
    if len(train_df) < 200:
        print(f"[warn] {args.token} not enough rows ({len(train_df)}) – skipping model.")
        return

    # Keep only numeric scalar columns to avoid LightGBM errors
    numeric_tps = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    }
    feature_cols = [
        c for c, tp in zip(df.columns, df.dtypes)
        if c not in {"dt", "future_ret", "y"} and tp in numeric_tps
    ]
    # Cast features to float to avoid sequence/heterogeneous errors
    train_features_df = train_df.select(feature_cols).with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in feature_cols]
    )
    val_features_df = val_df.select(feature_cols).with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in feature_cols]
    )

    # Prepare output directory early
    out_dir = MODELS_DIR / f"{args.token}_{args.mode}_{args.horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_features_df.to_numpy()
    y_train = train_df["y"].to_numpy()
    X_val = val_features_df.to_numpy()
    y_val = val_df["y"].to_numpy()
    if len(feature_cols) == 0:
        raise SystemExit("No numeric features available after filtering – abort.")

    model = train_lightgbm(X_train, y_train, args.class_weight)
    val_pred_raw = model.predict_proba(X_val)[:, 1]

    # --- Probability calibration (isotonic) ---
    val_pos = int(y_val.sum())
    calibrator = None
    if val_pos >= 50:
        from sklearn.isotonic import IsotonicRegression
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(val_pred_raw, y_val)
        val_pred = calibrator.predict(val_pred_raw)
        joblib.dump(calibrator, out_dir / "calibrator.pkl")
    else:
        # Too few positives – use raw LightGBM probability
        val_pred = val_pred_raw

    # Handle ROC-AUC calculation with graceful fallback
    try:
        roc_auc = roc_auc_score(y_val, val_pred)
    except ValueError as e:
        if "Only one class present" in str(e):
            print(f"[warn] Single class in validation set for {args.token}_{args.mode}_{args.horizon} - using accuracy fallback")
            roc_auc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
        else:
            raise e

    # Persist
    # out_dir already created above
    model_path = out_dir / "model.pkl"
    joblib.dump(model, model_path)

    meta = {
        "trained_at": timestamp(),
        "calibrated": bool(calibrator),
        "token": args.token,
        "mode": args.mode,
        "horizon_min": args.horizon,
        "target_pct": args.target_pct,
        "feature_cols": feature_cols,
        "roc_auc": round(float(roc_auc), 4),
        "model_path": str(model_path),
    }
    dump_json(meta, out_dir / "metadata.json")
    print(f"[+] saved model → {model_path} | ROC-AUC={meta['roc_auc']}")


if __name__ == "__main__":
    main()
