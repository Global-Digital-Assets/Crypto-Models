"""Train all bucket and specialist models based on bucket_mapping.csv & registry.yaml."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import yaml
import polars as pl

# Resolve project root two levels up from this file ( .../crypto-models )
ROOT_DIR = Path(__file__).resolve().parents[1]
BUCKET_MAPPING_FILE = ROOT_DIR / "bucket_mapping.csv"

REGISTRY_PATH = ROOT_DIR / "registry.yaml"


def run_cmd(cmd: list[str]) -> None:
    """Run subprocess, log non-zero return codes but keep going."""
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print(f"[warn] cmd {' '.join(cmd[:4])} ... returned {result.returncode}")

def main() -> None:  # noqa: D401
    """Orchestrate bucket + specialist training with parallel workers.

    We build the list of *command arrays* first, then execute them with a
    ThreadPool (IO-bound) capped at 4 concurrent workers.  Each LightGBM
    training script (`train.py`) internally opens 2 threads, so 4 × 2 = 8
    logical cores→ full CPU utilisation on the VPS.
    """
    import concurrent.futures, itertools

    reg = yaml.safe_load(REGISTRY_PATH.read_text())
    mapping = pl.read_csv(BUCKET_MAPPING_FILE)

    cmds: list[list[str]] = []

    # 1. Bucket models (long + short, 15m & 60m)
    for bucket in mapping["bucket"].unique():  # type: ignore
        symbols = mapping.filter(pl.col("bucket") == bucket)["symbol"].to_list()  # type: ignore
        for horizon, mode in itertools.product((15, 60, 120), ("long", "short")):
            if horizon == 120 and mode == "long":
                continue  # 120-min horizon only useful for shorts
            cmds.append([
                sys.executable,
                str(ROOT_DIR / "train" / "train.py"),
                bucket,
                "--mode", mode,
                "--tokens", ",".join(symbols),
                "--horizon", str(horizon),
            ])

    # 2. Specialist models – only queue work if features exist AND model not already trained
    for entry in reg["models"]:
        token: str = entry["token"]
        feature_path = ROOT_DIR / "features" / f"{token}.parquet"
        for horizon, mode in itertools.product((15, 60, 120), entry["modes"]):
            if horizon == 120 and mode == "long":
                continue
            model_dir = ROOT_DIR / "models" / f"{token}_{mode}_{horizon}"
            # Skip if model already exists in models/ or features parquet missing
            if model_dir.exists():
                continue
            if not feature_path.exists():
                print(f"[skip] {token} – feature parquet missing → skipping {mode}/{horizon}m")
                continue
            cmds.append([
                sys.executable,
                str(ROOT_DIR / "train" / "train.py"),
                token,
                "--mode", mode,
                "--horizon", str(horizon),
            ])

    # Execute with a pool of 4 workers
    max_workers = 4

    def _worker(cmd: list[str]):
        """Wrapper for subprocess plus logging."""
        result = subprocess.run(cmd, cwd=str(ROOT_DIR))
        if result.returncode != 0:
            print(f"[warn] cmd {' '.join(cmd[:4])} … returned {result.returncode}")
        return result.returncode

    print(f"[info] training jobs queued: {len(cmds)}; running with {max_workers} parallel workers …")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_worker, cmds))

    print("[✓] All training jobs finished")


if __name__ == "__main__":
    main()
