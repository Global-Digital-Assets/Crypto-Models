"""Train all bucket and specialist models based on bucket_mapping.csv & registry.yaml."""
from __future__ import annotations

import subprocess
import yaml
import polars as pl
from utils import ROOT_DIR, BUCKET_MAPPING_FILE

REGISTRY_PATH = ROOT_DIR / "registry.yaml"


def main() -> None:  # noqa: D401
    reg = yaml.safe_load(REGISTRY_PATH.read_text())
    mapping = pl.read_csv(BUCKET_MAPPING_FILE)

    # 1. Train bucket models
    for bucket in mapping["bucket"].unique():  # type: ignore
        symbols = mapping.filter(pl.col("bucket") == bucket)["symbol"].to_list()  # type: ignore
        subprocess.run([
            "python", "train/train.py", bucket, "--mode", "long", "--tokens", ",".join(symbols)
        ])
        subprocess.run([
            "python", "train/train.py", bucket, "--mode", "short", "--tokens", ",".join(symbols)
        ])

    # 2. Train specialist models listed in registry.yaml
    for entry in reg["models"]:
        token = entry["token"]
        for mode in entry["modes"]:
            subprocess.run(["python", "train/train.py", token, "--mode", mode])


if __name__ == "__main__":
    main()
