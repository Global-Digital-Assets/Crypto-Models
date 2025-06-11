"""Produce signals for all tokens in registry via specialist or bucket model."""
from __future__ import annotations

import subprocess
import sys
import time
import yaml
from utils import ROOT_DIR

REGISTRY_PATH = ROOT_DIR / "registry.yaml"
WATCHDOG_FILE = ROOT_DIR / "infer_alive.txt"


def main() -> None:  # noqa: D401
    reg = yaml.safe_load(REGISTRY_PATH.read_text())
    subprocess.run([sys.executable, "infer/infer.py"])
    WATCHDOG_FILE.write_text(str(time.time()))


if __name__ == "__main__":
    main()
