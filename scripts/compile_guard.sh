#!/usr/bin/env bash
# Fast compile-time import guard for critical entrypoints
set -euo pipefail
PY=${PY:-python}
ENTRYPOINTS=(
  crypto-models/infer/infer.py
  crypto-models/train/train.py
)
for f in "${ENTRYPOINTS[@]}"; do
  echo "[compile] $f" >&2
  $PY -m py_compile "$f"
done
