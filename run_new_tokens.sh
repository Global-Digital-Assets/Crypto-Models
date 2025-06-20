#!/usr/bin/env bash
set -euo pipefail
TOKENS="MAGICUSDT ALTUSDT BMTUSDT DGBUSDT WCTUSDT RESOLVUSDT"
for t in $TOKENS; do
  for h in 60 180; do
    for m in long short; do
      echo "=== training $t $m $h ==="
      /root/crypto-models/venv/bin/python /root/crypto-models/train/train.py "$t" --mode "$m" --horizon "$h"
    done
  done
done
