#!/usr/bin/env bash
# Nightly backup script for crypto-models – creates compressed archive under /var/backups
set -euo pipefail

DEST="/var/backups"
TS="$(date +%F_%H-%M)"
ARCHIVE="$DEST/crypto-models-$TS.tar.zst"

mkdir -p "$DEST"

tar --exclude='venv' -I 'zstd -19' -cf "$ARCHIVE" -C /home/cm crypto-models

echo "[+] backup written → $ARCHIVE"
