#!/usr/bin/env bash
# Usage: watchdog_wrapper.sh <command...>
# 1. Starts the target command in background.
# 2. Waits up to 10s and hits http://127.0.0.1:8001/healthz (override via HEALTH_URL env).
# 3. If health-check fails, attempts rollback via `systemctl revert` on caller unit (if possible).
set -euo pipefail
HEALTH_URL=${HEALTH_URL:-http://127.0.0.1:8001/healthz}
CMD="$*"
log() { echo "[watchdog] $*"; }
log "Starting: $CMD"; eval "$CMD &"; pid=$!
for i in {1..10}; do
  sleep 1
  if curl -fs "$HEALTH_URL" >/dev/null 2>&1; then
    log "Health OK"; wait $pid; exit 0
  fi
done
log "Health check failed – attempting rollback";
unit=$(systemctl --no-legend --property=Names show $$ | cut -d= -f2 | cut -d';' -f1 || true)
if [[ -n "$unit" ]]; then
  systemctl revert "$unit" || true
  systemctl restart "$unit" || true
fi
kill $pid || true
exit 1
