#!/usr/bin/env bash
set -euo pipefail
ROOT="${LUCIDOTA_HOME:-/home/mfspx/LUCIDOTA}"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"
LOG_DIR="$ROOT/04_RUNTIME"
LOG_FILE="$LOG_DIR/korpus_embedding_worker.log"
PAUSE_FILE="$LOG_DIR/korpus_embedding_worker.pause"
INTERVAL="${KORPUS_EMBED_INTERVAL:-10}"
LIMIT="${KORPUS_EMBED_LIMIT:-200}"
mkdir -p "$LOG_DIR"
echo "[$(date -Iseconds)] KORPUS embedding worker awake interval=${INTERVAL}s limit=$LIMIT" | tee -a "$LOG_FILE"
while true; do
  if [[ -e "$PAUSE_FILE" ]]; then
    echo "[$(date -Iseconds)] KORPUS embedding worker paused: $(cat "$PAUSE_FILE" 2>/dev/null || true)" >> "$LOG_FILE"
    sleep 60
    continue
  fi
  "$PY" "$ROOT/scripts/korpus_krampii.py" --json embed-pending --limit "$LIMIT" >> "$LOG_FILE" 2>&1 || true
  sleep "$INTERVAL"
done
