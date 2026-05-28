#!/usr/bin/env bash
set -u
ROOT="/home/mfspx/LUCIDOTA"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/04_RUNTIME/lucidota_stream_river_worker.log"
BYTEWAX_LIMIT="${BYTEWAX_LIMIT:-250}"
RIVER_REFLEX_LIMIT="${RIVER_REFLEX_LIMIT:-5000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-10}"
export LUCIDOTA_HOME="$ROOT"
export DBOS_SYSTEM_DATABASE_URL="${DBOS_SYSTEM_DATABASE_URL:-postgresql:///lucidota_state}"
mkdir -p "$ROOT/04_RUNTIME"
cd "$ROOT"
echo "[$(date -Iseconds)] LUCIDOTA stream river worker online: Bytewax dataflow + River reflex" >> "$LOG"
while true; do
  echo "[$(date -Iseconds)] bytewax live cursor tick" >> "$LOG"
  "$PY" "$ROOT/scripts/lucidota_bytewax_mini.py" --json --live-cursor --limit "$BYTEWAX_LIMIT" >> "$LOG" 2>&1 || true
  echo "[$(date -Iseconds)] river reflex tick" >> "$LOG"
  "$PY" "$ROOT/scripts/lucidota_river_reflex.py" --json --limit "$RIVER_REFLEX_LIMIT" >> "$LOG" 2>&1 || true
  sleep "$SLEEP_SECONDS"
done
