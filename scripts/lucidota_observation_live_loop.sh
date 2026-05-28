#!/usr/bin/env bash
# Live updater for the canonical LUCIDOTA observation dashboard.
# Safe: one watchdog tick per interval under flock; does not touch ingestion workers.
set -euo pipefail
ROOT="/home/mfspx/LUCIDOTA"
PY="$ROOT/.venv/bin/python"
LOCK="$ROOT/04_RUNTIME/ingestion_watchdog.lock"
LOG="$ROOT/04_RUNTIME/ingestion_observation_live_loop.log"
INTERVAL="${LUCIDOTA_OBSERVATION_LIVE_INTERVAL:-30}"
CRITICAL_INTERVAL="${LUCIDOTA_OBSERVATION_CRITICAL_INTERVAL:-60}"
IO_CRIT="${LUCIDOTA_OBSERVATION_IO_CRIT:-35}"
cd "$ROOT"
mkdir -p "$ROOT/04_RUNTIME"
echo "[$(date -Iseconds)] observation live loop start interval=${INTERVAL}s" >> "$LOG"
while true; do
  start=$(date +%s)
  /usr/bin/flock -n "$LOCK" "$PY" "$ROOT/scripts/lucidota_ingest_watchdog.py" --self-disable >> "$LOG" 2>&1 || true
  now=$(date +%s)
  target_interval="$INTERVAL"
  io_full="$(awk '/^full / {for (i=1;i<=NF;i++) if ($i ~ /^avg10=/) {split($i,a,\"=\"); print a[2]}}' /proc/pressure/io 2>/dev/null || echo 0)"
  if awk "BEGIN {exit !($io_full >= $IO_CRIT)}"; then
    target_interval="$CRITICAL_INTERVAL"
  fi
  elapsed=$((now-start))
  sleep_for=$((target_interval-elapsed))
  if (( sleep_for < 5 )); then sleep_for=5; fi
  sleep "$sleep_for"
done
