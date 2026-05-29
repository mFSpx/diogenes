#!/usr/bin/env bash
# Ensure the canonical observation dashboard live loop stays up.
set -euo pipefail
ROOT="/home/mfspx/LUCIDOTA"
PIDFILE="$ROOT/04_RUNTIME/ingestion_observation_live_loop.pid"
SCRIPT="$ROOT/scripts/lucidota_observation_live_loop.sh"
LOG="$ROOT/04_RUNTIME/ingestion_observation_live_loop.log"
mkdir -p "$ROOT/04_RUNTIME"
if [[ -s "$PIDFILE" ]]; then
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" == *"$SCRIPT"* ]]; then
      exit 0
    fi
  fi
fi
setsid -f /bin/bash "$SCRIPT" </dev/null >/dev/null 2>&1
sleep 1
pid="$(pgrep -f "^/bin/bash $SCRIPT$" | head -1 || true)"
if [[ -n "$pid" ]]; then
  echo "$pid" > "$PIDFILE"
  echo "[$(date -Iseconds)] ensure started observation live loop pid=$pid" >> "$LOG"
else
  echo "[$(date -Iseconds)] ensure failed to find observation live loop after start" >> "$LOG"
fi
