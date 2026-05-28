#!/usr/bin/env bash
set -euo pipefail
ROOT="${LUCIDOTA_HOME:-/home/mfspx/LUCIDOTA}"
source "$ROOT/scripts/lucidota_safe_ops_env.sh"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"
LOG="$ROOT/04_RUNTIME/indy_polycareer_watch.log"
PIDFILE="$ROOT/04_RUNTIME/indy_polycareer_watch.pid"
INTERVAL="${LUCIDOTA_INDY_POLYCAREER_INTERVAL:-120}"
SINCE_HOURS="${LUCIDOTA_INDY_POLYCAREER_SINCE_HOURS:-2}"
THRESHOLD="${LUCIDOTA_INDY_POLYCAREER_GLOW_THRESHOLD:-35}"
mkdir -p "$ROOT/04_RUNTIME" "$ROOT/05_OUTPUTS/indy_polycareer"
if [[ -f "$PIDFILE" ]]; then
  old="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "$old" && -d "/proc/$old" ]]; then
    exit 0
  fi
fi
nohup bash -c '
  set -u
  ROOT="$1"; PY="$2"; INTERVAL="$3"; SINCE_HOURS="$4"; THRESHOLD="$5"
  echo "[$(date -Iseconds)] INDY polycareer Glow Watch online interval=$INTERVAL since_hours=$SINCE_HOURS threshold=$THRESHOLD"
  while true; do
    "$PY" "$ROOT/scripts/lucidota_indy_polycareer.py" --json watch-once --since-hours "$SINCE_HOURS" --threshold "$THRESHOLD" --limit 25 || true
    sleep "$INTERVAL"
  done
' _ "$ROOT" "$PY" "$INTERVAL" "$SINCE_HOURS" "$THRESHOLD" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
