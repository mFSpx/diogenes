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

is_running_pid() {
  local pid="$1"
  kill -0 "$pid" >/dev/null 2>&1
}

: "${LUCIDOTA_INDY_POLYCAREER_MAX_WORKERS:=1}"
: "${LUCIDOTA_INDY_POLYCAREER_MAX_BATCH:=16}"
: "${LUCIDOTA_INDY_POLYCAREER_NICE:=5}"

export LUCIDOTA_MAX_WORKERS="${LUCIDOTA_INDY_POLYCAREER_MAX_WORKERS}"
export LUCIDOTA_MAX_BATCH="${LUCIDOTA_INDY_POLYCAREER_MAX_BATCH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ -f "$PIDFILE" ]]; then
  old="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "$old" ]] && is_running_pid "$old"; then
    exit 0
  fi
  rm -f "$PIDFILE"
fi

ulimit -n 4096 2>/dev/null || true
ulimit -v 5242880 2>/dev/null || true

setsid bash -c '
  set -u
  ROOT="$1"; PY="$2"; INTERVAL="$3"; SINCE_HOURS="$4"; THRESHOLD="$5"; WATCH_NICE="${LUCIDOTA_INDY_POLYCAREER_NICE:-5}"
  export PYTHONUNBUFFERED=1
  export LUCIDOTA_MAX_WORKERS="${LUCIDOTA_MAX_WORKERS}"
  export LUCIDOTA_MAX_BATCH="${LUCIDOTA_MAX_BATCH}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}"
  echo "[$(date -Iseconds)] INDY polycareer Glow Watch online interval=$INTERVAL since_hours=$SINCE_HOURS threshold=$THRESHOLD"
  while true; do
    nice -n "${WATCH_NICE}" "$PY" "$ROOT/scripts/lucidota_indy_polycareer.py" --json watch-once --since-hours "$SINCE_HOURS" --threshold "$THRESHOLD" --limit 25 || true
    sleep "$INTERVAL"
  done
' _ "$ROOT" "$PY" "$INTERVAL" "$SINCE_HOURS" "$THRESHOLD" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
