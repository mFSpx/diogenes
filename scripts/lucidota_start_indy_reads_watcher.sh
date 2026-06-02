#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lucidota_safe_ops_env.sh"
LOG_DIR="$ROOT/04_RUNTIME"
PID_FILE="$LOG_DIR/indy_reads_watcher.pid"
LOG_FILE="$LOG_DIR/indy_reads_watcher.log"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"
mkdir -p "$LOG_DIR" "$ROOT/BOOKS/.indy_reads"

is_running_pid() {
  local pid="$1"
  kill -0 "$pid" >/dev/null 2>&1
}

: "${LUCIDOTA_INDY_READS_MAX_WORKERS:=1}"
: "${LUCIDOTA_INDY_READS_MAX_BATCH:=16}"
: "${LUCIDOTA_INDY_WATCH_NICE:=5}"

export LUCIDOTA_MAX_WORKERS="${LUCIDOTA_INDY_READS_MAX_WORKERS}"
export LUCIDOTA_MAX_BATCH="${LUCIDOTA_INDY_READS_MAX_BATCH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ -s "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE" || true)"
  if [[ -n "${pid:-}" ]] && is_running_pid "$pid"; then
    echo "INDY_READs watcher already online: $pid"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

ulimit -n 4096 2>/dev/null || true
ulimit -v 5242880 2>/dev/null || true

setsid env \
  PYTHONUNBUFFERED=1 \
  LUCIDOTA_MAX_WORKERS="$LUCIDOTA_MAX_WORKERS" \
  LUCIDOTA_MAX_BATCH="$LUCIDOTA_MAX_BATCH" \
  OMP_NUM_THREADS="$OMP_NUM_THREADS" \
  OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
  MKL_NUM_THREADS="$MKL_NUM_THREADS" \
  NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
  nice -n "$LUCIDOTA_INDY_WATCH_NICE" \
  "$PY" "$ROOT/scripts/lucidota_indy_reads_watcher.py" \
  --interval "${LUCIDOTA_INDY_WATCH_INTERVAL:-5}" \
  --append-lora-jsonl \
  >"$LOG_FILE" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
echo "INDY_READs watcher started: $(cat "$PID_FILE")"
