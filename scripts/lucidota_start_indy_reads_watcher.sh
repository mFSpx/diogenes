#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lucidota_safe_ops_env.sh"
LOG_DIR="$ROOT/04_RUNTIME"
PID_FILE="$LOG_DIR/indy_daemon.pid"
LOG_FILE="$LOG_DIR/indy_daemon.log"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"
mkdir -p "$LOG_DIR"
FOREGROUND=0
if [[ "${1:-}" == "--foreground" ]]; then
  FOREGROUND=1
  shift
fi

is_running_pid() {
  local pid="$1"
  kill -0 "$pid" >/dev/null 2>&1
}

is_daemon_pid() {
  local pid="$1"
  [[ -n "$pid" ]] || return 1
  ps -p "$pid" -o args= 2>/dev/null | grep -F "$ROOT/scripts/indy_daemon.py" >/dev/null
}

find_existing_daemon_pids() {
  pgrep -f "$ROOT/scripts/indy_daemon.py" 2>/dev/null | while read -r found_pid; do
    [[ -n "$found_pid" ]] || continue
    [[ "$found_pid" == "$$" ]] && continue
    if ps -p "$found_pid" -o args= 2>/dev/null | grep -F "$ROOT/scripts/indy_daemon.py" >/dev/null; then
      printf '%s\n' "$found_pid"
    fi
  done
}

: "${LUCIDOTA_INDY_READS_MAX_BATCH:=16}"
: "${LUCIDOTA_INDY_DAEMON_NICE:=5}"

export LUCIDOTA_MAX_BATCH="${LUCIDOTA_INDY_READS_MAX_BATCH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ -s "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE" || true)"
  if [[ -n "${pid:-}" ]] && is_running_pid "$pid" && is_daemon_pid "$pid"; then
    echo "INDY_READs daemon already online: $pid"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

existing_pid="$(find_existing_daemon_pids | head -n 1 || true)"
if [[ -n "$existing_pid" ]] && is_running_pid "$existing_pid"; then
  echo "$existing_pid" > "$PID_FILE"
  echo "INDY_READs daemon already online: $existing_pid"
  exit 0
fi

ulimit -n 4096 2>/dev/null || true
if [[ "${LUCIDOTA_INDY_USE_ULIMIT_V:-0}" == "1" ]]; then
  ulimit -v "${LUCIDOTA_INDY_ULIMIT_V_KB:-5242880}" 2>/dev/null || true
fi

DAEMON_CMD=(
  "$PY" "$ROOT/scripts/indy_daemon.py"
  --loop
  --json
  --limit "${LUCIDOTA_INDY_DAEMON_LIMIT:-25}"
  --max-items "${LUCIDOTA_INDY_DAEMON_MAX_ITEMS:-12}"
  --interval "${LUCIDOTA_INDY_WATCH_INTERVAL:-5}"
)

if [[ "$FOREGROUND" == "1" ]]; then
  echo "$$" > "$PID_FILE"
  exec env \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
    MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
    nice -n "$LUCIDOTA_INDY_DAEMON_NICE" \
    "${DAEMON_CMD[@]}"
fi

setsid env \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS="$OMP_NUM_THREADS" \
  OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
  MKL_NUM_THREADS="$MKL_NUM_THREADS" \
  NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
  nice -n "$LUCIDOTA_INDY_DAEMON_NICE" \
  "${DAEMON_CMD[@]}" \
  >"$LOG_FILE" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
echo "INDY_READs daemon started: $(cat "$PID_FILE")"
