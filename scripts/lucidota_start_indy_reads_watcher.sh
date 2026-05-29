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

if [[ -s "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE" || true)"
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    echo "INDY_READs watcher already online: $pid"
    exit 0
  fi
fi

setsid "$PY" "$ROOT/scripts/lucidota_indy_reads_watcher.py" \
  --interval "${LUCIDOTA_INDY_WATCH_INTERVAL:-5}" \
  --append-lora-jsonl \
  >"$LOG_FILE" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
echo "INDY_READs watcher started: $(cat "$PID_FILE")"
