#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COUNT="${LUCIDOTA_NEEDLE_COUNT:-6}"
BASE_PORT="${LUCIDOTA_NEEDLE_BASE_PORT:-8090}"
PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/04_RUNTIME/needle_swarm"
mkdir -p "$LOG_DIR"
for i in $(seq 0 $((COUNT-1))); do
  port=$((BASE_PORT+i))
  if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "needle-$i already online on :$port"
    continue
  fi
  setsid env PYTHONPATH="$ROOT/01_REPOS/needle" "$PY" "$ROOT/scripts/lucidota_needle_worker.py" \
    --port "$port" \
    --instance "needle-$i" \
    >"$LOG_DIR/needle-$i.log" 2>&1 < /dev/null &
  echo $! > "$LOG_DIR/needle-$i.pid"
  echo "started needle-$i pid $(cat "$LOG_DIR/needle-$i.pid") port $port"
done
