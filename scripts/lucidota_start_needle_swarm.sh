#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lucidota_safe_ops_env.sh"
COUNT="${LUCIDOTA_NEEDLE_COUNT:-6}"
BASE_PORT="${LUCIDOTA_NEEDLE_BASE_PORT:-8090}"
SHARED_SERVER="${LUCIDOTA_NEEDLE_SHARED_SERVER:-1}"
SLOTS="${LUCIDOTA_NEEDLE_SLOTS:-6}"
PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/04_RUNTIME/needle_swarm"
mkdir -p "$LOG_DIR"
if [[ "$SHARED_SERVER" == "1" ]]; then
  port="$BASE_PORT"
  if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "needle-shared-${SLOTS} already online on :$port"
    exit 0
  fi
  setsid env \
    PYTHONPATH="$ROOT/01_REPOS/needle" \
    CUDA_VISIBLE_DEVICES="${LUCIDOTA_NEEDLE_CUDA_VISIBLE_DEVICES:-}" \
    JAX_PLATFORMS="${LUCIDOTA_NEEDLE_JAX_PLATFORMS:-cpu}" \
    XLA_PYTHON_CLIENT_PREALLOCATE="false" \
    "$PY" "$ROOT/scripts/lucidota_needle_worker.py" \
    --port "$port" \
    --slots "$SLOTS" \
    --instance "needle-shared-${SLOTS}" \
    >"$LOG_DIR/needle-shared-${SLOTS}.log" 2>&1 < /dev/null &
  echo $! > "$LOG_DIR/needle-shared-${SLOTS}.pid"
  echo "started needle-shared-${SLOTS} pid $(cat "$LOG_DIR/needle-shared-${SLOTS}.pid") port $port"
  exit 0
fi
for i in $(seq 0 $((COUNT-1))); do
  port=$((BASE_PORT+i))
  if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "needle-$i already online on :$port"
    continue
  fi
  setsid env \
    PYTHONPATH="$ROOT/01_REPOS/needle" \
    CUDA_VISIBLE_DEVICES="" \
    JAX_PLATFORMS="cpu" \
    XLA_PYTHON_CLIENT_PREALLOCATE="false" \
    "$PY" "$ROOT/scripts/lucidota_needle_worker.py" \
    --port "$port" \
    --instance "needle-$i" \
    >"$LOG_DIR/needle-$i.log" 2>&1 < /dev/null &
  echo $! > "$LOG_DIR/needle-$i.pid"
  echo "started needle-$i pid $(cat "$LOG_DIR/needle-$i.pid") port $port"
done
