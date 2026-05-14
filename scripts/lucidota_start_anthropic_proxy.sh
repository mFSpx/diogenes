#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${LUCIDOTA_ANTHROPIC_PROXY_PORT:-8088}"
UPSTREAM="${LUCIDOTA_DEEPSEEK_API_BASE:-http://127.0.0.1:8080/v1}"
MODEL="${LUCIDOTA_DEEPSEEK_MODEL:-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf}"
LOG_DIR="$ROOT/04_RUNTIME"
mkdir -p "$LOG_DIR"
if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "anthropic proxy already online on :$PORT"
  exit 0
fi
setsid "$ROOT/.venv/bin/python" "$ROOT/scripts/lucidota_anthropic_llama_proxy.py" \
  --port "$PORT" --upstream "$UPSTREAM" --model "$MODEL" \
  >"$LOG_DIR/anthropic_proxy.log" 2>&1 < /dev/null &
echo $! > "$LOG_DIR/anthropic_proxy.pid"
for _ in $(seq 1 30); do
  curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && exit 0
  sleep 0.2
done
echo "anthropic proxy failed to start" >&2
exit 1
