#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-$ROOT/03_VAULT/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf}"
HOST="${LUCIDOTA_LLAMA_HOST:-127.0.0.1}"
PORT="${LUCIDOTA_LLAMA_PORT:-8080}"
CTX="${LUCIDOTA_LLAMA_CTX:-2048}"
NGL="${LUCIDOTA_LLAMA_NGL:-99}"
exec "$ROOT/01_REPOS/llama.cpp/build-cuda/bin/llama-server" \
  -m "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  -ngl "$NGL" \
  -c "$CTX"
