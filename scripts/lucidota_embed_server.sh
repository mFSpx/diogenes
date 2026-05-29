#!/usr/bin/env bash
# Embedding server: BGE-M3 via llama-server on :8082
# Usage: bash scripts/lucidota_embed_server.sh [--port 8082]
set -euo pipefail
PORT="${1:-8082}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GGUF="${ROOT}/04_RUNTIME/models/bge-m3-q8_0.gguf"
LIB_PATH="${ROOT}/01_REPOS/llama.cpp/build-cuda/bin"
SERVER="${ROOT}/01_REPOS/llama.cpp/build-cuda/bin/llama-server"

if [[ ! -f "${GGUF}" ]]; then
  echo "ERROR: BGE-M3 GGUF not found at ${GGUF}" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${LIB_PATH}:${LD_LIBRARY_PATH:-}"
exec "${SERVER}" \
  --model "${GGUF}" \
  --port "${PORT}" \
  --host 127.0.0.1 \
  --embedding \
  --pooling mean \
  --ctx-size 8192 \
  --batch-size 512 \
  --threads 4 \
  --no-mmap \
  --log-disable
