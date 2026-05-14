#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-$ROOT/03_VAULT/models/mamba-1.4b-hf-Q2_K.gguf}"
HOST="${LUCIDOTA_MAMBA_HOST:-127.0.0.1}"
PORT="${LUCIDOTA_MAMBA_PORT:-8081}"
CTX="${LUCIDOTA_MAMBA_CTX:-2048}"
NGL="${LUCIDOTA_MAMBA_NGL:-99}"
exec "$ROOT/01_REPOS/llama.cpp/build-cuda/bin/llama-server" \
  -m "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  -ngl "$NGL" \
  -c "$CTX"
