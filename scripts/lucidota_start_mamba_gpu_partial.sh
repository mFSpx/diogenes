#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lucidota_safe_ops_env.sh"
exec "$ROOT/01_REPOS/llama.cpp/build-cuda/bin/llama-server" \
  -m "${LUCIDOTA_MAMBA_GPU_MODEL:-$ROOT/03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf}" \
  --host "${LUCIDOTA_MAMBA_GPU_HOST:-127.0.0.1}" \
  --port "${LUCIDOTA_MAMBA_GPU_PORT:-8083}" \
  -ngl "${LUCIDOTA_MAMBA_GPU_NGL:-24}" \
  -c "${LUCIDOTA_MAMBA_GPU_CTX:-128}" \
  --parallel 1 --batch-size 16 --ubatch-size 4 --cache-ram 0 --no-warmup
