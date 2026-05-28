#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lucidota_safe_ops_env.sh"
export CUDA_VISIBLE_DEVICES=""
MODEL="${1:-$ROOT/03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf}"
HOST="${LUCIDOTA_MAMBA_HOST:-127.0.0.1}"
PORT="${LUCIDOTA_MAMBA_PORT:-8081}"
CTX="${LUCIDOTA_MAMBA_CTX:-256}"
NGL="${LUCIDOTA_MAMBA_NGL:-0}"
PARALLEL="${LUCIDOTA_MAMBA_PARALLEL:-1}"
BATCH="${LUCIDOTA_MAMBA_BATCH:-32}"
UBATCH="${LUCIDOTA_MAMBA_UBATCH:-8}"
CACHE_RAM="${LUCIDOTA_MAMBA_CACHE_RAM:-0}"
SLOT_SAVE_PATH="${LUCIDOTA_MAMBA_SLOT_SAVE_PATH:-$ROOT/04_RUNTIME/inference_os/mamba_slots}"
EXTRA_ARGS=()
if [[ "${LUCIDOTA_MAMBA_NO_WARMUP:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--no-warmup)
fi
mkdir -p "$SLOT_SAVE_PATH"
EXTRA_ARGS+=(--slot-save-path "$SLOT_SAVE_PATH")
exec "$ROOT/01_REPOS/llama.cpp/build-cuda/bin/llama-server" \
  -m "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  -ngl "$NGL" \
  -c "$CTX" \
  --parallel "$PARALLEL" \
  --batch-size "$BATCH" \
  --ubatch-size "$UBATCH" \
  --cache-ram "$CACHE_RAM" \
  "${EXTRA_ARGS[@]}"
