#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lucidota_safe_ops_env.sh"
export CUDA_VISIBLE_DEVICES=""
HOST="${LUCIDOTA_BONSAI_HOST:-127.0.0.1}"
PORT="${LUCIDOTA_BONSAI_PORT:-8082}"
CTX="${LUCIDOTA_BONSAI_CTX:-1024}"
NGL="${LUCIDOTA_BONSAI_NGL:-0}"
LLAMA_ROOT="${LUCIDOTA_BONSAI_LLAMA_ROOT:-$ROOT/01_REPOS/prismml_llama.cpp}"
LLAMA_SERVER="${LUCIDOTA_BONSAI_LLAMA_SERVER:-$LLAMA_ROOT/build-cuda/bin/llama-server}"
MODEL="${LUCIDOTA_BONSAI_MODEL:-$ROOT/03_VAULT/models/prism-ml/Ternary-Bonsai-4B-gguf/Ternary-Bonsai-4B-Q2_0.gguf}"
SLOT_SAVE_PATH="${LUCIDOTA_BONSAI_SLOT_SAVE_PATH:-$ROOT/04_RUNTIME/inference_os/bonsai_slots}"
mkdir -p "$SLOT_SAVE_PATH"
if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "Missing PrismML Q2_0 llama-server: $LLAMA_SERVER" >&2
  echo "Build: scripts/build_bonsai_ternary_llama_cuda.sh" >&2
  exit 4
fi
export LD_LIBRARY_PATH="$LLAMA_ROOT/build-cuda/bin:$LLAMA_ROOT/build/bin:${LD_LIBRARY_PATH:-}"
exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  -ngl "$NGL" \
  -c "$CTX" \
  --parallel "${LUCIDOTA_BONSAI_PARALLEL:-1}" \
  --batch-size "${LUCIDOTA_BONSAI_BATCH:-128}" \
  --ubatch-size "${LUCIDOTA_BONSAI_UBATCH:-32}" \
  --cache-ram "${LUCIDOTA_BONSAI_CACHE_RAM:-0}" \
  --slot-save-path "$SLOT_SAVE_PATH" \
  --no-warmup
