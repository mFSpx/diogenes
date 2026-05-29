#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lucidota_safe_ops_env.sh"
MODEL="${1:-$ROOT/03_VAULT/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf}"
HOST="${LUCIDOTA_LLAMA_HOST:-127.0.0.1}"
PORT="${LUCIDOTA_LLAMA_PORT:-8080}"
CTX="${LUCIDOTA_LLAMA_CTX:-2048}"
NGL="${LUCIDOTA_LLAMA_NGL:-99}"
PARALLEL="${LUCIDOTA_LLAMA_PARALLEL:-1}"
BATCH="${LUCIDOTA_LLAMA_BATCH:-256}"
UBATCH="${LUCIDOTA_LLAMA_UBATCH:-64}"
CACHE_RAM="${LUCIDOTA_LLAMA_CACHE_RAM:-0}"
SLOT_SAVE_PATH="${LUCIDOTA_LLAMA_SLOT_SAVE_PATH:-$ROOT/04_RUNTIME/inference_os/llama_slots}"
EXTRA_ARGS=()
if [[ "${LUCIDOTA_LLAMA_NO_WARMUP:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--no-warmup)
fi
mkdir -p "$SLOT_SAVE_PATH"
EXTRA_ARGS+=(--slot-save-path "$SLOT_SAVE_PATH")
if [[ -n "${LUCIDOTA_LLAMA_LORA_GGUF:-}" ]]; then
  IFS=',' read -r -a LORA_PATHS <<< "$LUCIDOTA_LLAMA_LORA_GGUF"
  for lora_path in "${LORA_PATHS[@]}"; do
    [[ -n "$lora_path" ]] && EXTRA_ARGS+=(--lora "$lora_path")
  done
  if [[ "${LUCIDOTA_LLAMA_LORA_INIT_WITHOUT_APPLY:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--lora-init-without-apply)
  fi
fi
if [[ -n "${LUCIDOTA_LLAMA_EXTRA_ARGS:-}" ]]; then
  # Critical assumption: operator-provided extra args are simple whitespace-delimited llama.cpp flags.
  # shellcheck disable=SC2206
  USER_EXTRA_ARGS=(${LUCIDOTA_LLAMA_EXTRA_ARGS})
  EXTRA_ARGS+=("${USER_EXTRA_ARGS[@]}")
fi
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
