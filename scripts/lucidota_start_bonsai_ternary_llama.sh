#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lucidota_safe_ops_env.sh"
# Bonsai 8B lane. Default is the explicit Q1_0 / 1-bit, two-slot,
# unified-KV VRAM topology used by IronClaw; Q2_0 remains available by
# LUCIDOTA_BONSAI_VARIANT=q2_0 for local experiments.
export CUDA_VISIBLE_DEVICES="${LUCIDOTA_BONSAI_CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
HOST="${LUCIDOTA_BONSAI_HOST:-127.0.0.1}"
PORT="${LUCIDOTA_BONSAI_PORT:-8082}"
CTX="${LUCIDOTA_BONSAI_CTX:-2048}"
NGL="${LUCIDOTA_BONSAI_NGL:-999}"
LLAMA_ROOT="${LUCIDOTA_BONSAI_LLAMA_ROOT:-$ROOT/01_REPOS/prismml_llama.cpp}"
LLAMA_SERVER="${LUCIDOTA_BONSAI_LLAMA_SERVER:-$LLAMA_ROOT/build-cuda/bin/llama-server}"
MODEL_Q2="${ROOT}/03_VAULT/models/prism-ml/Ternary-Bonsai-8B-gguf/Ternary-Bonsai-8B-Q2_0.gguf"
MODEL_Q1="${ROOT}/03_VAULT/models/prism-ml/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf"
MODEL_OVERRIDE="${LUCIDOTA_BONSAI_MODEL:-}"
MODEL_VARIANT="${LUCIDOTA_BONSAI_VARIANT:-q1_0}"
MODEL_HF_Q1="${LUCIDOTA_BONSAI_HF_MODEL:-prism-ml/Bonsai-8B-gguf:Q1_0}"
SLOT_SAVE_PATH="${LUCIDOTA_BONSAI_SLOT_SAVE_PATH:-$ROOT/04_RUNTIME/inference_os/bonsai_q1_shared_slots}"
mkdir -p "$SLOT_SAVE_PATH"
if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "Missing PrismML Q2_0 llama-server: $LLAMA_SERVER" >&2
  echo "Build: scripts/build_bonsai_ternary_llama_cuda.sh" >&2
  exit 4
fi
export LD_LIBRARY_PATH="$LLAMA_ROOT/build-cuda/bin:$LLAMA_ROOT/build/bin:${LD_LIBRARY_PATH:-}"
COMMON_ARGS=(
  --host "$HOST"
  --port "$PORT"
  -ngl "$NGL"
  -c "$CTX"
  --parallel "${LUCIDOTA_BONSAI_PARALLEL:-2}"
  --batch-size "${LUCIDOTA_BONSAI_BATCH:-128}"
  --ubatch-size "${LUCIDOTA_BONSAI_UBATCH:-32}"
  --cache-ram "${LUCIDOTA_BONSAI_CACHE_RAM:-0}"
  --slot-save-path "$SLOT_SAVE_PATH"
  --no-warmup
)
if [[ "${LUCIDOTA_BONSAI_KV_UNIFIED:-1}" == "1" ]]; then
  COMMON_ARGS+=(--kv-unified)
fi
if [[ "${LUCIDOTA_BONSAI_KV_OFFLOAD:-1}" == "1" ]]; then
  COMMON_ARGS+=(--kv-offload)
fi
if [[ "${LUCIDOTA_BONSAI_CACHE_PROMPT:-1}" == "1" ]]; then
  COMMON_ARGS+=(--cache-prompt)
fi
COMMON_ARGS+=(--cache-type-k "${LUCIDOTA_BONSAI_CACHE_TYPE_K:-q8_0}")
COMMON_ARGS+=(--cache-type-v "${LUCIDOTA_BONSAI_CACHE_TYPE_V:-q8_0}")
COMMON_ARGS+=(--alias "${LUCIDOTA_BONSAI_ALIAS:-bonsai8b-q1-shared2}")
if [[ -n "$MODEL_OVERRIDE" && -f "$MODEL_OVERRIDE" ]]; then
  exec "$LLAMA_SERVER" \
    -m "$MODEL_OVERRIDE" \
    "${COMMON_ARGS[@]}"
else
  case "$MODEL_VARIANT" in
    q1_0|q1|1bit)
      if [[ -f "$MODEL_Q1" ]]; then
        exec "$LLAMA_SERVER" \
          -m "$MODEL_Q1" \
          "${COMMON_ARGS[@]}"
      fi
      exec "$LLAMA_SERVER" \
        -hf "$MODEL_HF_Q1" \
        "${COMMON_ARGS[@]}"
      ;;
    q2_0|q2|ternary|*)
      if [[ ! -f "$MODEL_Q2" ]]; then
        echo "Missing local Bonsai Q2_0 model: $MODEL_Q2" >&2
        echo "Set LUCIDOTA_BONSAI_VARIANT=q1_0 to use the switchable 1-bit lane." >&2
        exit 5
      fi
      exec "$LLAMA_SERVER" \
        -m "$MODEL_Q2" \
        "${COMMON_ARGS[@]}"
      ;;
  esac
fi
