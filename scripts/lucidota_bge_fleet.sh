#!/usr/bin/env bash
# BGE-M3 embedding FLEET — N llama-server embedders that SHARE mmap'd weights.
#
# Per-MISSION horizontal scale knob, NOT a system default: pass COUNT explicitly.
# The system default for ordinary work stays modest; a big crush mission asks for
# a wider fleet on purpose. Requests are round-robined by the client across all
# live members (LUCIDOTA_BGE_FLEET), so the fleet can scale up/down freely.
#
# Why this fits a 7.6GB box: the 634MB gguf is mmap'd (NOT --no-mmap), so all N
# servers share ONE physical copy of the weights; only the per-server compute
# buffer is private. CPU-only (CUDA_VISIBLE_DEVICES=) keeps the 4GB VRAM free for
# the GPU chat model. ctx/batch tuned for ~500-token chunks.
#
# Usage:  scripts/lucidota_bge_fleet.sh [COUNT]
#         scripts/lucidota_bge_fleet.sh stop      # tear the fleet down
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lucidota_safe_ops_env.sh"

RUN="$ROOT/04_RUNTIME/inference_os"; mkdir -p "$RUN"
BASE_PORT="${LUCIDOTA_BGE_FLEET_BASE_PORT:-8101}"

if [[ "${1:-}" == "stop" ]]; then
  for pidf in "$RUN"/bge_fleet_*.pid; do
    [[ -s "$pidf" ]] || continue
    pid="$(cat "$pidf")"; kill "$pid" 2>/dev/null && echo "stopped bge pid $pid" || true
    rm -f "$pidf"
  done
  rm -f "$RUN/bge_fleet.endpoints"
  exit 0
fi

COUNT="${1:-${LUCIDOTA_BGE_FLEET_COUNT:-4}}"
GGUF="${ROOT}/04_RUNTIME/models/bge-m3-q8_0.gguf"
SERVER="${ROOT}/01_REPOS/llama.cpp/build-cuda/bin/llama-server"
LIB="${ROOT}/01_REPOS/llama.cpp/build-cuda/bin"
CTX="${LUCIDOTA_BGE_FLEET_CTX:-2048}"
# physical batch (ubatch) MUST cover a whole chunk's tokens or llama.cpp 500s with
# "input too large to process. increase the physical batch size". A 1800-char chunk
# is up to ~1800 tokens, so batch+ubatch=2048 covers it. (The old default of 64 was the bug.)
BATCH="${LUCIDOTA_BGE_FLEET_BATCH:-2048}"
UBATCH="${LUCIDOTA_BGE_FLEET_UBATCH:-2048}"
THREADS="${LUCIDOTA_BGE_FLEET_THREADS:-2}"

[[ -f "$GGUF" ]] || { echo "ERROR: missing $GGUF" >&2; exit 1; }
[[ -x "$SERVER" ]] || { echo "ERROR: missing llama-server $SERVER" >&2; exit 1; }
export LD_LIBRARY_PATH="$LIB:/usr/local/lib/ollama/cuda_v12:${LD_LIBRARY_PATH:-}"

urls=()
for ((i=0; i<COUNT; i++)); do
  port=$((BASE_PORT + i))
  urls+=("http://127.0.0.1:${port}")
  pidf="$RUN/bge_fleet_${port}.pid"
  log="$RUN/bge_fleet_${port}.log"
  if curl -fsS --max-time 1 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "bge:${port} already online"; continue
  fi
  setsid env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS="$THREADS" "$SERVER" \
    --model "$GGUF" --host 127.0.0.1 --port "$port" \
    --embedding --pooling mean --ctx-size "$CTX" --batch-size "$BATCH" --ubatch-size "$UBATCH" \
    --threads "$THREADS" \
    >"$log" 2>&1 < /dev/null & echo $! > "$pidf"
  echo "bge:${port} started pid $(cat "$pidf")"
done

fleet="$(IFS=,; echo "${urls[*]}")"
echo "$fleet" > "$RUN/bge_fleet.endpoints"
echo "LUCIDOTA_BGE_FLEET=$fleet"
