#!/usr/bin/env bash
# BGE-M3 embedding FLEET — GPU mode: 1 server, --parallel COUNT slots (one copy of weights in VRAM).
#                           CPU mode: N servers with mmap'd shared weights in RAM.
#
# GPU mode (NGL > 0, default):
#   One llama-server process owns the model weights once in VRAM (~620 MiB), serves
#   COUNT parallel embedding slots via continuous batching. --cache-ram 0 kills the
#   8192 MiB/instance default prompt-cache that previously OOM'd the box.
#
# CPU mode (NGL = 0, set LUCIDOTA_BGE_NGL=0):
#   N separate servers, weights mmap'd → all N share ONE physical RAM copy (~606 MiB).
#   Per-server compute buffer is private. --no-mmap breaks sharing; never pass it.
#
# Usage:  scripts/lucidota_bge_fleet.sh [COUNT]
#         scripts/lucidota_bge_fleet.sh stop
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

COUNT="${1:-${LUCIDOTA_BGE_FLEET_COUNT:-16}}"
GGUF="${ROOT}/04_RUNTIME/models/bge-m3-q8_0.gguf"
SERVER="${ROOT}/01_REPOS/llama.cpp/build-cuda/bin/llama-server"
LIB="${ROOT}/01_REPOS/llama.cpp/build-cuda/bin"
CTX="${LUCIDOTA_BGE_FLEET_CTX:-2048}"
BATCH="${LUCIDOTA_BGE_FLEET_BATCH:-4096}"
UBATCH="${LUCIDOTA_BGE_FLEET_UBATCH:-2048}"
THREADS="${LUCIDOTA_BGE_FLEET_THREADS:-2}"
NGL="${LUCIDOTA_BGE_NGL:-0}"

[[ -f "$GGUF" ]] || { echo "ERROR: missing $GGUF" >&2; exit 1; }
[[ -x "$SERVER" ]] || { echo "ERROR: missing llama-server $SERVER" >&2; exit 1; }
export LD_LIBRARY_PATH="$LIB:/usr/local/lib/ollama/cuda_v12:${LD_LIBRARY_PATH:-}"

if (( NGL > 0 )); then
  # ── GPU MODE ─────────────────────────────────────────────────────────────
  # LUCIDOTA_BGE_GPU_INSTANCES separate servers, each with SLOTS_PER parallel slots.
  # Each process loads its own model copy in VRAM (~456 MiB each).
  # 4 instances × 8 slots = 32 total concurrent slots, ~1840 MiB VRAM.
  GPU_INSTANCES="${LUCIDOTA_BGE_GPU_INSTANCES:-1}"
  SLOTS_PER=$(( COUNT / GPU_INSTANCES ))
  (( SLOTS_PER < 1 )) && SLOTS_PER=1
  CTX_PER=$(( SLOTS_PER * 2048 ))   # each slot gets full 2048 ctx

  urls=()
  for ((inst=0; inst<GPU_INSTANCES; inst++)); do
    port=$(( BASE_PORT + inst ))
    urls+=("http://127.0.0.1:${port}")
    pidf="$RUN/bge_fleet_${port}.pid"
    log="$RUN/bge_fleet_${port}.log"

    if curl -fsS --max-time 1 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "bge:${port} (gpu inst=$inst slots=${SLOTS_PER}) already online"; continue
    fi

    setsid env OMP_NUM_THREADS="$THREADS" "$SERVER" \
      --model "$GGUF" --host 127.0.0.1 --port "$port" \
      --embedding --pooling mean --ctx-size "$CTX_PER" --batch-size "$BATCH" --ubatch-size "$UBATCH" \
      --threads "$THREADS" --threads-http 20 -ngl "$NGL" --parallel "$SLOTS_PER" \
      --cache-ram 0 --no-cache-prompt \
      >"$log" 2>&1 < /dev/null & echo $! > "$pidf"
    echo "bge:${port} gpu started pid $(cat "$pidf") inst=$inst slots=${SLOTS_PER} ctx=${CTX_PER}"
    for ((wait=0; wait<45; wait++)); do
      sleep 1
      if curl -fsS --max-time 1 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        echo "bge:${port} healthy (${wait}s)"
        if grep -q "ggml_cuda_init: failed\|CUDA error\|failed to allocate CUDA\|load_backend: loaded CPU" "$log" 2>/dev/null; then
          echo "bge:${port} WARNING: CUDA load failed — check $log" >&2
        else
          vram_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo '?')"
          echo "bge:${port} VRAM confirmed (${vram_used} MiB total used)"
        fi
        break
      fi
    done
  done

  # Shrink: kill stale instances above GPU_INSTANCES
  for pidf2 in "$RUN"/bge_fleet_*.pid; do
    [[ -s "$pidf2" ]] || continue
    fname="${pidf2##*/}"; p="${fname#bge_fleet_}"; p="${p%.pid}"
    if (( p >= BASE_PORT + GPU_INSTANCES )); then
      pid2="$(cat "$pidf2")"
      kill "$pid2" 2>/dev/null && echo "shrink: stopped stale bge port=$p pid=$pid2" || true
      rm -f "$pidf2"
    fi
  done

else
  # ── CPU MODE ─────────────────────────────────────────────────────────────
  # N separate servers, weights mmap-shared in RAM.
  urls=()
  for ((i=0; i<COUNT; i++)); do
    port=$((BASE_PORT + i))
    urls+=("http://127.0.0.1:${port}")
    pidf="$RUN/bge_fleet_${port}.pid"
    log="$RUN/bge_fleet_${port}.log"
    if curl -fsS --max-time 1 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "bge:${port} already online"; continue
    fi
    setsid env OMP_NUM_THREADS="$THREADS" "$SERVER" \
      --model "$GGUF" --host 127.0.0.1 --port "$port" \
      --embedding --pooling mean --ctx-size "$CTX" --batch-size "$BATCH" --ubatch-size "$UBATCH" \
      --threads "$THREADS" -ngl 0 --cache-ram 0 \
      >"$log" 2>&1 < /dev/null & echo $! > "$pidf"
    echo "bge:${port} started pid $(cat "$pidf")"
    for ((wait=0; wait<30; wait++)); do
      sleep 1
      if curl -fsS --max-time 1 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        echo "bge:${port} healthy (${wait}s)"; break
      fi
    done
  done

  # Shrink: kill instances above COUNT
  for pidf in "$RUN"/bge_fleet_*.pid; do
    [[ -s "$pidf" ]] || continue
    fname="${pidf##*/}"; port="${fname#bge_fleet_}"; port="${port%.pid}"
    if (( port >= BASE_PORT + COUNT )); then
      pid="$(cat "$pidf")"
      kill "$pid" 2>/dev/null && echo "shrink: stopped bge port=$port pid=$pid" || true
      rm -f "$pidf"
    fi
  done
fi

fleet="$(IFS=,; echo "${urls[*]}")"
echo "$fleet" > "$RUN/bge_fleet.endpoints"
echo "LUCIDOTA_BGE_FLEET=$fleet"
