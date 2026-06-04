#!/bin/bash
# Start Bonsai 2-head full-GPU profile
# This script starts the llama-server with the required GPU configuration

set -euo pipefail

# Configuration
MODEL_PATH="/home/mfspx/LUCIDOTA/03_VAULT/models/prism-ml/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf"
LLAMA_SERVER_BIN="/home/mfspx/LUCIDOTA/01_REPOS/prismml_llama.cpp/build-cuda/bin/llama-server"
PORT=8082
HOST="127.0.0.1"

# GPU Configuration (non-negotiable)
NGL=999           # Use all GPU layers
CONTEXT=8192      # Context size
PARALLEL=2        # Parallel count
KV_UNIFIED=true   # Unified KV cache
KV_OFFLOAD=true    # KV offload
KV_CACHE_TYPE_K="q8_0"
KV_CACHE_TYPE_V="q8_0"
BATCH_SIZE=64
UBATCH_SIZE=16
CACHE_RAM=0
SLOT_SAVE_PATH="/home/mfspx/LUCIDOTA/04_RUNTIME/inference_os/bonsai_q1_shared_slots"

# CUDA Library Path
export LD_LIBRARY_PATH="/usr/local/lib/ollama/cuda_v12:${LD_LIBRARY_PATH:-}"

echo "Starting Bonsai 2-head full-GPU profile..."
echo "Model: Bonsai-8B-Q1_0"
echo "Port: $PORT"
echo "Host: $HOST"
echo "GPU Layers: $NGL"
echo "Context: $CONTEXT"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Check if llama-server is already running
if pgrep -f "llama-server.*port $PORT" >/dev/null 2>&1; then
    echo "ERROR: llama-server already running on port $PORT"
    exit 1
fi

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found at $MODEL_PATH"
    exit 1
fi

# Check if llama-server binary exists
if [ ! -x "$LLAMA_SERVER_BIN" ]; then
    echo "ERROR: llama-server binary not found at $LLAMA_SERVER_BIN"
    exit 1
fi

# Start llama-server
"$LLAMA_SERVER_BIN" \
    -m "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    -ngl "$NGL" \
    -c "$CONTEXT" \
    --parallel "$PARALLEL" \
    --kv-unified \
    --kv-offload \
    --cache-prompt \
    --cache-type-k "$KV_CACHE_TYPE_K" \
    --cache-type-v "$KV_CACHE_TYPE_V" \
    --batch-size "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --cache-ram "$CACHE_RAM" \
    --slot-save-path "$SLOT_SAVE_PATH" \
    --no-warmup \
    --alias bonsai8b-q1-shared2 &

echo "Bonsai GPU server starting with PID $!"

# Wait for server to be ready
MAX_RETRIES=30
RETRY_DELAY=2

for i in $(seq 1 $MAX_RETRIES); do
    if curl -s http://127.0.0.1:8082/health >/dev/null 2>&1; then
        echo "Bonsai GPU server is ready!"
        echo "BONSAI_2HEAD_FULLGPU_OK"
        exit 0
    fi
    if [ $i -lt $MAX_RETRIES ]; then
        echo "Waiting for server to start... ($i/$MAX_RETRIES)"
        sleep $RETRY_DELAY
    fi
done

echo "ERROR: Bonsai GPU server failed to start within $MAX_RETRIES seconds"
exit 1
