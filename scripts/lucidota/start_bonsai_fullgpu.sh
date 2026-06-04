#!/bin/bash
# LUCIDOTA Bonsai Full-GPU Profile Starter
# Starts Bonsai 2-head full-GPU profile:
# one llama-server, one weight load, port 8082, -ngl 999, -c 8192, --parallel 2, --kv-unified, --kv-offload, q8 KV

set -euo pipefail

echo "=== LUCIDOTA BONSAI FULL-GPU PROFILE STARTER ==="
echo "Timestamp: $(date -Iseconds)"
echo ""

# Configuration
WEIGHTS_DIR="/home/mfspx/LUCIDOTA/01_REPOS/ruflo/models"
MODEL_NAME="bonsai8b-q1-shared2"
MODEL_PATH="${WEIGHTS_DIR}/${MODEL_NAME}"
PORT=8082
LD_LIBRARY_PATH="/usr/local/lib/ollama/cuda_v12"

# Bonsai full-GPU parameters
NGL=999
CTX=8192
PARALLEL=2
KV_UNIFIED=true
KV_OFFLOAD=true
KV_QUANT="q8"

echo "Configuration:"
echo "  Model: ${MODEL_PATH}"
echo "  Port: ${PORT}"
echo "  NGL: ${NGL}"
echo "  Context: ${CTX}"
echo "  Parallel: ${PARALLEL}"
echo "  KV Unified: ${KV_UNIFIED}"
echo "  KV Offload: ${KV_OFFLOAD}"
echo "  KV Quant: ${KV_QUANT}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo ""

# Check if model exists
if [ ! -f "${MODEL_PATH}.gguf" ] && [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    echo "Please ensure the model is downloaded and available."
    exit 1
fi

echo "Model found: ${MODEL_PATH}"

# Check if port is available
if ss -tlnp | grep -q ":${PORT} "; then
    echo "WARNING: Port ${PORT} is already in use"
    read -p "Kill existing process and continue? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Find and kill the process using port 8082
        PID=$(ss -tlnp | grep ":${PORT} " | grep -o 'pid=[0-9]*' | grep -o '[0-9]*')
        if [ -n "$PID" ]; then
            echo "Killing process ${PID} on port ${PORT}"
            kill -9 "$PID" 2>/dev/null || true
            sleep 2
        fi
    else
        echo "Aborting."
        exit 1
    fi
fi

# Check for existing llama-server processes on GPU
echo "Checking for existing GPU processes..."
GPU_LLAMA=$(nvidia-smi --query-compute-apps=name,pid --format=csv 2>/dev/null | grep "llama-server" || echo "")

if [ -n "$GPU_LLAMA" ]; then
    echo "WARNING: Existing llama-server processes found on GPU:"
    echo "$GPU_LLAMA"
    read -p "Kill all llama-server processes and continue? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -9 llama-server 2>/dev/null || true
        sleep 2
        echo "Killed all llama-server processes"
    else
        echo "Aborting."
        exit 1
    fi
fi

# Build the command
CMD=(
    env
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
    /usr/local/bin/llama-server
    "${MODEL_PATH}"
    --port "${PORT}"
    --ngl "${NGL}"
    --ctx "${CTX}"
    --parallel "${PARALLEL}"
    --kv-unified
    --kv-offload
    --kv-quants "${KV_QUANT}"
)

echo "Starting Bonsai..."
echo "Command: ${CMD[*]}"
echo ""

# Start the server in background
exec "${CMD[@]}"
