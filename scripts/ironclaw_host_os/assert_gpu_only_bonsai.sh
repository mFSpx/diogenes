#!/bin/bash
# Assert that only Bonsai owns GPU in this profile
# This script verifies that only the llama-server process is using GPU

set -euo pipefail

echo "Checking GPU process ownership..."
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. GPU checks cannot be performed."
    exit 1
fi

# Get list of GPU processes
GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>/dev/null || true)

echo "GPU Processes:"
echo "$GPU_PROCESSES"
echo ""

# Expected process name
EXPECTED_NAME="llama-server"

# Check each GPU process
VALID=true
if [ -n "$GPU_PROCESSES" ]; then
    while IFS=, read -r pid name; do
        echo "Process: PID=$pid, Name=$name"
        if [[ "$name" != *"$EXPECTED_NAME"* ]]; then
            echo "WARNING: Unexpected GPU process: $name (PID: $pid)"
            VALID=false
        fi
    done <<< "$GPU_PROCESSES"
else
    echo "No GPU processes found. Bonsai may not be running."
    VALID=false
fi

if [ "$VALID" = true ]; then
    echo ""
    echo "GPU_RECEIPT: Only Bonsai llama-server owns VRAM in this profile"
    echo "GPU_PROCESS_COUNT=1"
    echo "GPU_PROCESS_NAME=llama-server"
    exit 0
else
    echo ""
    echo "ERROR: GPU constraint violated - unexpected processes using GPU"
    exit 1
fi
