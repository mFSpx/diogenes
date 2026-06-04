#!/bin/bash
# LUCIDOTA GPU Exclusivity Assertion
# Asserts that only Bonsai owns GPU VRAM in this profile
# Needles/Mamba stages must NOT steal GPU from Bonsai

set -euo pipefail

echo "=== LUCIDOTA GPU EXCLUSIVITY ASSERTION ==="
echo "Timestamp: $(date -Iseconds)"
echo ""

# Track results
PASSED=0
FAILED=0
RECEIPTS=()

log_receipt() {
    local test_name="$1"
    local status="$2"
    local message="$3"
    RECEIPTS+=("${test_name}: ${status} - ${message}")
    echo "  [${status}] ${test_name}: ${message}"
}

# Test 1: Get all GPU compute processes
echo "Test 1: Enumerate GPU compute processes"
GPU_PROCS=$(nvidia-smi --query-compute-apps=name,pid,used_memory --format=csv 2>/dev/null || echo "")
echo "GPU Processes:"
echo "$GPU_PROCS"
echo ""

# Test 2: Check only llama-server is using GPU
echo "Test 2: Check for non-llama-server GPU processes"
NON_LLAMA=$(echo "$GPU_PROCS" | grep -v "llama-server" | grep -v "name,pid" || echo "")

if [ -z "$NON_LLAMA" ]; then
    log_receipt "GPU_EXCLUSIVITY" "PASS" "Only llama-server processes on GPU"
    PASSED=$((PASSED + 1))
else
    log_receipt "GPU_EXCLUSIVITY" "FAIL" "Non-llama-server processes found on GPU"
    FAILED=$((FAILED + 1))
    echo "  Offending processes:"
    echo "$NON_LLAMA" | while read line; do
        echo "    - $line"
    done
fi

# Test 3: Count llama-server processes
echo "Test 3: Count llama-server processes"
LLAMA_COUNT=$(echo "$GPU_PROCS" | grep -c "llama-server" || echo "0")

if [ "$LLAMA_COUNT" -eq 1 ]; then
    log_receipt "GPU_LLAMA_ONLY" "PASS" "Exactly 1 llama-server process (Bonsai)"
    PASSED=$((PASSED + 1))
else
    log_receipt "GPU_LLAMA_ONLY" "FAIL" "Expected 1 llama-server, found ${LLAMA_COUNT}"
    FAILED=$((FAILED + 1))
fi

# Test 4: Check for DeepSeek/Mamba GPU squatters on 8080/8083
echo "Test 4: Check for GPU squatters on reserved ports"
SQUATTERS=()

if ss -tlnp | grep -q ":8080 "; then
    SQUATTERS+=("8080")
fi
if ss -tlnp | grep -q ":8083 "; then
    SQUATTERS+=("8083")
fi

if [ ${#SQUATTERS[@]} -eq 0 ]; then
    log_receipt "GPU_PORT_SQUATTERS" "PASS" "No GPU squatters on 8080/8083"
    PASSED=$((PASSED + 1))
else
    log_receipt "GPU_PORT_SQUATTERS" "FAIL" "Found squatters on ports: ${SQUATTERS[*]}"
    FAILED=$((FAILED + 1))
fi

# Test 5: Verify LD_LIBRARY_PATH for CUDA
echo "Test 5: Verify CUDA library path"
EXPECTED_LD_LIB="/usr/local/lib/ollama/cuda_v12"
CURRENT_LD_LIB=$(env | grep -o 'LD_LIBRARY_PATH=[^:]*' | grep -o '/[^:]*$' || echo "")

if [ "$CURRENT_LD_LIB" = "$EXPECTED_LD_LIB" ]; then
    log_receipt "GPU_LD_LIBRARY" "PASS" "LD_LIBRARY_PATH includes CUDA v12"
    PASSED=$((PASSED + 1))
else
    log_receipt "GPU_LD_LIBRARY" "INFO" "LD_LIBRARY_PATH: ${CURRENT_LD_LIB}"
fi

# Summary
echo ""
echo "=== SUMMARY ==="
echo "Passed: ${PASSED}"
echo "Failed: ${FAILED}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "Result: GPU_EXCLUSIVITY_OK"
    echo "Receipt: Only Bonsai (llama-server) owns GPU VRAM"
    exit 0
else
    echo "Result: GPU_EXCLUSIVITY_VIOLATED"
    exit 1
fi
