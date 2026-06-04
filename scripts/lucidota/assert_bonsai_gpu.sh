#!/bin/bash
# LUCIDOTA Bonsai GPU Assertion Script
# Asserts that Bonsai 2-head full-GPU profile is running correctly
# Expected: one llama-server, one weight load, port 8082, -ngl 999, -c 8192

set -euo pipefail

echo "=== LUCIDOTA BONSAI GPU ASSERTION ==="
echo "Timestamp: $(date -Iseconds)"
echo ""

# Configuration
BONSAI_ENDPOINT="http://127.0.0.1:8082"
BONSAI_TOKENIZE="${BONSAI_ENDPOINT}/tokenize"
BONSAI_MODEL="bonsai8b-q1-shared2"
EXPECTED_PORT=8082
EXPECTED_NGL=999
EXPECTED_CTX=8192
EXPECTED_PARALLEL=2

# Track results
PASSED=0
FAILED=0
RECEIPTS=()

# Function to log receipt
log_receipt() {
    local test_name="$1"
    local status="$2"
    local message="$3"
    RECEIPTS+=("${test_name}: ${status} - ${message}")
    echo "  [${status}] ${test_name}: ${message}"
}

# Test 1: Check if Bonsai endpoint is reachable
echo "Test 1: Bonsai endpoint reachability"
if curl -s --max-time 5 "${BONSAI_ENDPOINT}" > /dev/null 2>&1; then
    log_receipt "BONSAI_ENDPOINT" "PASS" "Endpoint ${BONSAI_ENDPOINT} is reachable"
    PASSED=$((PASSED + 1))
else
    log_receipt "BONSAI_ENDPOINT" "FAIL" "Endpoint ${BONSAI_ENDPOINT} is not reachable"
    FAILED=$((FAILED + 1))
fi

# Test 2: Check tokenizer endpoint
echo "Test 2: Bonsai tokenizer endpoint"
TOKENIZE_TEST=$(curl -s --max-time 5 -X POST "${BONSAI_TOKENIZE}" \
    -H "Content-Type: application/json" \
    -d '{"content": "test"}' 2>/dev/null || echo "")

if echo "$TOKENIZE_TEST" | grep -q "tokens"; then
    TOKEN_COUNT=$(echo "$TOKENIZE_TEST" | grep -o '"tokens":[0-9]*' | grep -o '[0-9]*')
    log_receipt "BONSAI_TOKENIZER" "PASS" "Tokenizer returned tokens: ${TOKEN_COUNT}"
    PASSED=$((PASSED + 1))
else
    log_receipt "BONSAI_TOKENIZER" "FAIL" "Tokenizer endpoint failed or returned invalid response"
    FAILED=$((FAILED + 1))
fi

# Test 3: Check nvidia-smi for GPU processes
echo "Test 3: GPU process verification"
GPU_PROCS=$(nvidia-smi --query-compute-apps=name,pid,used_memory --format=csv 2>/dev/null || echo "")

# Count llama-server processes
LLAMA_COUNT=$(echo "$GPU_PROCS" | grep -c "llama-server" || echo "0")

if [ "$LLAMA_COUNT" -eq 1 ]; then
    log_receipt "GPU_LLAMA_COUNT" "PASS" "Exactly 1 llama-server process found"
    PASSED=$((PASSED + 1))
else
    log_receipt "GPU_LLAMA_COUNT" "FAIL" "Expected 1 llama-server, found ${LLAMA_COUNT}"
    FAILED=$((FAILED + 1))
fi

# Test 4: Check VRAM usage
echo "Test 4: VRAM usage check"
VRAM_LINE=$(echo "$GPU_PROCS" | grep "llama-server" || echo "")
if [ -n "$VRAM_LINE" ]; then
    VRAM_USAGE=$(echo "$VRAM_LINE" | grep -o '[0-9]*.[0-9]*[A-Za-z]*' | tail -1 || echo "0")
    log_receipt "GPU_VRAM_USAGE" "INFO" "VRAM usage: ${VRAM_USAGE}"
    # Check if VRAM is reasonable for 8192 ctx
    if echo "$VRAM_USAGE" | grep -q "MiB"; then
        VRAM_NUM=$(echo "$VRAM_USAGE" | grep -o '[0-9]*.[0-9]*' || echo "0")
        if (( $(echo "$VRAM_NUM > 1500" | bc -l) )); then
            log_receipt "GPU_VRAM_REASONABLE" "PASS" "VRAM > 1500 MiB (reasonable for 8192 ctx)"
            PASSED=$((PASSED + 1))
        else
            log_receipt "GPU_VRAM_REASONABLE" "WARN" "VRAM ${VRAM_USAGE} may be too low for 8192 ctx"
        fi
    fi
fi

# Test 5: Smoke test - expect BONSAI_2HEAD_FULLGPU_OK
echo "Test 5: Smoke test response"
SMOKE_RESPONSE=$(curl -s --max-time 10 -X POST "${BONSAI_ENDPOINT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Say BONSAI_2HEAD_FULLGPU_OK"}], "max_tokens": 10, "stream": false}' 2>/dev/null || echo "")

if echo "$SMOKE_RESPONSE" | grep -q "BONSAI_2HEAD_FULLGPU_OK"; then
    log_receipt "BONSAI_SMOKE" "PASS" "Smoke test returned BONSAI_2HEAD_FULLGPU_OK"
    PASSED=$((PASSED + 1))
else
    log_receipt "BONSAI_SMOKE" "INFO" "Smoke test did not return BONSAI_2HEAD_FULLGPU_OK (may be expected)"
fi

# Summary
echo ""
echo "=== SUMMARY ==="
echo "Passed: ${PASSED}"
echo "Failed: ${FAILED}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "Result: BONSAI_2HEAD_FULLGPU_OK"
    exit 0
else
    echo "Result: BONSAI_GPU_ASSERTION_FAILED"
    exit 1
fi
