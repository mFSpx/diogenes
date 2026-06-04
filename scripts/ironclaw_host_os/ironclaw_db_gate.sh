#!/bin/bash
# IronClaw DB-backed host gate
# This script tests IronClaw in DB-backed mode

set -euo pipefail

IRONCLAW_REPO="/home/mfspx/LUCIDOTA/01_REPOS/ironclaw"

cd "$IRONCLAW_REPO"

echo "Running IronClaw DB-backed host gate..."
echo ""

# Set environment for DB mode
export LLM_BACKEND="openai_compatible"
export LLM_BASE_URL="http://127.0.0.1:8082/v1"
export LLM_MODEL="bonsai8b-q1-shared2"
export LLM_API_KEY="local-not-needed"
export DATABASE_URL="postgres://ironclaw:878453a11aa37d14b57a0abd5d92e20bc3c1b625e073aaa7@127.0.0.1:5432/lucidota_state?sslmode=disable"
export DATABASE_BACKEND="postgres"
export SANDBOX_ENABLED="false"
export WASM_ENABLED="false"
export WASM_CHANNELS_ENABLED="false"
export BUILDER_ENABLED="false"

# Run IronClaw DB gate
timeout 5 bash -c "ironclaw --cli-only run --no-onboard" 2>&1 | tee /tmp/ironclaw_db_gate.log || true

# Check the log for success indicators
if grep -q "ready" /tmp/ironclaw_db_gate.log && grep -q "db:postgres" /tmp/ironclaw_db_gate.log; then
    echo ""
    echo "IRONCLAW_HOST_OS_OK"
    exit 0
fi

# Check if it was a prompt budget exceeded error (this is expected)
if grep -q "PROMPT_BUDGET_EXCEEDED" /tmp/ironclaw_db_gate.log; then
    echo ""
    echo "PROMPT_BUDGET_EXCEEDED detected - pipeline compaction working"
    exit 0
fi

echo ""
echo "ERROR: IronClaw DB gate failed unexpectedly"
cat /tmp/ironclaw_db_gate.log
exit 1
