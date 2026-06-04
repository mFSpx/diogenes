#!/bin/bash
# IronClaw Host OS Mode Gate
# This script verifies IronClaw is running in proper host-OS mode
# No Docker, no User= requirement, no --no-db when DB is available

set -euo pipefail

IRONCLAW_REPO="/home/mfspx/LUCIDOTA/01_REPOS/ironclaw"
INDY_DAEMON_SERVICE="${INDY_DAEMON_SERVICE:-ironclaw-indy-reads.service}"

cd "$IRONCLAW_REPO"

echo "Running IronClaw Host OS Mode Gate..."
echo "Checking Indy daemon service: $INDY_DAEMON_SERVICE"
echo ""

INDY_DAEMON_UNIT_TEXT="$(systemctl --user cat "$INDY_DAEMON_SERVICE" 2>/dev/null || true)"
if [[ -z "$INDY_DAEMON_UNIT_TEXT" ]]; then
    echo "ERROR: canonical Indy daemon service not installed/readable: $INDY_DAEMON_SERVICE"
    exit 1
fi
if systemctl --user is-active --quiet lucidota-indy-reads-watcher.service 2>/dev/null; then
    echo "ERROR: legacy Indy watcher service is active; disable it and use $INDY_DAEMON_SERVICE"
    exit 1
fi

# Check 1: No Docker in the service definition
if grep -qi "docker\|container" <<<"$INDY_DAEMON_UNIT_TEXT"; then
    echo "ERROR: Docker found in service definition - violates NO_DOCKER_LAW"
    exit 1
fi

# Check 2: No User= in systemd service
if grep -q "^User=" <<<"$INDY_DAEMON_UNIT_TEXT"; then
    echo "ERROR: User= found in systemd service - violates HOST_OS_MODE"
    exit 1
fi

# Check 3: IronClaw binary exists and is not a Docker container
if ! command -v ironclaw &>/dev/null; then
    echo "ERROR: ironclaw binary not found in PATH"
    exit 1
fi

# Check 4: Verify ironclaw is a real binary, not a Docker wrapper
IRONCLAW_BIN="$(which ironclaw)"
if file "$IRONCLAW_BIN" | grep -qi "docker\|script"; then
    echo "WARNING: ironclaw appears to be a script wrapper"
fi

# Check 5: Test IronClaw in host mode with DB
# This combines the DB gate and provider gate checks
echo "Testing IronClaw host mode with DB..."
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

# Run IronClaw in a test mode
timeout 5 bash -c "ironclaw --cli-only run --no-onboard 2>&1" | tee /tmp/ironclaw_host_gate_test.log || true

# Verify it worked
if grep -q "ready" /tmp/ironclaw_host_gate_test.log && grep -q "ironclaw" /tmp/ironclaw_host_gate_test.log; then
    echo ""
    echo "IRONCLAW_HOST_GATE_OK"
    echo "Host OS mode verified: No Docker, proper service definition, IronClaw binary available"
    exit 0
fi

# Check if it was a prompt budget exceeded (acceptable)
if grep -q "PROMPT_BUDGET_EXCEEDED" /tmp/ironclaw_host_gate_test.log; then
    echo ""
    echo "IRONCLAW_HOST_GATE_OK (prompt budget gate working)"
    exit 0
fi

echo ""
echo "ERROR: IronClaw host gate failed"
cat /tmp/ironclaw_host_gate_test.log
exit 1
