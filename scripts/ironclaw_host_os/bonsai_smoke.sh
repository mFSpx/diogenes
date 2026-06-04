#!/bin/bash
# Bonsai smoke test - verify endpoint is responding

set -euo pipefail

echo "Running Bonsai smoke test..."
echo ""

# Test health endpoint
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" http://127.0.0.1:8082/health 2>/dev/null || echo "")
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -n1)
HEALTH_BODY=$(echo "$HEALTH_RESPONSE" | head -n1)

echo "Health endpoint HTTP code: $HTTP_CODE"
echo "Health endpoint body: $HEALTH_BODY"

if [ "$HTTP_CODE" = "200" ] && echo "$HEALTH_BODY" | grep -q '"status":"ok"'; then
    echo ""
    echo "BONSAI_2HEAD_FULLGPU_OK"
    exit 0
else
    echo ""
    echo "ERROR: Bonsai smoke test failed"
    exit 1
fi
