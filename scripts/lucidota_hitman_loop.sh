#!/usr/bin/env bash
set -euo pipefail

LOG="${LUCIDOTA_HITMAN_LOOP_LOG:-/tmp/lucidota_hitman_loop.log}"
mkdir -p "$(dirname "$LOG")"
LOCK="${LUCIDOTA_HITMAN_LOOP_LOCK:-/tmp/lucidota_hitman_loop.lock}"

if [[ "${LUCIDOTA_HITMAN_LOOP_INHIBIT:-0}" != "1" ]] && command -v systemd-inhibit >/dev/null 2>&1; then
    if env LUCIDOTA_HITMAN_LOOP_INHIBIT=1 systemd-inhibit \
      --who=lucidota-hitman-loop \
      --why="Prevent sleep during autonomous repair loop" \
      --what=idle:sleep:shutdown:handle-lid-switch \
      --mode=block \
      "$0" >> "$LOG" 2>&1; then
      exit 0
    else
      inhibit_rc=$?
      echo "HITMAN_LOOP_INHIBIT_SKIPPED=$(date -u +%FT%TZ) reason=access_denied_or_unavailable rc=$inhibit_rc" >> "$LOG"
    fi
fi

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "HITMAN_LOOP_ALREADY_RUNNING=$(date -u +%FT%TZ)" >> "$LOG" 2>/dev/null || true
    exit 0
fi

while true; do
    {
    # 1. Force clear cache lines and stale model slots
    rm -rf /home/mfspx/LUCIDOTA/04_RUNTIME/inference_os/llama_slots/*

    # 2. Re-index current truth from Postgres
    python3 scripts/system_runtime_facts_refresh.py --execute

    # 3. Fire local Spark lane with explicit context reset and short token budget.
    if ! python3 scripts/model_runner_cli.py local-chat --lane spark --prompt "ping" --clear-history --max-tokens 120 --json --execute; then
      if ! python3 scripts/model_runner_cli.py cohere-chat --prompt ping --max-tokens 120 --json --execute; then
        python3 scripts/groq_chat_cli.py --prompt ping --max-tokens 120 --json --execute || true
      fi
    fi

    # 4. Check if the database facts marked it done. If yes, break.
    if psql postgresql:///lucidota_state -Atc "SELECT fact_value->>'status' FROM lucidota_control.runtime_status_fact WHERE fact_key='llxprt_groq_login_wired'" | grep -q "complete"; then
        echo "LUCIDOTA SYSTEM FULLY OPERATIONAL. LOOP TERMINATED."
        break
    fi

    # 5. Sleep 30 seconds to breathe between sessions, then cycle
    sleep 30
    } >> "$LOG" 2>&1
done
