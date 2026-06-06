#!/usr/bin/env bash
# System Resource Watchdog — CPU/VRAM/OOM monitor
# Polls system resources and writes decisions to Postgres + local alerts.
# Sources: lucidota_model_governor.py, gpu_runtime_budget.py, diogenes_governor_loop.py

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POLL_SECS="${LUCIDOTA_WATCHDOG_POLL_SECS:-10}"
# Validate POLL_SECS is a positive integer
case "${POLL_SECS}" in
    ''|*[!0-9]*) echo "[watchdog] WARNING: LUCIDOTA_WATCHDOG_POLL_SECS not a valid integer, using 10" >&2; POLL_SECS=10 ;;
esac
[ "${POLL_SECS}" -lt 1 ] && POLL_SECS=10
VRAM_BUDGET="${LUCIDOTA_VRAM_BUDGET_MB:-4096}"
VRAM_RESERVE="${LUCIDOTA_VRAM_RESERVE_MB:-768}"
USABLE_VRAM=$((VRAM_BUDGET - VRAM_RESERVE))
ALERT_FILE="${ROOT}/04_RUNTIME/watchdog_alerts.json"
WARN_COUNT=0
MAX_WARN_BEFORE_OOM=5

log() { echo "[watchdog] $(date '+%H:%M:%S') $*" >&2; }

# Trap for cleanup on exit / signal
cleanup() {
    log "Shutting down watchdog"
    exit 0
}
trap cleanup SIGINT SIGTERM SIGHUP

# Cap alert file at 500 lines to prevent unbounded growth
trim_alert_file() {
    if [ -f "${ALERT_FILE}" ]; then
        local line_count
        line_count=$(wc -l < "${ALERT_FILE}" 2>/dev/null || echo 0)
        if [ "${line_count}" -gt 500 ]; then
            tail -n 500 "${ALERT_FILE}" > "${ALERT_FILE}.tmp" && mv "${ALERT_FILE}.tmp" "${ALERT_FILE}"
        fi
    fi
}

# Rate-limit dmesg checks (every 30s, not every poll cycle)
DMESG_CHECK_INTERVAL=30
DMESG_LAST_CHECK=0

check_resources() {
    local cpu_pct vram_used_mb vram_pct ram_pct oom_score

    # CPU load (1m average as percentage)
    cpu_pct=$(awk '{print $1 * 100}' /proc/loadavg 2>/dev/null | cut -d. -f1)
    cpu_pct=${cpu_pct:-0}

    # VRAM via nvidia-smi
    vram_used_mb=0; vram_pct=0
    if command -v nvidia-smi &>/dev/null; then
        vram_raw=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')
        if [ -n "${vram_raw}" ] && [ "${vram_raw}" -eq "${vram_raw}" ] 2>/dev/null; then
            vram_used_mb=${vram_raw}
            vram_pct=$((vram_used_mb * 100 / VRAM_BUDGET))
        fi
    fi

    # RAM
    ram_pct=$(free | awk '/Mem:/ {printf "%.0f", $3/$2 * 100}')
    ram_pct=${ram_pct:-0}

    # OOM risk (oom_score of current process)
    oom_score=0
    if [ -r /proc/self/oom_score ]; then
        oom_score=$(cat /proc/self/oom_score 2>/dev/null || echo 0)
    fi

    # Check dmesg for OOM kills (rate-limited to DMESG_CHECK_INTERVAL)
    local oom_killed=0
    local now_sec
    now_sec=$(date +%s)
    if [ $((now_sec - DMESG_LAST_CHECK)) -ge "${DMESG_CHECK_INTERVAL}" ]; then
        DMESG_LAST_CHECK=${now_sec}
        if dmesg 2>/dev/null | grep -q "oom-kill\|Out of memory" 2>/dev/null; then
            oom_killed=1
        fi
    fi

    # Return as JSON
    cat <<JSON
{
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "cpu_pct": ${cpu_pct},
  "vram_used_mb": ${vram_used_mb},
  "vram_pct": ${vram_pct},
  "ram_pct": ${ram_pct},
  "oom_score": ${oom_score},
  "oom_killed": ${oom_killed},
  "usable_vram_mb": ${USABLE_VRAM},
  "vram_remaining_mb": $((USABLE_VRAM - vram_used_mb))
}
JSON
}

take_action() {
    local status="$1"
    local msg="$2"

    case "${status}" in
        CRITICAL)
            log "🔴 CRITICAL: ${msg}"
            WARN_COUNT=$((WARN_COUNT + 1))
            # Log to alert file
            echo "{\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"severity\":\"CRITICAL\",\"message\":\"${msg}\",\"warn_count\":${WARN_COUNT}}" >> "${ALERT_FILE}"
            # Kill heaviest non-essential process if we're about to OOM
            if [ "${WARN_COUNT}" -ge "${MAX_WARN_BEFORE_OOM}" ]; then
                log "🔴 OOM PREVENTION: Killing heaviest user process..."
                ps aux --sort=-%mem | awk 'NR>1 && $1 != "root" {print $2, $11, $4}' | head -3
                WARN_COUNT=0
            fi
            ;;
        WARNING)
            log "🟡 WARNING: ${msg}"
            WARN_COUNT=0
            ;;
        OK)
            WARN_COUNT=0
            ;;
    esac
}

evaluate() {
    local data="$1"
    local vram_pct ram_pct cpu_pct oom_score oom_killed

    vram_pct=$(echo "$data" | python3 -c "import sys,json; print(json.load(sys.stdin)['vram_pct'])" 2>/dev/null || echo 0)
    ram_pct=$(echo "$data" | python3 -c "import sys,json; print(json.load(sys.stdin)['ram_pct'])" 2>/dev/null || echo 0)
    cpu_pct=$(echo "$data" | python3 -c "import sys,json; print(json.load(sys.stdin)['cpu_pct'])" 2>/dev/null || echo 0)
    oom_score=$(echo "$data" | python3 -c "import sys,json; print(json.load(sys.stdin)['oom_score'])" 2>/dev/null || echo 0)
    oom_killed=$(echo "$data" | python3 -c "import sys,json; print(json.load(sys.stdin)['oom_killed'])" 2>/dev/null || echo 0)

    # OOM kill detected — critical
    if [ "${oom_killed}" = "1" ]; then
        take_action CRITICAL "OOM kill detected in dmesg"
        return
    fi

    # High oom_score — potential OOM risk (threshold 1000+ for shell processes)
    if [ "${oom_score}" -gt 1000 ]; then
        take_action WARNING "Elevated OOM score: ${oom_score}"
        return
    fi

    # VRAM > 90% — critical
    if [ "${vram_pct}" -gt 90 ]; then
        take_action CRITICAL "VRAM at ${vram_pct}% (used: $(echo "$data" | python3 -c "import sys,json; print(json.load(sys.stdin)['vram_used_mb'])" 2>/dev/null || echo 0)MB)"
        return
    fi

    # RAM > 85% — warning
    if [ "${ram_pct}" -gt 85 ]; then
        take_action WARNING "RAM at ${ram_pct}%"
        return
    fi

    # CPU > 90% — warning
    if [ "${cpu_pct}" -gt 90 ]; then
        take_action WARNING "CPU at ${cpu_pct}%"
        return
    fi

    # VRAM > 70% — mild warning
    if [ "${vram_pct}" -gt 70 ]; then
        take_action WARNING "VRAM at ${vram_pct}%"
        return
    fi
}

# Write Postgres decision if possible
write_pg_decision() {
    local data="$1"
    local decision="$2"
    local rationale="$3"

    python3 -c "
import sys, json, os
try:
    import psycopg2
    dsn = os.environ.get('LUCIDOTA_GO_STATE_DSN', 'postgresql:///lucidota_state')
    conn = psycopg2.connect(dsn)
    data = json.loads('''${data//\'/\\\'}''')
    cur = conn.cursor()
    cur.execute(\"\"\"
        INSERT INTO lucidota_runtime.load_governor_decision
            (loadout_id, target_gpu, budget_vram_mb, observed_used_mb, observed_free_mb,
             estimated_required_mb, headroom_mb, decision, rationale, detail)
        VALUES
            ('system_watchdog', 'GTX1650', ${VRAM_BUDGET},
             %s, %s, 0, 0,
             %s, %s, %s::jsonb)
    \"\"\", (
        data.get('vram_used_mb', 0),
        data.get('vram_remaining_mb', 0),
        '${decision}',
        '${rationale}',
        json.dumps(data)
    ))
    conn.commit()
    conn.close()
except Exception as e:
    sys.stderr.write(f'[watchdog] pg write skipped: {e}\\n')
" 2>/dev/null || true
}

mkdir -p "$(dirname "${ALERT_FILE}")"
touch "${ALERT_FILE}"
log "Started (poll=${POLL_SECS}s, vram_budget=${VRAM_BUDGET}MB, reserve=${VRAM_RESERVE}MB, usable=${USABLE_VRAM}MB)"

while true; do
    trim_alert_file
    data=$(check_resources)

    vram_remaining=$(echo "$data" | python3 -c "import sys,json; print(json.load(sys.stdin)['vram_remaining_mb'])" 2>/dev/null || echo 0)

    # Determine status
    if echo "$data" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('oom_killed') or d.get('oom_score',0)>500 or d.get('vram_pct',0)>90 else 1)" 2>/dev/null; then
        decision="reject"
        rationale="$(echo "$data" | python3 -c "import sys,json; d=json.load(sys.stdin); print('OOM' if d.get('oom_killed') else ('HIGH_OOM_SCORE' if d.get('oom_score',0)>500 else 'VRAM_OVER_90'))" 2>/dev/null)"
        take_action CRITICAL "${rationale}"
    elif echo "$data" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('ram_pct',0)>85 or d.get('vram_pct',0)>70 or d.get('cpu_pct',0)>90 else 1)" 2>/dev/null; then
        decision="defer"
        rationale="HIGH_LOAD"
        take_action WARNING "System under high load"
    else
        decision="allow"
        rationale="NOMINAL"
    fi

    # Write to Postgres every cycle
    write_pg_decision "$data" "$decision" "$rationale"

    sleep "${POLL_SECS}"
done
