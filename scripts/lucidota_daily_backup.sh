#!/usr/bin/env bash
# lucidota_daily_backup.sh — once-per-day backup to GitHub + DoltHub
# Run via cron: 0 2 * * * source ~/.config/lucidota/secrets.env && bash /home/mfspx/LUCIDOTA/scripts/lucidota_daily_backup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOLT_DIR="$ROOT/04_RUNTIME/dolt_backup"
LOG="$ROOT/05_OUTPUTS/receipts/daily_backup_$(date +%Y%m%dT%H%M%SZ).json"

echo "{ \"started_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"," > "$LOG"

# 1. GitHub push
echo "  \"github\": " >> "$LOG"
if git -C "$ROOT" -c credential.helper='!gh auth git-credential' push diogenes main 2>&1; then
    echo "\"ok\"," >> "$LOG"
    echo "[backup] GitHub: pushed"
else
    echo "\"failed\"," >> "$LOG"
    echo "[backup] GitHub: failed (non-fatal)"
fi

# 2. Dolt export — schema snapshots only (no raw data, no secrets)
echo "  \"dolt\": " >> "$LOG"
if command -v dolt &>/dev/null && [[ -d "$DOLT_DIR/.dolt" ]]; then
    cd "$DOLT_DIR"
    # Export only schema files (06_SCHEMA/*.sql) and key docs
    cp -r "$ROOT/06_SCHEMA" . 2>/dev/null || true
    cp "$ROOT/00_PROJECT_BRAIN/SYSTEM_MAP_FULL.md" . 2>/dev/null || true
    cp "$ROOT/00_PROJECT_BRAIN/DOLTHUB_PROMPTFLOW_40_SHORTLIST.md" . 2>/dev/null || true
    dolt add -A
    dolt commit --skip-empty -m "daily-backup: schema+docs $(date +%Y-%m-%d)" 2>/dev/null && \
        echo "\"ok\"" >> "$LOG" || echo "\"no_changes\"" >> "$LOG"
    echo "[backup] Dolt: committed"
else
    echo "\"skipped_dolt_offline\"" >> "$LOG"
    echo "[backup] Dolt: offline or not init'd, skipped"
fi

echo ", \"finished_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" }" >> "$LOG"
echo "[backup] Receipt: $LOG"
