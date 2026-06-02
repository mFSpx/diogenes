#!/bin/bash

set -euo pipefail

LUCIDOTA_ROOT=/home/mfspx/LUCIDOTA
KRAMPUS=$LUCIDOTA_ROOT/KRAMPUSCHEWING
UNPACK_TMP=$LUCIDOTA_ROOT/09_STORAGE/krampuschewing_unpacked
PYTHON=$LUCIDOTA_ROOT/.venv/bin/python3
EXTRACTOR=$LUCIDOTA_ROOT/scripts/corpus_groq_extractor.py

DRY_RUN=0
if [ "$1" == "--dry-run" ]; then
  DRY_RUN=1
fi

check_disk() {
  local disk_usage=$(df -P / | awk 'NR==2{print $5}' | tr -d '%')
  if [ "$disk_usage" -gt 82 ]; then
    echo "Disk usage too high: $disk_usage%"
    exit 1
  fi
}

archives=(
  "docs_Luci-010.zip"
  "docs_NORDLEY_SQUEEZECOPY-001.zip"
  "docs_RICKSHAW_ROBBERY-002.zip"
  "docs_PHONE_BACKUP_MOTOGPLAY_20260415-011.zip"
  "Lucidota.zip"
  "NorthernStrike_ZIPPED_EVERYTHING_ELSE_20260508T0005-20260515T114241Z-3-003.zip"
  "NorthernStrike_ZIPPED_EVERYTHING_ELSE_20260508T0005-20260515T114241Z-3-004.zip"
  "NorthernStrike_ZIPPED_EVERYTHING_ELSE_20260508T0005-20260515T114241Z-3-006.zip"
  "NorthernStrike_ZIPPED_EVERYTHING_ELSE_20260508T0005-20260515T114241Z-3-008.zip"
  "Luci.zip"
  "NORTHERN_STRIKE-20260515T035215Z-3-001.zip"
  "meta-2026-Mar-23-23-17-05-20260515T034652Z-3-001.zip"
  "FIXME-20260515T034630Z-3-001.zip"
  "RICKSHAW_ROBBERY-20260514T060139Z-3-001.zip"
)

for archive in "${archives[@]}"; do
  if [ ! -f "$KRAMPUS/$archive" ]; then
    echo "Skipping missing archive: $archive"
    continue
  fi

  check_disk

  DEST=$UNPACK_TMP/$(basename "$archive" .zip)_$(date +%s)
  mkdir -p "$DEST"

  if [ "$DRY_RUN" -eq 0 ]; then
    unzip -q "$KRAMPUS/$archive" -d "$DEST" || { echo "UNZIP FAILED: skip"; rm -rf "$DEST"; continue; }
    echo "Extracted $archive -> $DEST"

    # Generate inventory of text files from extracted archive
    INVENTORY_TMP=/tmp/krampus_unpack_inventory_$$.jsonl
    find "$DEST" -type f \( -name "*.txt" -o -name "*.md" -o -name "*.py" -o -name "*.js" \
      -o -name "*.ts" -o -name "*.json" -o -name "*.jsonl" -o -name "*.sql" -o -name "*.csv" \
      -o -name "*.yaml" -o -name "*.yml" -o -name "*.sh" -o -name "*.toml" -o -name "*.conf" \
      -o -name "*.log" -o -name "*.html" -o -name "*.xml" -o -name "*.rst" \
      -o -name "*.eml" -o -name "*.mbox" \) \
      | $PYTHON -c "
import sys, json, os
for line in sys.stdin:
    p = line.strip()
    try:
        s = os.path.getsize(p)
        if s > 0 and s < 10*1024*1024:
            print(json.dumps({'path': p, 'size_bytes': s}))
    except: pass
" > "$INVENTORY_TMP" 2>/dev/null

    local inv_count
    inv_count=$(wc -l < "$INVENTORY_TMP")
    echo "  Inventory: $inv_count files to ingest"

    if [ "$inv_count" -gt 0 ]; then
      LUCIDOTA_SKIP_EMBED=1 $PYTHON "$LUCIDOTA_ROOT/scripts/corpus_ingest.py" \
        --inventory-jsonl "$INVENTORY_TMP" --execute --batch-size 50 2>&1 | tail -3 || true
    fi
    rm -f "$INVENTORY_TMP"
    rm -rf "$DEST"
    rm -f "$KRAMPUS/$archive"
    echo "DONE: $archive deleted"
  else
    echo "DRY RUN: Would extract $archive -> $DEST"
    echo "DRY RUN: Would ingest $DEST"
    echo "DRY RUN: Would delete $archive"
  fi

  check_disk
done
