#!/usr/bin/env bash
# KORPUS KRAMPII chronological migrator.
# Preserves River/DBSTREAM concept-drift order: OS birth time when available, else mtime.
set -euo pipefail

ROOT="/home/mfspx/LUCIDOTA"
PY="$ROOT/.venv/bin/python"
TARGET_DIR="${1:-$ROOT/KRAMPUSCHEWING}"
CASE_NAME="${2:-${CASE_NAME:-Krampus Holiday Pogroms}}"
MODE="${MODE:-both}"              # both | brain-only | korpus-only
LABEL="${LABEL:-chronological-migration}"
LOG="${LOG:-$ROOT/04_RUNTIME/chronological_migrator.log}"
MAX_FILES="${MAX_FILES:-0}"
RESET_BRAIN="${RESET_BRAIN:-0}"
MAX_FILE_MB="${MAX_FILE_MB:-64}"
MAX_TEXT_MB="${MAX_TEXT_MB:-8}"
COMMIT_EVERY="${COMMIT_EVERY:-25}"
INCLUDE_HIDDEN="${INCLUDE_HIDDEN:-0}"
NO_DEFAULT_EXCLUDES="${NO_DEFAULT_EXCLUDES:-0}"
BRAIN_STATE="${BRAIN_STATE:-$ROOT/03_VAULT/krampus_dbstream_brain.pkl}"
BRAIN_MAP="${BRAIN_MAP:-$ROOT/05_OUTPUTS/korpus_krampii/krampus_brain_map.jsonl}"
CHRONO_LEDGER="${CHRONO_LEDGER:-$ROOT/05_OUTPUTS/korpus_krampii/latest_chrono_ledger.csv}"
BRAIN_ABSURD_CHILD="${BRAIN_ABSURD_CHILD:-1}"
BRAIN_CHILD_TIMEOUT_SECONDS="${BRAIN_CHILD_TIMEOUT_SECONDS:-0}"  # 0 = ABSURD wrapper dynamic policy.

mkdir -p "$ROOT/04_RUNTIME" "$ROOT/05_OUTPUTS/korpus_krampii" "$ROOT/03_VAULT"
cd "$ROOT"

echo "[$(date -Iseconds)] chronological migrator start mode=$MODE target=$TARGET_DIR case=$CASE_NAME" | tee -a "$LOG"

if [[ "$RESET_BRAIN" == "1" ]]; then
  rm -f "$BRAIN_STATE" "$BRAIN_MAP"
  echo "[$(date -Iseconds)] reset brain state/map" | tee -a "$LOG"
fi

ledger_args=(build "$TARGET_DIR" --out "$CHRONO_LEDGER")
paths_args=(paths0 "$TARGET_DIR")
if (( MAX_FILES > 0 )); then ledger_args+=(--limit "$MAX_FILES"); paths_args+=(--limit "$MAX_FILES"); fi
if [[ "$INCLUDE_HIDDEN" == "1" ]]; then ledger_args+=(--include-hidden); paths_args+=(--include-hidden); fi
if [[ "$NO_DEFAULT_EXCLUDES" == "1" ]]; then ledger_args+=(--no-default-excludes); paths_args+=(--no-default-excludes); fi

"$PY" "$ROOT/scripts/krampus_chrono_ledger.py" --json "${ledger_args[@]}" | tee -a "$LOG"

if [[ "$MODE" == "brain-only" || "$MODE" == "both" ]]; then
  count=0
  while IFS= read -r -d '' path; do
    ((count += 1))
    echo "[$(date -Iseconds)] brain[$count]: $path" >> "$LOG"
    if [[ "$BRAIN_ABSURD_CHILD" == "1" ]]; then
      "$PY" "$ROOT/scripts/lucidota_absurd_brain_child.py" --json run-file \
        --timeout-seconds "$BRAIN_CHILD_TIMEOUT_SECONDS" \
        --state-path "$BRAIN_STATE" \
        --map-jsonl "$BRAIN_MAP" \
        "$path" >> "$LOG" 2>&1 || true
    else
      "$PY" "$ROOT/scripts/lucidota_brain_ingest.py" --json --state-path "$BRAIN_STATE" --map-jsonl "$BRAIN_MAP" "$path" >> "$LOG" 2>&1 || true
    fi
  done < <("$PY" "$ROOT/scripts/krampus_chrono_ledger.py" "${paths_args[@]}")
  echo "[$(date -Iseconds)] brain chronological run complete files_seen=$count map=$BRAIN_MAP" | tee -a "$LOG"
fi

if [[ "$MODE" == "korpus-only" || "$MODE" == "both" ]]; then
  args=(--json ingest "$TARGET_DIR" --case "$CASE_NAME" --label "$LABEL" --chronological --workers 1 --max-file-mb "$MAX_FILE_MB" --max-text-mb "$MAX_TEXT_MB" --commit-every "$COMMIT_EVERY")
  if (( MAX_FILES > 0 )); then args+=(--limit "$MAX_FILES"); fi
  if [[ "$INCLUDE_HIDDEN" == "1" ]]; then args+=(--include-hidden); fi
  if [[ "$NO_DEFAULT_EXCLUDES" == "1" ]]; then args+=(--no-default-excludes); fi
  echo "[$(date -Iseconds)] korpus chronological durable ingest start" | tee -a "$LOG"
  "$PY" "$ROOT/scripts/korpus_krampii.py" "${args[@]}" | tee -a "$LOG"
fi

echo "[$(date -Iseconds)] chronological migrator done" | tee -a "$LOG"
