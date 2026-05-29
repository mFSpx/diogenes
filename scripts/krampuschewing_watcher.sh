#!/usr/bin/env bash
set -euo pipefail

ROOT="${LUCIDOTA_HOME:-/home/mfspx/LUCIDOTA}"
WATCH_DIR="${KRAMPUScHEWING_DIR:-$ROOT/KRAMPUSCHEWING}"
DIGESTED_DIR="${KRAMPUScHEWING_DIGESTED_DIR:-$ROOT/03_VAULT/korpus_krampii/DIGESTED}"
LOG_DIR="$ROOT/04_RUNTIME"
LOG_FILE="$LOG_DIR/krampuschewing_watcher.log"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then PY="python3"; fi
CASE_NAME="${KRAMPUScHEWING_CASE:-KORPUS KRAMPII}"
WORKERS="${KRAMPUScHEWING_WORKERS:-4}"
QUIET_SECONDS="${KRAMPUScHEWING_QUIET_SECONDS:-2}"
MAX_FILE_MB="${KRAMPUScHEWING_MAX_FILE_MB:-64}"
MAX_TEXT_MB="${KRAMPUScHEWING_MAX_TEXT_MB:-8}"
HARDMATH="${KRAMPUScHEWING_HARDMATH:-1}"
HARDMATH_LIMIT="${KRAMPUScHEWING_HARDMATH_LIMIT:-50000}"
HARDMATH_OUTLIERS="${KRAMPUScHEWING_HARDMATH_OUTLIERS:-100}"
RECHRONO="${KRAMPUScHEWING_RECHRONO:-1}"
RECHRONO_HOURS="${KRAMPUScHEWING_RECHRONO_HOURS:-168}"

mkdir -p "$WATCH_DIR" "$DIGESTED_DIR" "$LOG_DIR"

echo "[$(date -Iseconds)] KRAMPUSCHEWING awake: watch=$WATCH_DIR case=$CASE_NAME workers=$WORKERS" | tee -a "$LOG_FILE"

wait_until_stable() {
  local target="$1"
  local last_size="-1"
  local size="0"
  while true; do
    [[ -e "$target" ]] || return 1
    size=$(du -sb "$target" 2>/dev/null | awk '{print $1}') || return 1
    if [[ "$size" == "$last_size" ]]; then
      sleep "$QUIET_SECONDS"
      local size2
      size2=$(du -sb "$target" 2>/dev/null | awk '{print $1}') || return 1
      [[ "$size2" == "$size" ]] && return 0
    fi
    last_size="$size"
    sleep "$QUIET_SECONDS"
  done
}

is_probable_chatdump() {
  local target="$1"
  local name low
  name="$(basename "$target")"
  low="${name,,}"
  if [[ "$low" == *openai* || "$low" == *chatgpt* || "$low" == *claude* || "$low" == *anthropic* ]]; then
    return 0
  fi
  if [[ -f "$target" && "$low" == *.zip ]]; then
    if unzip -Z1 "$target" 2>/dev/null | grep -Eiq '(^|/)(conversations|conversation|chat|chats|messages|message)\.json$'; then
      return 0
    fi
  fi
  if [[ -d "$target" ]]; then
    if find "$target" -maxdepth 3 -type f \( -iname 'conversations.json' -o -iname 'messages.json' -o -iname '*openai*.zip' -o -iname '*chatgpt*.zip' -o -iname '*claude*.zip' -o -iname '*anthropic*.zip' \) | head -1 | grep -q .; then
      return 0
    fi
  fi
  return 1
}

is_probable_commdump() {
  local target="$1"
  local name low
  name="$(basename "$target")"
  low="${name,,}"
  if [[ "$low" == *facebook* || "$low" == *meta* || "$low" == *messenger* || "$low" == *email* || "$low" == *gmail* || "$low" == *mail* || "$low" == *takeout* || "$low" == *sms* || "$low" == *text* || "$low" == *imessage* || "$low" == *whatsapp* || "$low" == *signal* ]]; then
    return 0
  fi
  if [[ -f "$target" && ( "$low" == *.json || "$low" == *.jsonl || "$low" == *.ndjson || "$low" == *.zip ) ]]; then
    return 0
  fi
  if [[ -d "$target" ]]; then
    if find "$target" -maxdepth 4 -type f \( -iname '*.json' -o -iname '*.jsonl' -o -iname '*.ndjson' -o -iname '*.zip' -o -iname 'message_*.json' \) | head -1 | grep -q .; then
      return 0
    fi
  fi
  return 1
}

run_commdump_pipeline() {
  local target="$1"
  echo "[$(date -Iseconds)] COMMDUMP UNIVERSAL AUTO-INGEST: $target" | tee -a "$LOG_FILE"
  "$PY" "$ROOT/scripts/lucidota_commdump_timeline.py" --json ingest --source-kind auto "$target" >> "$LOG_FILE" 2>&1 || return 1
  echo "[$(date -Iseconds)] DECISION DELTA AUTO-RUN (COMMDUMP)" | tee -a "$LOG_FILE"
  "$PY" "$ROOT/scripts/lucidota_decision_delta.py" --json run --source both >> "$LOG_FILE" 2>&1 || return 1
  echo "[$(date -Iseconds)] COMMDUMP STATUS" >> "$LOG_FILE"
  "$PY" "$ROOT/scripts/lucidota_commdump_timeline.py" --json status >> "$LOG_FILE" 2>&1 || true
}

run_chatdump_pipeline() {
  local target="$1"
  echo "[$(date -Iseconds)] CHATDUMP TIMELINE AUTO-INGEST: $target" | tee -a "$LOG_FILE"
  "$PY" "$ROOT/scripts/lucidota_chatdump_timeline.py" --json ingest --provider auto "$target" >> "$LOG_FILE" 2>&1 || return 1
  echo "[$(date -Iseconds)] DECISION DELTA AUTO-RUN" | tee -a "$LOG_FILE"
  "$PY" "$ROOT/scripts/lucidota_decision_delta.py" --json run --source both >> "$LOG_FILE" 2>&1 || return 1
  echo "[$(date -Iseconds)] CHATDUMP STATUS" >> "$LOG_FILE"
  "$PY" "$ROOT/scripts/lucidota_chatdump_timeline.py" --json status >> "$LOG_FILE" 2>&1 || true
  "$PY" "$ROOT/scripts/lucidota_decision_delta.py" --json status >> "$LOG_FILE" 2>&1 || true
}

run_brain_sidecar() {
  local target="$1"
  if [[ -d "$target" ]]; then
    echo "[$(date -Iseconds)] BRAIN CHRONO SIDECAR AUTO-RUN: $target" | tee -a "$LOG_FILE"
    MODE=brain-only RESET_BRAIN=0 MAX_FILES=0 "$ROOT/scripts/chronological_migrator.sh" "$target" "$CASE_NAME" >> "$LOG_FILE" 2>&1 || return 1
  elif [[ -f "$target" ]]; then
    case "${target,,}" in
      *.md|*.mdx|*.txt|*.log|*.json|*.jsonl|*.yaml|*.yml)
        echo "[$(date -Iseconds)] BRAIN SINGLE-FILE SIDECAR: $target" | tee -a "$LOG_FILE"
        "$PY" "$ROOT/scripts/lucidota_brain_ingest.py" --json "$target" >> "$LOG_FILE" 2>&1 || return 1
        ;;
    esac
  fi
}

run_hardmath_pipeline() {
  [[ "$HARDMATH" == "1" ]] || return 0
  echo "[$(date -Iseconds)] HARD TRUTH MATH AUTO-RUN" | tee -a "$LOG_FILE"
  "$PY" "$ROOT/scripts/lucidota_hard_truth_math.py" --json run --limit "$HARDMATH_LIMIT" --outliers "$HARDMATH_OUTLIERS" >> "$LOG_FILE" 2>&1
}

run_rechrono_pipeline() {
  [[ "$RECHRONO" == "1" ]] || return 0
  echo "[$(date -Iseconds)] KRAMPUS FORENSIC RECHRONO REFRESH recent_hours=$RECHRONO_HOURS" | tee -a "$LOG_FILE"
  "$PY" "$ROOT/scripts/krampus_rechronologize.py" --json refresh --recent-hours "$RECHRONO_HOURS" >> "$LOG_FILE" 2>&1
}

run_korpus_pipeline() {
  local target="$1"
  echo "[$(date -Iseconds)] KORPUS HARD INGEST: $target" | tee -a "$LOG_FILE"
  local args=(--json ingest "$target" --case "$CASE_NAME" --label "krampuschewing-drop" --max-file-mb "$MAX_FILE_MB" --max-text-mb "$MAX_TEXT_MB")
  if [[ -d "$target" ]]; then
    # Directories are time-machine batches: extract content/path dates and learn sequentially.
    args+=(--chronological --workers 1)
  else
    args+=(--workers "$WORKERS")
  fi
  "$PY" "$ROOT/scripts/korpus_krampii.py" "${args[@]}" >> "$LOG_FILE" 2>&1
}

move_to_digested() {
  local target="$1"
  if [[ "$target" == "$WATCH_DIR"/* && -e "$target" ]]; then
    local base dest
    base="$(basename "$target")"
    dest="$DIGESTED_DIR/$(date +%Y%m%dT%H%M%S).$base"
    mv "$target" "$dest" 2>/dev/null || true
  fi
}

chew_path() {
  local target="$1"
  [[ -e "$target" ]] || return 0
  wait_until_stable "$target" || return 0
  echo "[$(date -Iseconds)] KRAMPUS IS CHEWING HARD: $target" | tee -a "$LOG_FILE"
  local ok=1
  local chatdump=0
  if is_probable_chatdump "$target"; then
    run_chatdump_pipeline "$target" || ok=0
    chatdump=1
  fi
  if [[ "$chatdump" != "1" ]] && is_probable_commdump "$target"; then
    run_commdump_pipeline "$target" || ok=0
  fi
  run_brain_sidecar "$target" || ok=0
  run_korpus_pipeline "$target" || ok=0
  run_rechrono_pipeline || ok=0
  run_hardmath_pipeline || ok=0
  if [[ "$ok" == "1" ]]; then
    echo "[$(date -Iseconds)] KRAMPUS DIGESTED: $target" | tee -a "$LOG_FILE"
    move_to_digested "$target"
  else
    echo "[$(date -Iseconds)] KRAMPUS FAILED: $target" | tee -a "$LOG_FILE"
  fi
}

# Initial sweep: chew anything already waiting in the belly.
find "$WATCH_DIR" -mindepth 1 -maxdepth 1 -print0 | while IFS= read -r -d '' item; do
  chew_path "$item"
done

# Kernel event loop: no polling; trigger when copies finish or files/dirs are moved in.
inotifywait -m -r -e close_write,moved_to,create --format '%w%f' "$WATCH_DIR" | while IFS= read -r new_path; do
  # For nested file creates, chew only the top-level drop directory/file.
  top="$new_path"
  rel="${new_path#"$WATCH_DIR"/}"
  first="${rel%%/*}"
  if [[ -n "$first" && "$first" != "$rel" ]]; then
    top="$WATCH_DIR/$first"
  fi
  chew_path "$top"
done
