#!/usr/bin/env bash
# Drains the corpus_chunk -> staging_packet (graph hypothesis field) backlog via Groq.
# Incremental (only un-bridged chunks), idempotent, Groq-hands (no Claude, no local model).
# Exits after 3 consecutive empty batches (backlog drained + crush idle) or MAXITER.
set -uo pipefail
ROOT="${LUCIDOTA_HOME:-/home/mfspx/LUCIDOTA}"
source "$ROOT/scripts/lucidota_safe_ops_env.sh" >/dev/null 2>&1
LOG="$ROOT/04_RUNTIME/corpus_to_graph_loop.log"
BATCH="${C2G_BATCH:-100}"
MAXITER="${C2G_MAXITER:-500}"
dry=0
echo "$(date -Iseconds) corpus->graph drain loop start batch=$BATCH" >> "$LOG"
for ((i=0; i<MAXITER; i++)); do
  out=$(python3 "$ROOT/scripts/corpus_to_graph.py" --limit "$BATCH" 2>&1 | tail -1)
  echo "$(date -Iseconds) $out" >> "$LOG"
  read_n=$(echo "$out" | grep -oE '"chunks_read": [0-9]+' | grep -oE '[0-9]+$')
  if [ "${read_n:-0}" -eq 0 ]; then dry=$((dry + 1)); else dry=0; fi
  [ "$dry" -ge 3 ] && { echo "$(date -Iseconds) backlog drained, exit" >> "$LOG"; break; }
  sleep 8
done
echo "$(date -Iseconds) corpus->graph drain loop exit" >> "$LOG"
