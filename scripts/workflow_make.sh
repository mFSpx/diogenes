#!/usr/bin/env bash
# workflow_make.sh — harden any instruction via groq_chat_cli.py iteration
# Usage: ./scripts/workflow_make.sh "instruction" "quality_bar" [max_iter=5]
set -euo pipefail
INSTRUCTION="${1:?need instruction}"
QUALITY_BAR="${2:?need quality_bar}"
MAX_ITER="${3:-5}"
OUT="05_OUTPUTS/workflow_maker"
mkdir -p "$OUT"

for i in $(seq 1 "$MAX_ITER"); do
  RESULT=$(python3 scripts/groq_chat_cli.py \
    --prompt "Instruction: $INSTRUCTION
Quality bar: $QUALITY_BAR
Iteration: $i of $MAX_ITER
Previous feedback: ${FEEDBACK:-none}
Output ONLY code. ≤100 LOC." \
    --max-tokens 1200 --execute --json 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('response',{}).get('choices',[{}])[0].get('message',{}).get('content',''))" 2>/dev/null || echo "")

  LOC=$(echo "$RESULT" | wc -l)
  echo "[iter $i] LOC=$LOC"

  if python3 -c "import ast; ast.parse(open('/dev/stdin').read())" <<< "$RESULT" 2>/dev/null && [ "$LOC" -le 100 ]; then
    TS=$(date -u +%Y%m%dT%H%M%SZ)
    echo "$RESULT" > "$OUT/result_${TS}.py"
    echo "PASS: written to $OUT/result_${TS}.py (LOC=$LOC)"
    exit 0
  fi
  FEEDBACK="LOC=$LOC, fix syntax/length"
done
echo "PARTIAL: max iterations reached"
