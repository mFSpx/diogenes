#!/usr/bin/env bash
set -euo pipefail
ROOT="${LUCIDOTA_HOME:-/home/mfspx/LUCIDOTA}"
# Activate LUCIDOTA venv so psycopg/river/bytewax are available.
[[ -f "${ROOT}/.venv/bin/activate" ]] && source "${ROOT}/.venv/bin/activate"
PY="${PYTHON:-python3}"
BYTEWAX_LIMIT="${BYTEWAX_LIMIT:-250}"
RIVER_REFLEX_LIMIT="${RIVER_REFLEX_LIMIT:-5000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-10}"
export LUCIDOTA_HOME="$ROOT"
export DBOS_SYSTEM_DATABASE_URL="${DBOS_SYSTEM_DATABASE_URL:-postgresql:///lucidota_state}"
cd "$ROOT"
if [[ "${ONCE:-0}" == "1" ]]; then
  "$PY" scripts/lucidota_bytewax_mini.py --json --live-cursor --limit "$BYTEWAX_LIMIT"
  "$PY" scripts/lucidota_river_reflex.py --json --limit "$RIVER_REFLEX_LIMIT"
  exit 0
fi
while true; do
  "$PY" scripts/lucidota_bytewax_mini.py --json --live-cursor --limit "$BYTEWAX_LIMIT"
  "$PY" scripts/lucidota_river_reflex.py --json --limit "$RIVER_REFLEX_LIMIT"
  sleep "$SLEEP_SECONDS"
done
