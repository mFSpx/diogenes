#!/bin/bash
set -euo pipefail

HELPER_BIN="${HELPER_BIN:-python3 scripts/graph_materialization_helper.py}"
get_command_uuid() {
  python3 - <<'PY'
import psycopg
from psycopg.rows import dict_row
with psycopg.connect('postgresql:///lucidota_state', row_factory=dict_row) as conn:
    row = conn.execute("""
        SELECT command_uuid::text
        FROM lucidota_control.conversation_command
        WHERE status IN ('queued', 'staged', 'accepted', 'executed')
        ORDER BY created_at DESC
        LIMIT 1
    """).fetchone()
    if row:
        print(row["command_uuid"])
PY
}

COMMAND_UUID="$(get_command_uuid)"
if [[ -z "${COMMAND_UUID}" ]]; then
  echo "ERROR: no conversation_command UUID available for graph materialization helper." >&2
  exit 127
fi

echo "====================================================================="
echo "INITIATING DIRECT OROBOROS SPRINT VIA LUCI (THE RUST FORK)"
echo "====================================================================="

echo ">>> [1/4] IGNITING PILLAR: DIOGENES (KERNEL ROUTER)..."
$HELPER_BIN materialize --execute --operator-confirmed --command-envelope-uuid "$COMMAND_UUID" --candidate-payload-json '{"term":"CLAIM","label":"Modify core/telemetry/diogenes.py. Wire the asynchronous human I/O buffer and attach compressed_activity to SKIP LOCKED queue rows.","status":"staged","source_pillar":"diogenes"}' --evidence-ref "scripts/ignite_pillars.sh" --source-system "ignite_pillars.sh" --authority-class "operator_authored_assertion" --rationale "Materialize DIOGENES pillar through guarded helper path."

echo ">>> [2/4] IGNITING PILLAR: PERCYPHON.AI (CRYPTOGRAPHIC MASKS)..."
$HELPER_BIN materialize --execute --operator-confirmed --command-envelope-uuid "$COMMAND_UUID" --candidate-payload-json '{"term":"CLAIM","label":"Modify ALGOS/percyphon.py. Build the Procedural Entity Generator with SHA-256 UUID chunking and CPU-only arithmetic.","status":"staged","source_pillar":"percyphon"}' --evidence-ref "scripts/ignite_pillars.sh" --source-system "ignite_pillars.sh" --authority-class "operator_authored_assertion" --rationale "Materialize PERCYPHON pillar through guarded helper path."

echo ">>> [3/4] IGNITING PILLAR: THE LANGUAGE MEMBRANE (MULTIPLEXER)..."
$HELPER_BIN materialize --execute --operator-confirmed --command-envelope-uuid "$COMMAND_UUID" --candidate-payload-json '{"term":"CLAIM","label":"Modify core/language_membrane.py. Build the 4-lane Weaver with templates, MinHash retrievals, DeepSeek synthesis, and FairyFuse smoothing.","status":"staged","source_pillar":"language_membrane"}' --evidence-ref "scripts/ignite_pillars.sh" --source-system "ignite_pillars.sh" --authority-class "operator_authored_assertion" --rationale "Materialize LANGUAGE MEMBRANE pillar through guarded helper path."

echo ">>> [4/4] IGNITING PILLAR: THE TICKLETRUNK (EVOLUTIONARY WARCHEST)..."
$HELPER_BIN materialize --execute --operator-confirmed --command-envelope-uuid "$COMMAND_UUID" --candidate-payload-json '{"term":"CLAIM","label":"Modify core/tickletrunk_war_chest.py. Run pytest on new code and index passes via tickletrunk_scan.","status":"staged","source_pillar":"tickletrunk_war_chest"}' --evidence-ref "scripts/ignite_pillars.sh" --source-system "ignite_pillars.sh" --authority-class "operator_authored_assertion" --rationale "Materialize TICKLETRUNK pillar through guarded helper path."

echo "====================================================================="
echo "SWARM EXECUTION COMPLETE. VERIFYING PHYSICS..."
echo "====================================================================="
python3 -m pytest math/ -v
