#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL="$ROOT/01_REPOS/doggystyle"
CLAW="$ROOT/01_REPOS/claudecode/rust"
LUCIDOTA_VENV="$ROOT/.venv"
KERNEL_SMOKE_HOME="$(mktemp -d /tmp/lucidota-kernel-smoke.XXXXXX)"
CLAW_SMOKE_HOME="$(mktemp -d /tmp/lucidota-clawd-smoke.XXXXXX)"
trap 'rm -rf "$KERNEL_SMOKE_HOME" "$CLAW_SMOKE_HOME"' EXIT

cd "$ROOT"
if command -v psql >/dev/null 2>&1 && psql -d lucidota_graph -v ON_ERROR_STOP=1 >/dev/null <<'SQL'
LOAD 'age';
SET search_path = ag_catalog, "$user", public;
SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector;
SELECT create_graph('lucidota_check_graph');
SELECT * FROM cypher('lucidota_check_graph', $$ RETURN 1 AS ok $$) AS (ok agtype);
SELECT drop_graph('lucidota_check_graph', true);
SQL
then
  :
fi

if [[ ! -x "$LUCIDOTA_VENV/bin/python" ]]; then
  python3 -m venv "$LUCIDOTA_VENV"
fi
"$LUCIDOTA_VENV/bin/python" -m pip install dbos >/dev/null
DBOS_SYSTEM_DATABASE_URL="${DBOS_SYSTEM_DATABASE_URL:-postgresql://mfspx@/lucidota_state}" \
  "$LUCIDOTA_VENV/bin/python" scripts/lucidota_dbos_smoke.py
scripts/apply_lucidota_control_schema.sh
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_runtime_smoke.py
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_security_scan.py >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_code_language_scan.py >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_algos_smoke.py >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_source_policy_seed.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_river_reflex.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_cas_index.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_cas_gc.py --json >/dev/null

# CAS dual-write phantom proof: bytes without authoritative DB metadata must be reported, not hidden by the byte index.
DUAL_WRITE_VAULT="$(mktemp -d /tmp/lucidota-cas-dual.XXXXXX)"
PYTHONPATH="$ROOT/scripts" "$LUCIDOTA_VENV/bin/python" -c "from pathlib import Path; from lucidota_survey import store_cas; store_cas(Path('$DUAL_WRITE_VAULT'), b'phantom dual write proof')"
PYTHONPATH="$ROOT/scripts" "$LUCIDOTA_VENV/bin/python" scripts/lucidota_cas_gc.py --vault "$DUAL_WRITE_VAULT" --recover-journal --json | grep '"orphan_candidates": 1' >/dev/null
rm -rf "$DUAL_WRITE_VAULT"
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_workflow_registry.py >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_extractor_registry.py --json >/dev/null
psql "${DBOS_SYSTEM_DATABASE_URL:-postgresql://mfspx@/lucidota_state}" -v ON_ERROR_STOP=1 >/dev/null <<'SQL'
DO $$
DECLARE
  ev uuid;
  n integer;
BEGIN
  INSERT INTO lucidota_control.workflow_event (workflow_id, run_id, phase, status, source, detail)
  VALUES ('wake-trigger-smoke', gen_random_uuid()::text, 'wake', 'succeeded', 'check_diogenes', '{}'::jsonb)
  RETURNING event_id INTO ev;
  SELECT count(*) INTO n FROM lucidota_control.event_outbox WHERE event_id = ev;
  IF n <> 1 THEN
    RAISE EXCEPTION 'wake bus outbox trigger failed for %', ev;
  END IF;
END $$;
SQL
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_wake_bus.py --seed --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_bytewax_mini.py --live-cursor --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_treelite_router.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_hop_pivot.py https://example.com --fetch --keyword example --max-depth 1 --max-pivots 1 --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_body_capture.py https://example.com --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_body_capture_policy.py https://example.com --profile content_truth --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_body_capture_evidence.py https://example.com --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_browser_body_capture.py https://example.com --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_age_edges.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_big_board.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_indy_contract.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_indy_brief.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_indy_brief.py auth-list --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_indy_brief.py queue-list --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_wiki_query.py VIBESCONTROL --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_wake_bus_audit.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_dbos_signoff.py smoke --workflow dbos-smoke --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_dbos_dispatch.py workflow-replay --auto-approve --retries 1 -- --workflow dbos-smoke --limit 5 --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_dbos_dispatch.py --auto-approve --no-json-flag survey-protocol -- scripts/lucidota_survey.py --keyword optional >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_dbos_dispatch.py --auto-approve body-capture-capture -- https://example.com --disable-signal-gate >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_dbos_dispatch.py model-governor --auto-approve >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_dbos_watcher.py --seed --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_dbos_external_draft.py --target external://check --draft "check draft only" --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_dbos_drive_map.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_dbos_big_board.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_validator_noise_stress.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_model_governor_smoke.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_model_governor.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_indy_regression.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_cockpit.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_drive_manifest.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_auth_report.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_operator_demo.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_regression_dashboard.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_release_checklist.py --json >/dev/null
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_survey.py scripts/lucidota_survey.py --fetch --keyword optional >/dev/null

cd "$KERNEL"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
python -m pip install -e . pytest >/dev/null
python -m pytest -q
python scripts/diogenes_grpc_smoke.py --home "$KERNEL_SMOKE_HOME"

cd "$CLAW"
cargo test --workspace
cargo build --release -p claw-cli
./target/release/claw diogenes-smoke --home "$CLAW_SMOKE_HOME"
printf 'LUCIDOTA survey smoke\n' > "$CLAW_SMOKE_HOME/survey.txt"
./target/release/claw lucidota-survey "$CLAW_SMOKE_HOME/survey.txt" --keyword LUCIDOTA >/dev/null
cd "$ROOT"
"$LUCIDOTA_VENV/bin/python" scripts/lucidota_dbos_survey.py "$CLAW_SMOKE_HOME/survey.txt" --keyword LUCIDOTA >/dev/null
