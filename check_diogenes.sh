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
