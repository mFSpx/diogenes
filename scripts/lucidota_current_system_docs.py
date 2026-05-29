#!/usr/bin/env python3
"""Refresh the current human/robot operating docs from live repo + Postgres state."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "system_docs"
PB = ROOT / "00_PROJECT_BRAIN"
EXCLUDE = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def direct_dirs(path: Path) -> list[Path]:
    return [p for p in sorted(path.iterdir(), key=lambda p: p.name) if p.is_dir() and p.name not in EXCLUDE]


def tree_lines(path: Path, depth: int = 2, indent: int = 0) -> list[str]:
    if depth < 0:
        return []
    label = path.name + "/" if path.is_dir() else path.name
    rows = ["  " * indent + f"- {label}"]
    if depth == 0 or not path.is_dir():
        return rows
    for child in direct_dirs(path):
        rows.extend(tree_lines(child, depth=depth - 1, indent=indent + 1))
    return rows


def tree_index() -> dict[str, Any]:
    top = []
    for path in sorted(ROOT.iterdir(), key=lambda p: p.name):
        if not path.is_dir():
            continue
        if path.name in EXCLUDE:
            continue
        children = direct_dirs(path)
        top.append(
            {
                "path": rel(path),
                "child_dirs": [rel(c) for c in children],
                "child_dir_count": len(children),
                "top_line": tree_lines(path, depth=1),
            }
        )
    return {"generated_at": now(), "root": rel(ROOT), "trees": top}


def db_counts(conn: psycopg.Connection, schema: str) -> dict[str, Any]:
    cur = conn.cursor()
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema=%s ORDER BY 1", (schema,))
    tables = [r[0] for r in cur.fetchall()]
    counts = {}
    for table in tables:
        cur.execute(f"SELECT count(*) FROM {schema}.{table}")
        counts[table] = int(cur.fetchone()[0])
    return {"schema": schema, "table_count": len(tables), "tables": tables, "counts": counts}


def postgres_audit() -> dict[str, Any]:
    state_dsn = os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"
    storage_dsn = os.environ.get("KORPUS_DATABASE_URL") or "postgresql:///lucidota_storage"
    payload: dict[str, Any] = {"generated_at": now(), "state_dsn": state_dsn, "storage_dsn": storage_dsn}
    with psycopg.connect(state_dsn) as conn:
        payload["state_schemas"] = {}
        for schema in ["lucidota_control", "lucidota_learning", "lucidota_runtime", "lucidota_go", "lucidota_indy", "lucidota_pivot", "lucidota_ternary", "lucidota_vault", "lucidota_bus"]:
            payload["state_schemas"][schema] = db_counts(conn, schema)
        cur = conn.cursor()
        cur.execute("SELECT fact_key, fact_value FROM lucidota_control.runtime_status_fact WHERE subsystem='system' ORDER BY fact_key")
        payload["state_runtime_facts"] = {k: v for k, v in cur.fetchall()}
        cur.execute("SELECT status, count(*) FROM lucidota_control.absurd_queue_job GROUP BY status ORDER BY status")
        payload["state_absurd_queue_job"] = {k: int(v) for k, v in cur.fetchall()}
        cur.execute("SELECT status, queue_name, count(*) FROM lucidota_control.absurd_queue GROUP BY status, queue_name ORDER BY status, queue_name")
        payload["state_absurd_queue"] = [dict(status=s, queue_name=q, count=int(n)) for s, q, n in cur.fetchall()]
        cur.execute("SELECT resolved, count(*) FROM lucidota_control.absurd_queue_dead_letter GROUP BY resolved ORDER BY resolved")
        payload["state_dead_letter_resolved"] = {str(k): int(v) for k, v in cur.fetchall()}
    with psycopg.connect(storage_dsn) as conn:
        payload["storage_schemas"] = {}
        for schema in ["lucidota_archaeology", "lucidota_chatdump", "lucidota_commdump", "lucidota_etl", "lucidota_go", "lucidota_hardmath", "lucidota_indy", "lucidota_investigation", "lucidota_korpus", "lucidota_learning", "lucidota_runtime", "lucidota_semantic", "lucidota_spine", "lucidota_vault", "lucidota_workflow_foundry"]:
            payload["storage_schemas"][schema] = db_counts(conn, schema)
    return payload


def render_readthisfirst(tree: dict[str, Any], db: dict[str, Any]) -> str:
    facts = db["state_runtime_facts"]
    refresh = facts.get("runtime_facts_refresh", {}).get("absurd_queue", {})
    return f"""# LUCIDOTA READ THIS FIRST — CURRENT

Generated: `{tree['generated_at']}`

## Start here

1. Read `AGENTS.md`.
2. Read `00_PROJECT_BRAIN/TICKLETRUNK.md` and `00_PROJECT_BRAIN/ACTIVE_INSTRUCTION_INDEX.md`.
3. Read `GOALS/CURRENT_HANDOFF.md`.
4. Use `python3 scripts/dev_library_scan.py --query <topic>` before inventing anything.
5. Use receipts and runtime facts, not prose memory, as the authority layer.

## What is ledgered vs theater

- **Ledgered:** `lucidota_control.runtime_status_fact`, `05_OUTPUTS/*` receipts, `*_latest` audit docs, Postgres tables, and generated JSONL/MD receipts.
- **Theater:** stale handoffs, old prompts, uncited prose, guessed status, and any "looks green" claim without a fresh receipt.

## Ontology rule

LUCIDOTA is a highly advanced ternary/binary ontological truth machine: truth means the current status as understood, no more and no less. The active math is base 35 plus `1`, `0`, and `-1` lane logic, and the fast path stays deterministic.
- The language router stays GO-25 strict inside workflows and only flips to final-print behavior at explicit workflow-end markers.

## Slop laws in effect

- Dev Library reuse first; do not pave sovereign originals.
- Blueprint first, model second.
- Proportional proof only; don't freeze low-risk work with useless audit bloat.
- Temporal truth: current means freshly verified.
- Capability preservation: do not remove ability unless the operator explicitly asks or receipts prove it is dead.

## Orchestration lane

- This agent is orchestration only.
- Groq and the local system do the heavy work; this agent coordinates, checks, and keeps the loop cheap.
- Prefer bounded scripts and current Postgres facts over long prose.
- Keep the loop cheap: one instruction, one receipt, one verification.
- Scenario batches and worker packets run through `scripts/goal_scenario_batch.py` and `scripts/goal_agent_packet.py` using GO-25 JSON packets.
- Holdout evaluation for scenario batches runs through `scripts/goal_scenario_batch.py`'s holdout report path, which scores held-out routes, emits promoted rules, and keeps the decision tree compact.
- The scenario batch CLI accepts `--holdout-stride N` to emit the holdout evaluator receipt directly when the operator wants a compact train/holdout comparison instead of a plain training batch.
- The scenario batch CLI accepts `--compare-report <path>` to bias family ordering from the latest compare report before it generates the next batch or holdout receipt.
- Receipt-to-receipt comparison for scenario batches runs through `scripts/goal_scenario_compare.py`, which compares rule conditions/actions across reports and proposes the next GO-25 seed plus scenario focus.
- The scenario matrix now explicitly includes evidence-ingest and queue-integrity families so the batch lane tests byte-perfect evidence ingestion, embeddings, and no-loss JSON queue handoff alongside the normal/adversarial/noisy/fast-slow cases.
- `GOALS/SWARM_CURRENT_BRIEF.md` plus `scripts/goal_swarm_brief.py` turn the compare-driven loop into compact per-script worker packets for new sessions.
- `scripts/goal_swarm_brief.py --launch` turns the brief into bounded queue receipts, one packet at a time, without inventing a new queue layer.
- Worker packets now carry an explicit output contract: required fields are `status`, `result`, `next_action`, and `receipt_path`, with minimum decision-pair evidence requested from the worker.
- Exact JSON envelopes should be given breathing room: the worker packet contract now recommends a 256-token floor and 512-token ceiling for compact, parseable responses.
- Confidence-weighted symbol condensation runs through `scripts/graph_symbol_condensation.py` against graph/Postgres evidence, with Bayes-style confidence and morphing claims.
- Temporal comparison runs through `scripts/graph_symbol_compare.py` to diff condensation receipts and propose the next GO-25 seed.
- Symbol dispatch runs through `scripts/graph_symbol_dispatch.py` to fan that next seed out to Groq/local worker packets.
- Model receipt normalization and drift audit runs through `scripts/model_output_contract_audit.py`, which canonicalizes `decisions` into `decision_pairs`, parses fenced JSON text when present, and recovers scalar fields from truncated envelopes when possible.
- Durable queue handoff runs through `scripts/goal_swarm_dispatch.py` and `scripts/absurd_queue_spine.py` when the packet should actually be consumed asynchronously.

## Current batching rule

- When shuttling intent to another worker/model, batch ontology in small chunks, usually 2-7 primitives.
- Use the smallest batch that preserves meaning.
- If the receiver's answer makes the next instruction obvious, send only the next small instruction.

## Current operating snapshot

- Active DB state: `lucidota_state`
- Storage DB: `lucidota_storage`
- Active ontology: `GO-25 / OBJECT-EVENT-EDGE`
- Current queue health: succeeded `{refresh.get('succeeded')}`, failed `{refresh.get('failed')}`, queued `{refresh.get('queued')}`, total `{refresh.get('total')}`
- Baseline queue reconciliation fact: `absurd_queue_health.dead_letter_rows={facts.get('absurd_queue_health', {}).get('dead_letter_rows')}` and unresolved dead letters `0`
- Active queues: `{len(db['state_absurd_queue'])}`
- Groq wiring: `{facts.get('llxprt_groq_login_wired', {}).get('provider')} / {facts.get('llxprt_groq_login_wired', {}).get('model')}`
- Hitman loop: `{facts.get('autonomous_hitman_loop', {}).get('status')}`

## Operator commands that matter

```bash
python3 scripts/system_runtime_facts_refresh.py --execute
python3 scripts/dev_library_scan.py --query <topic>
python3 scripts/slop_audit_law.py --query <topic>
python3 scripts/goal_scenario_batch.py --scenario-count 12 --batch-size 4 --json
python3 scripts/goal_scenario_batch.py --scenario-count 18 --batch-size 6 --holdout-stride 3 --json
python3 scripts/goal_scenario_batch.py --scenario-count 18 --batch-size 6 --holdout-stride 3 --compare-report 05_OUTPUTS/goals/<compare>.json --json
python3 scripts/goal_scenario_compare.py --current 05_OUTPUTS/goals/<holdout>.json --baseline 05_OUTPUTS/goals/<batch>.json --json
python3 scripts/goal_swarm_brief.py --json
python3 scripts/goal_swarm_brief.py --launch --limit 2 --json
python3 - <<'PY'
from scripts.goal_scenario_batch import build_holdout_report, write_report
report = write_report(build_holdout_report(objective='...', target='groq|cohere|local', scenario_count=18, batch_size=6, holdout_stride=3, packet={{'schema':'lucidota.worker_order.v1','target':'local','intent':'scenario_batch'}}))
print(report['report_path'])
PY
python3 scripts/graph_symbol_condensation.py --symbols OBJECT,EVENT,EDGE --json
python3 scripts/graph_symbol_compare.py --current <current-report> --baseline <baseline-report> --json
python3 scripts/graph_symbol_dispatch.py --compare <compare-report> --lanes groq,local --json
python3 scripts/goal_swarm_dispatch.py --target local --task <task> --jobs 2 --json
pytest -q tests/test_absurd_*_contract.py
psql postgresql:///lucidota_state -Atc "SELECT fact_key,fact_value FROM lucidota_control.runtime_status_fact WHERE subsystem='system' ORDER BY fact_key;"
```

## Where the rest of the map lives

- Exhaustive Dev Library: `00_PROJECT_BRAIN/TICKLETRUNK.md` and `00_PROJECT_BRAIN/TICKLETRUNK.json`
- Tree index: `00_PROJECT_BRAIN/FILESYSTEM_TREE_INDEX_CURRENT.md`
- Postgres audit: `00_PROJECT_BRAIN/POSTGRES_AUDIT_CURRENT.md`
- Live receipts: `05_OUTPUTS/`
- Queue/goal continuity: `GOALS/CURRENT_HANDOFF.md` and `GOALS/GOAL_LOG.md`

## Fast truth anchors

- ABSURD queue jobs: `{db['state_absurd_queue_job']}`
- State schemas: `{', '.join(f"{k}={v['table_count']}" for k, v in db['state_schemas'].items())}`
- Storage schemas: `{', '.join(f"{k}={v['table_count']}" for k, v in db['storage_schemas'].items())}`

## Read this if you only read one thing

This repo runs on receipts, schema facts, and gated workers. If a claim does not have a fresh receipt or live DB fact, it is not current.
"""


def render_tree_index(tree: dict[str, Any]) -> str:
    lines = [f"# LUCIDOTA FULL SYSTEM FILESYSTEM TREE INDEX — CURRENT", "", f"Generated: `{tree['generated_at']}`", "", "## Top-level trees", ""]
    for row in tree["trees"]:
        lines.append(f"### `{row['path']}`")
        lines.append(f"- Direct subtrees: {row['child_dir_count']}")
        if row["child_dirs"]:
            lines.append("- Children:")
            for child in row["child_dirs"]:
                lines.append(f"  - `{child}`")
        else:
            lines.append("- Children: none")
        lines.append("")
    lines += [
        "## Notes",
        "",
        "- This is the human-readable tree index. Exhaustive file-level inventories stay in TICKLETRUNK and the ouroboros filesystem audit outputs.",
        "- Output trees and caches are included as first-class trees because they are part of the operating system's receipts and working state.",
    ]
    return "\n".join(lines) + "\n"


def render_postgres_audit(db: dict[str, Any]) -> str:
    state = db["state_schemas"]
    storage = db["storage_schemas"]
    facts = db["state_runtime_facts"]
    return f"""# LUCIDOTA POSTGRES AUDIT — CURRENT

Generated: `{db['generated_at']}`

## Databases

- State DB: `{db['state_dsn']}`
- Storage DB: `{db['storage_dsn']}`

## State DB schemas

| Schema | Tables | Notes |
|---|---:|---|
| lucidota_control | {state['lucidota_control']['table_count']} | queue/workflow/contracts/runtime facts |
| lucidota_learning | {state['lucidota_learning']['table_count']} | river/bytewax/learning hints |
| lucidota_runtime | {state['lucidota_runtime']['table_count']} | runtime state and receipts |
| lucidota_go | {state['lucidota_go']['table_count']} | graph promotion evidence |
| lucidota_indy | {state['lucidota_indy']['table_count']} | Indy_READs state |
| lucidota_pivot | {state['lucidota_pivot']['table_count']} | routing/selection pivots |
| lucidota_ternary | {state['lucidota_ternary']['table_count']} | ternary lab |
| lucidota_vault | {state['lucidota_vault']['table_count']} | vault/secret custody |
| lucidota_bus | {state['lucidota_bus']['table_count']} | bus/wake events |

## Storage DB schemas

| Schema | Tables | Notes |
|---|---:|---|
| lucidota_korpus | {storage['lucidota_korpus']['table_count']} | raw/derived custody, present in storage not state |
| lucidota_go | {storage['lucidota_go']['table_count']} | storage-side graph staging |
| lucidota_learning | {storage['lucidota_learning']['table_count']} | storage-side learning artifacts |
| lucidota_archaeology | {storage['lucidota_archaeology']['table_count']} | audit/archaeology receipts |
| lucidota_spine | {storage['lucidota_spine']['table_count']} | spine/runtime contracts |
| lucidota_runtime | {storage['lucidota_runtime']['table_count']} | runtime artifacts |
| lucidota_vault | {storage['lucidota_vault']['table_count']} | vault artifacts |
| lucidota_indy | {storage['lucidota_indy']['table_count']} | Indy artifacts |
| lucidota_chatdump | {storage['lucidota_chatdump']['table_count']} | chat timeline imports |
| lucidota_commdump | {storage['lucidota_commdump']['table_count']} | comm timeline imports |
| lucidota_etl | {storage['lucidota_etl']['table_count']} | ETL pipeline receipts |
| lucidota_hardmath | {storage['lucidota_hardmath']['table_count']} | hard-truth math audit |
| lucidota_investigation | {storage['lucidota_investigation']['table_count']} | evidence/investigation |
| lucidota_semantic | {storage['lucidota_semantic']['table_count']} | semantic staging |
| lucidota_workflow_foundry | {storage['lucidota_workflow_foundry']['table_count']} | workflow foundry |

## Current runtime facts

```json
{json.dumps(facts, indent=2, sort_keys=True)}
```

## Operational reading

- `lucidota_state` is the active control plane for queues, contracts, learning hints, and runtime facts.
- `lucidota_storage` holds the heavier custody and ETL/graph/workflow evidence.
- Current queue state: succeeded `{facts.get('absurd_queue_health', {}).get('succeeded')}`, dead-lettered `{facts.get('absurd_queue_health', {}).get('dead_lettered')}`, unresolved dead-letter rows `0`.
- Current graph/materialization snapshot from runtime facts: `graph_items={facts.get('runtime_facts_refresh', {}).get('graph', {}).get('graph_items')}`, `graph_edges={facts.get('runtime_facts_refresh', {}).get('graph', {}).get('graph_edges')}`, `materializations={facts.get('runtime_facts_refresh', {}).get('graph', {}).get('materializations')}`.
- Storage-side KORPUS exists in `lucidota_storage`, not `lucidota_state`. That distinction matters.

## What to audit next if the DB changes

1. Re-run `python3 scripts/system_runtime_facts_refresh.py --execute`
2. Re-run this doc refresh script
3. Re-check queue counts, schema counts, and unresolved dead-letter rows
4. Keep receipts in `05_OUTPUTS/system_docs/`
"""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate current readthisfirst, filesystem tree, and postgres audit docs.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    tree = tree_index()
    db = postgres_audit()
    readme = PB / "READTHISFIRST_CURRENT.md"
    tree_doc = PB / "FILESYSTEM_TREE_INDEX_CURRENT.md"
    pg_doc = PB / "POSTGRES_AUDIT_CURRENT.md"
    write(readme, render_readthisfirst(tree, db))
    write(tree_doc, render_tree_index(tree))
    write(pg_doc, render_postgres_audit(db))
    receipt = {
        "schema": "lucidota.system_docs.refresh.v1",
        "generated_at": now(),
        "readthisfirst": rel(readme),
        "filesystem_tree_index": rel(tree_doc),
        "postgres_audit": rel(pg_doc),
        "status": "PASS",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    rp = OUT / f"system_docs_refresh_{stamp()}.json"
    receipt["report_path"] = rel(rp)
    rp.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    latest = OUT / "system_docs_refresh_latest.json"
    latest.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print("REPORT_PATH=" + rel(rp))
    print("SYSTEM_DOCS_REFRESH=PASS")
    if args.json:
        print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
