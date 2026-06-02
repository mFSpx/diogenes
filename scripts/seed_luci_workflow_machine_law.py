#!/usr/bin/env python3
"""Seed the LUCI workflow-machine doctrine into the DB-coordinate manual.

This script does not invent canon. It takes the operator directive saved in
GOALS/ROOT_ROTOR_WORKFLOW_MACHINE_LLM_DOCTRINE_VERBATIM.md and maps it to one
versioned bible node plus the workflow registry rows enforced by schema 145.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "GOALS" / "ROOT_ROTOR_WORKFLOW_MACHINE_LLM_DOCTRINE_VERBATIM.md"
RECEIPT_DIR = ROOT / "05_OUTPUTS" / "root_rotor_manuals"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_text() -> str:
    return SOURCE.read_text(encoding="utf-8")


def doctrine_payload() -> str:
    return (
        "LUCI is a typed workflow machine over objects, boxes, events, edges, "
        "receipts, state, ledger, graph, and Postgres/Absurd runtime queues. "
        "Objects are nouns. Workflows are verbs. Boxes are addresses and policies. "
        "Events are changes. Receipts are proof. Edges are relationships. State is "
        "the latest reducible view. Ledger is history. The default execution path "
        "is deterministic. If SQL, parser, hash, regex, schema validation, graph "
        "traversal, deterministic classifier, Treelite, XGBoost, River, system "
        "telemetry, adapter logic, queue state, or existing workflow state can "
        "answer, the workflow must not call an LLM. An LLM is a boxed typed "
        "judgment adapter. It cannot mutate canon directly. Its output is a claim "
        "or proposal until a validator workflow emits a receipt. Canon changes only "
        "through validated events."
    )


def bible_node() -> dict[str, Any]:
    text = source_text()
    source_hash = sha256_text(text)
    return {
        "node_id": "5.900.0",
        "parent_id": "5.0.0",
        "manual_id": "LEDGER",
        "title": "LUCI Workflow Machine Deterministic-First LLM Doctrine",
        "payload": doctrine_payload(),
        "payload_format": "text",
        "source_refs": [str(SOURCE.relative_to(ROOT))],
        "evidence_hashes": [source_hash],
        "dependencies": ["1.0.0", "4.0.0", "5.0.0"],
        "affects_nodes": ["1.0.0", "2.0.0", "3.0.0", "4.0.0"],
        "status": "verified",
    }


def workflow_registry_rows() -> list[dict[str, Any]]:
    return [
        {
            "workflow_name": "root-rotor-apply-node-payloads",
            "workflow_id": "root-rotor-apply-node-payloads",
            "owner": "root-rotor+canon",
            "phase": "145",
            "status": "active",
            "command": "scripts/root_rotor_apply_node_payloads.py --execute",
            "inputs": {"payload": "lucidota.root_rotor.bible_node_payload.v1"},
            "outputs": {"bible_node": "versioned_row", "receipt": "json"},
            "notes": "Deterministic validator and DB promotion step for model-proposed manual nodes.",
            "verb": "apply_node_payloads",
            "input_object_types": ["bible_node_payload", "model_output_file"],
            "output_object_types": ["bible_node", "workflow_receipt"],
            "deterministic_first": True,
            "llm_allowed": False,
            "llm_required": False,
            "allowed_models": [],
            "validator_workflow_id": None,
            "receipt_type": "workflow_receipt",
            "promotion_policy": "deterministic_receipt",
            "llm_allowed_reasons": [],
            "ontology_tags": ["WORKFLOW", "RECEIPT", "STATE"],
        },
        {
            "workflow_name": "root-rotor-red-team-audit",
            "workflow_id": "root-rotor-red-team-audit",
            "owner": "root-rotor+canon",
            "phase": "145",
            "status": "active",
            "command": "scripts/root_rotor_red_team_audit.py --json",
            "inputs": {"bible_nodes": "rows", "bible_dependencies": "rows", "postgrest": "endpoint"},
            "outputs": {"audit_verdict": "json", "receipt": "json"},
            "notes": "Deterministic adversarial audit for draft nodes, broken parents, cycles, and API availability.",
            "verb": "red_team_audit",
            "input_object_types": ["bible_node", "bible_dependency", "api_endpoint"],
            "output_object_types": ["audit_receipt", "review_required_flag"],
            "deterministic_first": True,
            "llm_allowed": False,
            "llm_required": False,
            "allowed_models": [],
            "validator_workflow_id": None,
            "receipt_type": "audit_receipt",
            "promotion_policy": "deterministic_receipt",
            "llm_allowed_reasons": [],
            "ontology_tags": ["WORKFLOW", "RECEIPT", "EDGE", "STATE"],
        },
        {
            "workflow_name": "root-rotor-canon-forge",
            "workflow_id": "root-rotor-canon-forge",
            "owner": "root-rotor+canon",
            "phase": "145",
            "status": "active",
            "command": "scripts/root_rotor_manual_queue.py -> scripts/vibe_sequencer.py -> scripts/root_rotor_apply_node_payloads.py",
            "inputs": {"audit_manifest": "json", "source_file": "bounded_text"},
            "outputs": {"bible_node_payload": "json", "model_invocation_receipt": "json"},
            "notes": "Deterministic queue first; boxed model call only for per-file code/design review and technical manual transformation. Model output is a proposal until validator receipt.",
            "verb": "forge_canon_nodes",
            "input_object_types": ["source_file", "audit_manifest_entry"],
            "output_object_types": ["bible_node_payload", "model_invocation_receipt"],
            "deterministic_first": True,
            "llm_allowed": True,
            "llm_required": True,
            "allowed_models": ["vibes:codestral", "groq", "gpt-5.3-codex-spark", "gpt-5.4-mini", "gpt-5.5"],
            "validator_workflow_id": "root-rotor-apply-node-payloads",
            "receipt_type": "model_invocation_receipt",
            "promotion_policy": "proposal_until_validator_receipt",
            "llm_allowed_reasons": [
                "code_design_review",
                "natural_language_transformation",
                "human_facing_synthesis",
            ],
            "ontology_tags": ["WORKFLOW", "RECEIPT", "CLAIM", "STATE"],
        },
    ]


def apply_seed(*, dsn: str) -> dict[str, Any]:
    wf_sql = """
    INSERT INTO lucidota_control.workflow_registry(
        workflow_name, workflow_id, owner, phase, status, command, inputs, outputs, notes,
        verb, input_object_types, output_object_types, deterministic_first, llm_allowed, llm_required,
        allowed_models, validator_workflow_id, receipt_type, promotion_policy, llm_allowed_reasons, ontology_tags
    ) VALUES (
        %(workflow_name)s, %(workflow_id)s, %(owner)s, %(phase)s, %(status)s, %(command)s,
        %(inputs)s, %(outputs)s, %(notes)s, %(verb)s, %(input_object_types)s, %(output_object_types)s,
        %(deterministic_first)s, %(llm_allowed)s, %(llm_required)s, %(allowed_models)s,
        %(validator_workflow_id)s, %(receipt_type)s, %(promotion_policy)s, %(llm_allowed_reasons)s,
        %(ontology_tags)s
    )
    ON CONFLICT (workflow_name) DO UPDATE SET
        workflow_id=EXCLUDED.workflow_id,
        owner=EXCLUDED.owner,
        phase=EXCLUDED.phase,
        status=EXCLUDED.status,
        command=EXCLUDED.command,
        inputs=EXCLUDED.inputs,
        outputs=EXCLUDED.outputs,
        notes=EXCLUDED.notes,
        verb=EXCLUDED.verb,
        input_object_types=EXCLUDED.input_object_types,
        output_object_types=EXCLUDED.output_object_types,
        deterministic_first=EXCLUDED.deterministic_first,
        llm_allowed=EXCLUDED.llm_allowed,
        llm_required=EXCLUDED.llm_required,
        allowed_models=EXCLUDED.allowed_models,
        validator_workflow_id=EXCLUDED.validator_workflow_id,
        receipt_type=EXCLUDED.receipt_type,
        promotion_policy=EXCLUDED.promotion_policy,
        llm_allowed_reasons=EXCLUDED.llm_allowed_reasons,
        ontology_tags=EXCLUDED.ontology_tags,
        updated_at=now();
    """
    node_sql = """
    INSERT INTO lucidota_canon.bible_nodes(
        node_id, parent_id, node_sort_key, manual_id, title, payload, payload_format,
        source_refs, evidence_hashes, dependencies, affects_nodes, status, hash_current
    ) VALUES (
        %(node_id)s, %(parent_id)s, lucidota_canon.fn_bible_node_sort_key(%(node_id)s),
        %(manual_id)s, %(title)s, %(payload)s, %(payload_format)s, %(source_refs)s,
        %(evidence_hashes)s, %(dependencies)s, %(affects_nodes)s, %(status)s, repeat('0', 64)
    )
    ON CONFLICT (node_id) DO UPDATE SET
        parent_id=EXCLUDED.parent_id,
        manual_id=EXCLUDED.manual_id,
        title=EXCLUDED.title,
        payload=EXCLUDED.payload,
        payload_format=EXCLUDED.payload_format,
        source_refs=EXCLUDED.source_refs,
        evidence_hashes=EXCLUDED.evidence_hashes,
        dependencies=EXCLUDED.dependencies,
        affects_nodes=EXCLUDED.affects_nodes,
        status=EXCLUDED.status;
    """
    rows = workflow_registry_rows()
    node = bible_node()
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(wf_sql, {**row, "inputs": Jsonb(row["inputs"]), "outputs": Jsonb(row["outputs"])})
            cur.execute(node_sql, {**node, "source_refs": Jsonb(node["source_refs"]), "evidence_hashes": Jsonb(node["evidence_hashes"])})
        conn.commit()

    return {
        "schema": "lucidota.root_rotor.workflow_machine_law_seed.v1",
        "generated_at": now(),
        "status": "PASS",
        "dsn": dsn,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": bible_node()["evidence_hashes"][0],
        "bible_node_id": bible_node()["node_id"],
        "workflow_rows": [row["workflow_id"] for row in rows],
    }


def write_receipt(result: dict[str, Any], *, receipt_dir: Path = RECEIPT_DIR) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = receipt_dir / f"workflow_machine_law_seed_{stamp}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed LUCI workflow machine doctrine into canon DB.")
    parser.add_argument("--dsn", default="postgresql:///lucidota_state")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.execute:
        result = apply_seed(dsn=args.dsn)
        result["receipt_path"] = str(write_receipt(result).relative_to(ROOT))
    else:
        result = {
            "schema": "lucidota.root_rotor.workflow_machine_law_seed.v1",
            "generated_at": now(),
            "status": "DRY_RUN",
            "source": str(SOURCE.relative_to(ROOT)),
            "source_sha256": bible_node()["evidence_hashes"][0],
            "bible_node": bible_node(),
            "workflow_rows": workflow_registry_rows(),
        }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"WORKFLOW_MACHINE_LAW_SEED={result['status']}")
        print(f"SOURCE={result['source']}")
        print(f"NODE={result.get('bible_node_id', result.get('bible_node', {}).get('node_id'))}")
    return 0 if result["status"] in {"PASS", "DRY_RUN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
