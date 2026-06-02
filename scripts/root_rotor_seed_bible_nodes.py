#!/usr/bin/env python3
"""Seed DB-coordinate canon nodes from a Root-Rotor audit manifest.

This does not perform the final model analysis. It reserves one draft coordinate per
active file and marks it as requiring a dedicated model call.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "GOALS" / "ROOT_ROTOR_ACTIVE_SOFTWARE_AUDIT_DUMP.manifest.json"

MANUAL_ROOTS = {
    "SYSTEM_ARCH": (1, "System Architecture Manual", "Relational schema, structural state, and map coordinates."),
    "RUNTIME_GOVERNOR": (2, "Runtime Governor Manual", "Resource control, process execution, and hardware admission rules."),
    "AVIONICS": (3, "Algorithms Avionics Manual", "Algorithmic primitives, mathematical routers, and endpoint logic."),
    "FLIGHT_MAN": (4, "Operations Flight Manual", "CLI tools, workflows, ingestion paths, and operator procedures."),
    "LEDGER": (5, "Ledger Amendment Manual", "Version history, receipts, handoffs, and authority records."),
}

VALID_NODE_KINDS = {
    "MANUAL_SECTION", "OBJECT", "WORKFLOW", "EVENT", "RECEIPT", "EDGE", "STATE", "BOX",
    "CLAIM", "SOURCE", "LEDGER", "SCHEMA", "CONFIG", "SCRIPT", "ALGORITHM", "MODEL",
    "DAEMON", "TEST", "REFERENCE",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def manual_for_path(path: str) -> str:
    p = path.lower()
    if p.startswith("06_schema/") or p.endswith(".sql"):
        return "SYSTEM_ARCH"
    if "governor" in p or "bge" in p or "model_runtime" in p or "slice" in p or p.startswith("build_automation/"):
        return "RUNTIME_GOVERNOR"
    if p.startswith("algos/") or p.startswith("pypeline/") or p.startswith("math/") or "m1_s2" in p or "router" in p:
        return "AVIONICS"
    if p.startswith("goals/") or p.startswith("00_project_brain/") or p in {"agents.md", "claude.md", "llxprt.md", "skills-lock.json"}:
        return "LEDGER"
    return "FLIGHT_MAN"


def classify_node_kind_tags(path: str, manual_id: str | None = None) -> tuple[str, list[str]]:
    p = path.lower()
    if p.startswith("06_schema/") or p.endswith(".sql"):
        return "SCHEMA", ["OBJECT", "STATE", "CHURN"]
    if p.startswith("algos/") or p.startswith("math/") or "router" in p:
        return "ALGORITHM", ["OBJECT", "WORKFLOW"]
    if p.startswith("tests/") or "/test_" in p or p.startswith("test_"):
        return "TEST", ["WORKFLOW", "RECEIPT"]
    if p.startswith("goals/") or p.startswith("00_project_brain/") or p in {"agents.md", "claude.md", "llxprt.md"}:
        return "LEDGER", ["LEDGER", "RECEIPT", "STATE"]
    if "daemon" in p or "postgrest" in p or p.endswith(".service") or "watch" in p:
        return "DAEMON", ["DAEMON", "WORKFLOW", "STATE"]
    if p.endswith((".json", ".toml", ".yaml", ".yml", ".ini")):
        return "CONFIG", ["OBJECT", "STATE"]
    if p.startswith("scripts/") or p.endswith((".py", ".sh", ".rs", ".ts", ".tsx", ".js")):
        return "WORKFLOW", ["WORKFLOW", "OBJECT", "RECEIPT"]
    return "OBJECT", ["OBJECT"]


def root_nodes() -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for manual_id, (volume, title, payload) in MANUAL_ROOTS.items():
        nodes.append({
            "node_id": f"{volume}.0.0",
            "parent_id": None,
            "manual_id": manual_id,
            "node_kind": "MANUAL_SECTION",
            "title": title,
            "payload": payload,
            "payload_format": "text",
            "ontology_tags": ["LEDGER", "STATE"] if manual_id == "LEDGER" else ["OBJECT", "STATE"],
            "source_refs": [],
            "evidence_hashes": [],
            "dependencies": [],
            "affects_nodes": [],
            "status": "draft",
        })
    return nodes


def payload_for_file(entry: dict[str, Any], manual_id: str, node_kind: str, ontology_tags: list[str]) -> str:
    return json.dumps({
        "schema": "lucidota.root_rotor.file_seed_payload.v1",
        "state": "pending_dedicated_model_analysis",
        "source_path": entry["path"],
        "source_sha256": entry["sha256"],
        "node_kind": node_kind,
        "ontology_tags": ontology_tags,
        "size_bytes": entry.get("size_bytes"),
        "bytes_read": entry.get("bytes_read"),
        "truncated": bool(entry.get("truncated")),
        "target_manual_id": manual_id,
        "required_model_call": True,
        "target_output_contract": "lucidota.root_rotor.bible_node_payload.v1",
    }, sort_keys=True)


def build_seed_nodes(manifest_path: Path = DEFAULT_MANIFEST) -> list[dict[str, Any]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    nodes = root_nodes()
    counters = {manual_id: 0 for manual_id in MANUAL_ROOTS}
    for entry in manifest.get("files", []):
        path = str(entry["path"])
        manual_id = manual_for_path(path)
        node_kind, ontology_tags = classify_node_kind_tags(path, manual_id)
        counters[manual_id] += 1
        volume = MANUAL_ROOTS[manual_id][0]
        nodes.append({
            "node_id": f"{volume}.{counters[manual_id]}.0",
            "parent_id": f"{volume}.0.0",
            "manual_id": manual_id,
            "node_kind": node_kind,
            "title": path,
            "payload": payload_for_file(entry, manual_id, node_kind, ontology_tags),
            "payload_format": "json",
            "ontology_tags": ontology_tags,
            "source_refs": [path],
            "evidence_hashes": [entry["sha256"]],
            "dependencies": [],
            "affects_nodes": [],
            "status": "draft",
        })
    return nodes


def current_manifest_sources(nodes: list[dict[str, Any]]) -> set[str]:
    sources: set[str] = set()
    for node in nodes:
        refs = node.get("source_refs") or []
        if refs:
            sources.add(str(refs[0]))
    return sources


def retire_stale_sql() -> str:
    return """
    UPDATE lucidota_canon.bible_nodes
    SET status = 'deprecated',
        valid_to = now(),
        updated_at = now()
    WHERE source_refs <> '[]'::jsonb
      AND source_refs->>0 <> ALL(%(current_sources)s)
      AND status <> 'deprecated'
      AND payload_format = 'json'
      AND (
          payload::jsonb->>'schema' = 'lucidota.root_rotor.file_seed_payload.v1'
          OR payload::jsonb->>'schema' = 'lucidota.root_rotor.bible_node_payload.v1'
      );
    """


def retire_duplicate_sources_sql() -> str:
    return """
    WITH ranked AS (
        SELECT
            node_id,
            row_number() OVER (
                PARTITION BY source_refs->>0
                ORDER BY
                    CASE status WHEN 'verified' THEN 0 WHEN 'review_required' THEN 1 WHEN 'draft' THEN 2 ELSE 3 END,
                    node_sort_key ASC,
                    version DESC
            ) AS keep_rank
        FROM lucidota_canon.bible_nodes
        WHERE source_refs <> '[]'::jsonb
          AND status <> 'deprecated'
          AND payload_format = 'json'
          AND (
              payload::jsonb->>'schema' = 'lucidota.root_rotor.file_seed_payload.v1'
              OR payload::jsonb->>'schema' = 'lucidota.root_rotor.bible_node_payload.v1'
          )
    )
    UPDATE lucidota_canon.bible_nodes node
    SET status = 'deprecated',
        valid_to = now(),
        updated_at = now()
    FROM ranked
    WHERE node.node_id = ranked.node_id
      AND ranked.keep_rank > 1;
    """


def upsert_seed_sql() -> str:
    return """
    INSERT INTO lucidota_canon.bible_nodes(
        node_id, parent_id, node_sort_key, manual_id, node_kind, title, payload, payload_format,
        ontology_tags, source_refs, evidence_hashes, dependencies, affects_nodes, status, hash_current
    ) VALUES (
        %(node_id)s, %(parent_id)s, lucidota_canon.fn_bible_node_sort_key(%(node_id)s), %(manual_id)s,
        %(node_kind)s, %(title)s, %(payload)s, %(payload_format)s, %(ontology_tags)s, %(source_refs)s, %(evidence_hashes)s,
        %(dependencies)s, %(affects_nodes)s, %(status)s, repeat('0', 64)
    )
    ON CONFLICT (node_id) DO UPDATE SET
        parent_id = EXCLUDED.parent_id,
        manual_id = EXCLUDED.manual_id,
        node_kind = EXCLUDED.node_kind,
        title = EXCLUDED.title,
        payload = EXCLUDED.payload,
        payload_format = EXCLUDED.payload_format,
        ontology_tags = EXCLUDED.ontology_tags,
        source_refs = EXCLUDED.source_refs,
        evidence_hashes = EXCLUDED.evidence_hashes,
        dependencies = EXCLUDED.dependencies,
        affects_nodes = EXCLUDED.affects_nodes,
        status = EXCLUDED.status
    WHERE lucidota_canon.bible_nodes.status IN ('draft', 'deprecated')
      AND (
          (
              lucidota_canon.bible_nodes.payload_format = 'json'
              AND lucidota_canon.bible_nodes.payload::jsonb->>'schema' = 'lucidota.root_rotor.file_seed_payload.v1'
          )
          OR lucidota_canon.bible_nodes.status = 'deprecated'
      );
    """


def upsert_seed_nodes(nodes: list[dict[str, Any]], *, dsn: str, retire_stale: bool = False) -> dict[str, Any]:
    sql = upsert_seed_sql()
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for node in nodes:
                cur.execute(sql, {
                    **node,
                    "source_refs": Jsonb(node["source_refs"]),
                    "evidence_hashes": Jsonb(node["evidence_hashes"]),
                })
            retired = 0
            if retire_stale:
                cur.execute(retire_stale_sql(), {"current_sources": sorted(current_manifest_sources(nodes))})
                retired = int(cur.rowcount or 0)
                cur.execute(retire_duplicate_sources_sql())
                retired += int(cur.rowcount or 0)
        conn.commit()
    return {
        "schema": "lucidota.root_rotor.seed_bible_nodes.v1",
        "generated_at": now(),
        "dsn": dsn,
        "nodes_upserted": len(nodes),
        "stale_nodes_deprecated": retired,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed Root-Rotor bible nodes from an audit manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dsn", default="postgresql:///lucidota_state")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--retire-stale", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    nodes = build_seed_nodes(Path(args.manifest))
    if args.execute:
        result = upsert_seed_nodes(nodes, dsn=args.dsn, retire_stale=args.retire_stale)
    else:
        result = {
            "schema": "lucidota.root_rotor.seed_bible_nodes.v1",
            "generated_at": now(),
            "manifest": args.manifest,
            "nodes_planned": len(nodes),
            "status": "DRY_RUN",
        }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=False))
    else:
        print(f"ROOT_ROTOR_SEED={result['status']}")
        print(f"NODES={result.get('nodes_upserted', result.get('nodes_planned', 0))}")
    return 0 if result["status"] in {"PASS", "DRY_RUN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
