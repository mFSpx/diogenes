#!/usr/bin/env python3
"""Apply Root-Rotor model node payloads to DB canon nodes."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NODE_DIR = ROOT / "05_OUTPUTS" / "root_rotor_nodes"
TARGET_SCHEMA = "lucidota.root_rotor.bible_node_payload.v1"
LAW_OF_ROOT_NODE_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")
VALID_NODE_KINDS = {
    "MANUAL_SECTION", "OBJECT", "WORKFLOW", "EVENT", "RECEIPT", "EDGE", "STATE", "BOX",
    "CLAIM", "SOURCE", "LEDGER", "SCHEMA", "CONFIG", "SCRIPT", "ALGORITHM", "MODEL",
    "DAEMON", "TEST", "REFERENCE",
}


def normalize_node_kind(value: Any, source_path: str = "") -> str:
    kind = str(value or "").upper()
    if kind in VALID_NODE_KINDS:
        return kind
    p = source_path.lower()
    if p.startswith("06_schema/") or p.endswith(".sql"):
        return "SCHEMA"
    if p.startswith("algos/") or "router" in p:
        return "ALGORITHM"
    if p.startswith("tests/") or "/test_" in p:
        return "TEST"
    if p.endswith((".json", ".toml", ".yaml", ".yml", ".ini")):
        return "CONFIG"
    if p.startswith("scripts/") or p.endswith((".py", ".sh", ".rs", ".ts", ".tsx", ".js")):
        return "WORKFLOW"
    return "OBJECT"


def normalize_ontology_tags(values: Any, node_kind: str) -> list[str]:
    tags = [str(x).upper() for x in (values or []) if str(x).strip()]
    if not tags:
        tags = ["OBJECT"]
    if node_kind in {"WORKFLOW", "SCHEMA", "ALGORITHM", "DAEMON", "TEST"} and "OBJECT" not in tags:
        tags.append("OBJECT")
    if node_kind == "WORKFLOW" and "WORKFLOW" not in tags:
        tags.append("WORKFLOW")
    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags:
        if tag not in seen:
            ordered.append(tag)
            seen.add(tag)
    return ordered


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_node_payloads(node_dir: Path = DEFAULT_NODE_DIR) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if not node_dir.exists():
        return payloads
    for path in sorted(node_dir.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if obj.get("schema") != TARGET_SCHEMA:
            continue
        if not obj.get("source_path") or not obj.get("source_sha256"):
            continue
        obj["_payload_path"] = str(path)
        payloads.append(obj)
    return payloads


def to_update_record(payload: dict[str, Any]) -> dict[str, Any]:
    source_path = str(payload["source_path"])
    title = str(payload.get("node_title") or source_path)
    node_kind = normalize_node_kind(payload.get("node_kind"), source_path)
    ontology_tags = normalize_ontology_tags(payload.get("ontology_tags"), node_kind)
    dependencies = [str(x) for x in payload.get("dependencies") or [] if LAW_OF_ROOT_NODE_RE.match(str(x))]
    affects_nodes = [str(x) for x in payload.get("affects_nodes") or [] if LAW_OF_ROOT_NODE_RE.match(str(x))]
    evidence_refs = payload.get("evidence_refs") or []
    source_refs = sorted(set([source_path] + [str(x) for x in evidence_refs]))
    return {
        "source_path": source_path,
        "title": title,
        "node_kind": node_kind,
        "payload": json.dumps(payload, sort_keys=True),
        "payload_format": "json",
        "ontology_tags": ontology_tags,
        "source_refs": source_refs,
        "evidence_hashes": [str(payload["source_sha256"])],
        "dependencies": dependencies,
        "affects_nodes": affects_nodes,
        "status": "verified",
    }


def apply_payloads(payloads: list[dict[str, Any]], *, dsn: str) -> dict[str, Any]:
    updates = [to_update_record(p) for p in payloads]
    sql = """
        UPDATE lucidota_canon.bible_nodes
        SET title = %(title)s,
            payload = %(payload)s,
            payload_format = %(payload_format)s,
            node_kind = %(node_kind)s,
            ontology_tags = %(ontology_tags)s,
            source_refs = %(source_refs)s,
            evidence_hashes = %(evidence_hashes)s,
            dependencies = %(dependencies)s,
            affects_nodes = %(affects_nodes)s,
            status = %(status)s
        WHERE source_refs @> %(source_path_json)s::jsonb
    """
    applied = 0
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for update in updates:
                cur.execute(sql, {
                    **update,
                    "source_refs": Jsonb(update["source_refs"]),
                    "evidence_hashes": Jsonb(update["evidence_hashes"]),
                    "source_path_json": json.dumps([update["source_path"]]),
                })
                applied += int(cur.rowcount or 0)
        conn.commit()
    return {
        "schema": "lucidota.root_rotor.apply_node_payloads.v1",
        "generated_at": now(),
        "payloads_seen": len(payloads),
        "nodes_updated": applied,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Root-Rotor model node payloads to bible_nodes.")
    parser.add_argument("--node-dir", default=str(DEFAULT_NODE_DIR))
    parser.add_argument("--dsn", default="postgresql:///lucidota_state")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payloads = load_node_payloads(Path(args.node_dir))
    result = apply_payloads(payloads, dsn=args.dsn) if args.execute else {
        "schema": "lucidota.root_rotor.apply_node_payloads.v1",
        "generated_at": now(),
        "payloads_seen": len(payloads),
        "nodes_planned": len(payloads),
        "status": "DRY_RUN",
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=False))
    else:
        print(f"ROOT_ROTOR_APPLY={result['status']}")
        print(f"PAYLOADS={result.get('payloads_seen', 0)} UPDATED={result.get('nodes_updated', 0)}")
    return 0 if result["status"] in {"PASS", "DRY_RUN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
