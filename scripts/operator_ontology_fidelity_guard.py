#!/usr/bin/env python3
"""Runtime Operator ontology fidelity guard for extraction outputs.

Checks actual extraction/report outputs for forbidden softening or renamed labels
before they can be used by archaeology, workflow, or graph promotion paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "05_OUTPUTS/contracts/operator_ontology_labels.json"
SCHEMA = ROOT / "06_SCHEMA/078_operator_ontology_fidelity_runtime.sql"
OUT = ROOT / "05_OUTPUTS/contracts"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def db_url(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    try:
        if path.suffix.lower() == ".json":
            text = json.dumps(json.loads(raw.decode("utf-8", errors="replace")), sort_keys=True, ensure_ascii=False)
        else:
            text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = raw.decode("utf-8", errors="replace")
    return sha_bytes(raw), text


def load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def evaluate(path: Path) -> dict[str, Any]:
    contract = load_contract()
    input_sha, text = read_text(path)
    labels = [str(x) for x in contract.get("required_exact_labels", [])]
    seen = [label for label in labels if label in text]
    missing = [label for label in labels if label not in text]
    hits = []
    for canonical, substitutes in contract.get("forbidden_softened_substitutes", {}).items():
        canonical_present = canonical in text
        for substitute in substitutes:
            if substitute in text and not canonical_present:
                hits.append({"canonical": canonical, "softened_substitute": substitute})
    passed = not hits
    return {
        "input_path": rel(path),
        "input_sha256": input_sha,
        "status": "PASS" if passed else "FAIL",
        "required_labels_seen": seen,
        "required_labels_missing": missing,
        "forbidden_softening_hits": hits,
        "graph_promotion_allowed": False,
    }


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"operator_ontology_fidelity_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(db_url(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA.read_text(encoding="utf-8"))
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {"action": "init_schema", "execute_performed": bool(args.execute), "schema": rel(SCHEMA)})
    return 0


def check(args: argparse.Namespace) -> int:
    path = Path(args.input).expanduser().resolve()
    blockers: list[str] = []
    if not path.exists() or not path.is_file():
        blockers.append("input_file_missing")
        result = {"input_path": rel(path), "input_sha256": "0" * 64, "status": "FAIL", "required_labels_seen": [], "required_labels_missing": [], "forbidden_softening_hits": [], "graph_promotion_allowed": False}
    else:
        result = evaluate(path)
    if result["forbidden_softening_hits"]:
        blockers.append("forbidden_softening_detected")
    report = {"action": "check", "execute_performed": bool(args.execute), "db_writes_performed": False, "graph_writes_performed": False, "result": result, "blockers": blockers}
    if args.execute:
        with psycopg.connect(db_url(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA.read_text(encoding="utf-8"))
                cur.execute(
                    """
                    INSERT INTO lucidota_archaeology.ontology_fidelity_runtime_check(
                      input_path, input_sha256, status, required_labels_seen, required_labels_missing,
                      forbidden_softening_hits, graph_promotion_allowed, detail
                    ) VALUES (%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,false,%s::jsonb)
                    RETURNING check_uuid::text
                    """,
                    (
                        result["input_path"], result["input_sha256"], result["status"],
                        json.dumps(result["required_labels_seen"]), json.dumps(result["required_labels_missing"]),
                        json.dumps(result["forbidden_softening_hits"]), json.dumps({"script": "scripts/operator_ontology_fidelity_guard.py"}),
                    ),
                )
                report["check_uuid"] = cur.fetchone()[0]
            conn.commit()
        report["db_writes_performed"] = True
    write_report("check_execute" if args.execute else "check_dry_run", report)
    print(f"ONTOLOGY_FIDELITY={result['status']}")
    return 0 if result["status"] == "PASS" and not blockers else 4


def main() -> int:
    p = argparse.ArgumentParser(description="Runtime Operator ontology fidelity guard")
    p.add_argument("--database-url")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("init-schema"); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=init_schema)
    sp = sub.add_parser("check"); sp.add_argument("--input", required=True); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=check)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
