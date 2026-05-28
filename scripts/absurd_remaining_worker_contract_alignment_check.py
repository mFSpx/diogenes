#!/usr/bin/env python3
"""Static contract-alignment check for remaining ABSURD workers.

This covers the current open blocker scope: Chrono, Surface/CEP, document_parse,
and KORPUS. It proves source-visible contract validation/rejection hooks and SQL
registry job-kind alignment without touching the live database.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from spine_common import ROOT, receipt, rel  # noqa: E402

SCHEMA = "lucidota.absurd_remaining_worker_contract_alignment.v1"
OUT_ROOT = "05_OUTPUTS/absurd"
CONTRACT_SQL = ROOT / "06_SCHEMA" / "082_absurd_worker_contract_registry_enforcement.sql"


@dataclass(frozen=True)
class WorkerSpec:
    worker_key: str
    queue_name: str
    script_path: str


WORKERS = [
    WorkerSpec("chrono_worker", "chrono", "scripts/absurd_chrono_worker.py"),
    WorkerSpec("surface_cep_worker", "surface_cep", "scripts/spine_surface_cep_worker.py"),
    WorkerSpec("document_parse_worker", "document_parse", "scripts/spine_document_parse_worker.py"),
    WorkerSpec("krampus_worker", "korpus", "scripts/spine_krampus_worker.py"),
]


def source_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8", errors="ignore")


def literal_strings(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        values: list[str] = []
        for elt in node.elts:
            values.extend(literal_strings(elt))
        return values
    return []


def source_job_kinds(path: str) -> list[str]:
    tree = ast.parse(source_text(path))
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & {"JOB_KIND", "JOB_KINDS", "KIND", "KINDS"}:
                values.update(literal_strings(node.value))
    return sorted(values)


def schema_job_kinds(worker_key: str) -> list[str]:
    text = CONTRACT_SQL.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"\('%s'\s*,\s*'[^']+'\s*,\s*'[^']+'\s*,\s*'\{\"job_kind\":\"([^\"]+)\"" % re.escape(worker_key), re.S)
    match = pattern.search(text)
    if not match:
        return []
    return sorted(x.strip() for x in match.group(1).split("|") if x.strip())


def worker_once_source(text: str) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        marker = text.find("def worker_once")
        return text[marker:] if marker != -1 else text
    lines = text.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "worker_once":
            start = max(0, node.lineno - 1)
            end = getattr(node, "end_lineno", None) or len(lines)
            return "\n".join(lines[start:end])
    return text


def checked_before_running(text: str) -> bool:
    worker_text = worker_once_source(text)
    validation = worker_text.find("validate_worker_contract")
    rejection = worker_text.find("record_worker_contract_rejection")
    running = worker_text.find("status='running'")
    if running == -1:
        running = worker_text.find('status="running"')
    return validation != -1 and rejection != -1 and running != -1 and validation < running and rejection < running


def audit_worker(spec: WorkerSpec) -> dict[str, Any]:
    path = ROOT / spec.script_path
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    src_kinds = source_job_kinds(spec.script_path) if path.exists() else []
    sql_kinds = schema_job_kinds(spec.worker_key)
    return {
        "worker_key": spec.worker_key,
        "queue_name": spec.queue_name,
        "script_path": spec.script_path,
        "script_exists": path.exists(),
        "schema_job_kinds": sql_kinds,
        "source_job_kinds": src_kinds,
        "schema_matches_source_job_kinds": sql_kinds == src_kinds,
        "uses_validate_worker_contract": "validate_worker_contract" in text,
        "uses_record_worker_contract_rejection": "record_worker_contract_rejection" in text,
        "contract_checked_before_running_status": checked_before_running(text),
    }


def build_report() -> dict[str, Any]:
    rows = [audit_worker(spec) for spec in WORKERS]
    errors: list[str] = []
    for row in rows:
        key = row["worker_key"]
        if not row["script_exists"]:
            errors.append(f"worker_script_missing:{key}:{row['script_path']}")
        if not row["schema_job_kinds"]:
            errors.append(f"schema_contract_missing:{key}")
        if not row["schema_matches_source_job_kinds"]:
            errors.append(f"job_kind_mismatch:{key}:schema={row['schema_job_kinds']}:source={row['source_job_kinds']}")
        if not row["uses_validate_worker_contract"]:
            errors.append(f"missing_validate_worker_contract:{key}")
        if not row["uses_record_worker_contract_rejection"]:
            errors.append(f"missing_record_worker_contract_rejection:{key}")
        if not row["contract_checked_before_running_status"]:
            errors.append(f"contract_not_checked_before_running_status:{key}")
    return {
        "schema": SCHEMA,
        "passed": not errors,
        "worker_count": len(rows),
        "contract_sql": rel(CONTRACT_SQL),
        "workers": rows,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check remaining ABSURD worker contract alignment")
    parser.add_argument("--json", action="store_true", help="write JSON receipt")
    args = parser.parse_args()
    report = build_report()
    if args.json:
        receipt("absurd_remaining_worker_contract_alignment", report, root=OUT_ROOT)
    print("ABSURD_REMAINING_WORKER_CONTRACT_ALIGNMENT=" + ("PASS" if report["passed"] else "FAIL"))
    print(f"WORKER_COUNT={report['worker_count']}")
    print(f"ERRORS={len(report['errors'])}")
    for err in report["errors"]:
        print(err)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
