#!/usr/bin/env python3
"""Phase 0.5 Brain Archaeology scaffold/prep runner.

Default mode is dry-run. This script does not perform full corpus ingestion.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "06_SCHEMA" / "030_phase05_brain_archaeology.sql"
OUT_DIR = ROOT / "05_OUTPUTS" / "phase05"
SECURITY_DIR = ROOT / "05_OUTPUTS" / "security"

REQUIRED_TABLES = [
    "operator_syllabus_seed",
    "classifier_label",
    "sticker_feature_vector",
    "telemetry_finding",
    "topology_finding",
    "design_atom",
    "workflow_blueprint",
    "syllabus_fidelity_violation",
    "master_eye_review",
]
REQUIRED_LABELS = [
    "Operator",
    "Rainmaker",
    "Paladin / God-Mode",
    "Psyche / State-Collapse",
    "Forensic Shield",
    "Infinite Sink",
    "Anchor Weight",
    "Server Wipe",
    "API Rate Limiting",
    "Environment Migration",
    "Cruelty Protocols",
    "Master’s Eye",
    "Chrono-Ledger",
    "KRAMPUSCHEWING",
    "KORPUS",
    "DIOGENES",
    "FairyFuse",
    "Job Fair Allocator",
    "Darwinian Surfaces",
    "Command Envelope Protocol",
]
FORBIDDEN_SQL = re.compile(r"\b(DROP|TRUNCATE|DELETE\s+FROM|ALTER\s+TABLE\s+.*\s+DROP)\b", re.I | re.S)


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def latest_security_manifest() -> Path | None:
    manifests = sorted(SECURITY_DIR.glob("security_quarantine_manifest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return manifests[0] if manifests else None


def read_security_state() -> dict:
    p = latest_security_manifest()
    if not p:
        return {"manifest_path": None, "clean_manifest": False, "blocker": "security_quarantine_manifest_missing"}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {"manifest_path": str(p.relative_to(ROOT)), "clean_manifest": bool(data.get("clean_manifest")), "blocker": None if data.get("clean_manifest") else "security_quarantine_manifest_not_clean"}
    except Exception as exc:
        return {"manifest_path": str(p.relative_to(ROOT)), "clean_manifest": False, "blocker": f"security_quarantine_manifest_unreadable:{exc}"}


def schema_check() -> dict:
    result = {"schema_path": str(SCHEMA.relative_to(ROOT)), "exists": SCHEMA.exists(), "required_tables": {}, "forbidden_sql_found": []}
    if not SCHEMA.exists():
        return result
    text = SCHEMA.read_text(encoding="utf-8")
    for table in REQUIRED_TABLES:
        result["required_tables"][table] = bool(re.search(rf"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+lucidota_archaeology\.{re.escape(table)}\b", text, re.I))
    result["forbidden_sql_found"] = sorted(set(m.group(1).upper() for m in FORBIDDEN_SQL.finditer(text)))
    return result


def run_psql_schema(database_url: str) -> dict:
    cmd = ["psql", database_url, "-v", "ON_ERROR_STOP=1", "-f", str(SCHEMA)]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return {"command": "psql <redacted_database_url> -v ON_ERROR_STOP=1 -f 06_SCHEMA/030_phase05_brain_archaeology.sql", "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}


def catchme_classify_paths(paths: list[str], database_url: str) -> list[dict]:
    decisions = []
    for path in paths:
        cmd = [sys.executable, "scripts/catchme_sensitivity_map.py", "--database-url", database_url, "classify", "--path", path, "--requested-use", "phase05_loader_context_use"]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        report_path = None
        for line in proc.stdout.splitlines():
            if line.startswith("REPORT_PATH="):
                report_path = line.split("=", 1)[1]
        decision = {"path": path, "returncode": proc.returncode, "report_path": report_path, "allowed": proc.returncode == 0, "stdout_tail": proc.stdout[-1000:], "stderr_tail": proc.stderr[-1000:]}
        if report_path and (ROOT / report_path).exists():
            try:
                data = json.loads((ROOT / report_path).read_text(encoding="utf-8"))
                decision.update({"allowed": bool(data.get("allowed")), "sensitivity_class": data.get("sensitivity_class"), "consent_status": data.get("consent_status"), "reason": data.get("reason")})
            except Exception as exc:
                decision["parse_error"] = str(exc)
        decisions.append(decision)
    return decisions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--execute-schema", action="store_true", help="Create schema only; does not ingest corpus")
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL", "postgresql:///lucidota_storage"))
    ap.add_argument("--catchme-database-url", default=os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
    ap.add_argument("--context-path", action="append", default=[], help="Path refs Phase 0.5 loaders want to use; checked through CatchMe")
    args = ap.parse_args()

    dry_run = args.dry_run or not args.execute_schema
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    check = schema_check()
    security = read_security_state()
    catchme_paths = args.context_path or (["00_PROJECT_BRAIN/SPEC-001.5-BRAIN-ARCHAEOLOGY.md"] if (ROOT / "00_PROJECT_BRAIN/SPEC-001.5-BRAIN-ARCHAEOLOGY.md").exists() else ["00_PROJECT_BRAIN/STATUS_LEDGER.md"])
    catchme_decisions = catchme_classify_paths(catchme_paths, args.catchme_database_url)
    blockers = []
    if not check["exists"]:
        blockers.append("phase05_schema_missing")
    missing_tables = [k for k, v in check.get("required_tables", {}).items() if not v]
    if missing_tables:
        blockers.append("phase05_schema_missing_tables:" + ",".join(missing_tables))
    if check.get("forbidden_sql_found"):
        blockers.append("phase05_schema_contains_forbidden_sql:" + ",".join(check["forbidden_sql_found"]))
    if not security.get("manifest_path"):
        blockers.append("security_quarantine_manifest_missing")
    full_ingest_allowed = bool(security.get("clean_manifest"))
    if not full_ingest_allowed:
        blockers.append(security.get("blocker") or "security_quarantine_manifest_not_clean")
    blocked_context = [d for d in catchme_decisions if not d.get("allowed")]
    if blocked_context:
        blockers.append("catchme_context_blocked:" + ",".join(d.get("path", "") for d in blocked_context))

    psql_result = None
    execute_performed = False
    if args.execute_schema:
        if not blockers or blockers == ["security_quarantine_manifest_not_clean"] or security.get("blocker"):
            # Schema execution is allowed even when full ingest remains blocked.
            psql_result = run_psql_schema(args.database_url)
            execute_performed = psql_result["returncode"] == 0
            if psql_result["returncode"] != 0:
                blockers.append("schema_execute_failed")
        else:
            blockers.append("schema_execute_blocked_by_schema_check")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "dry_run" if dry_run else "execute_schema",
        "execute_performed": execute_performed,
        "full_ingest_performed": False,
        "full_ingest_allowed": full_ingest_allowed,
        "security_manifest": security,
        "catchme_loader_context_decisions": catchme_decisions,
        "schema_check": check,
        "required_labels": REQUIRED_LABELS,
        "workflow_scaffolds": [
            "phase05_syllabus_seed_workflow",
            "phase05_korpus_componentize_workflow",
            "phase05_chrono_ledger_workflow",
            "phase05_sticker_vector_workflow",
            "phase05_streaming_brain_workflow",
            "phase05_cruelty_protocols_workflow",
            "phase05_design_atom_recovery_workflow",
            "phase05_workflow_blueprint_recovery_workflow",
            "phase05_fidelity_review_workflow",
            "phase05_operator_memory_graph_workflow",
        ],
        "psql_result": psql_result,
        "blockers": sorted(set(b for b in blockers if b)),
    }
    out = OUT_DIR / f"phase05_brain_archaeology_prep_{ts()}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    print(f"REPORT_PATH={out.relative_to(ROOT)}")
    print(f"FULL_INGEST_ALLOWED={str(full_ingest_allowed).lower()}")
    return 0 if check["exists"] and not missing_tables and not check.get("forbidden_sql_found") else 1


if __name__ == "__main__":
    raise SystemExit(main())
