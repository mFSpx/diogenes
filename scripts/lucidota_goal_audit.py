#!/usr/bin/env python3
"""Requirement-level audit for the LUCIDOTA RFC/organization goal.

This script intentionally audits the user's stated DONE criteria against current
files, registry subjects, and verifier outputs. It is not a replacement for the
RFCs; it is the receipt-producing checklist that prevents a shallow "all green"
claim from becoming the definition of done.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "00_PROJECT_BRAIN" / "RFCS" / "GOAL_REQUIREMENT_MATRIX.json"
REGISTRY = ROOT / "00_PROJECT_BRAIN" / "RFCS" / "RFC_SUBJECT_REGISTRY.json"
AUDIT_MD = ROOT / "00_PROJECT_BRAIN" / "RFCS" / "GOAL_COMPLETION_AUDIT.md"
OUT_DIR = ROOT / "05_OUTPUTS" / "rfcs"

COMMANDS = {

    "absurd_remaining_worker_contract_alignment": {
        "argv": [sys.executable, "scripts/absurd_remaining_worker_contract_alignment_check.py", "--json"],
        "contains": [
            "ABSURD_REMAINING_WORKER_CONTRACT_ALIGNMENT=PASS",
            "WORKER_COUNT=4",
            "ERRORS=0",
        ],
    },
    "rfc_claim_discipline": {
        "argv": [sys.executable, "scripts/rfc_claim_discipline_check.py", "--json"],
        "contains": [
            "RFC_CLAIM_DISCIPLINE=PASS",
            "SUBJECT_RFC_COUNT=20",
            "MISSING_CLAIM_DISCIPLINE=0",
            "MISSING_REQUIRED_LENSES=0",
        ],
    },
    "rfc_program_check": {
        "argv": [sys.executable, "scripts/rfc_program_check.py"],
        "contains": [
            "RFC_PROGRAM_CHECK=PASS",
            "MISSING_EVIDENCE=0",
            "RFC_FILES_MISSING=0",
            "SEEDED_OR_PLACEHOLDER_SUBJECTS=0",
            "BOUNDARY_VIOLATIONS=0",
            "DRAFTS_CHECKED=20",
            "DRAFT_SECTION_VIOLATIONS=0",
            "RFC_DEPTH_VIOLATIONS=0",
            "RFC_SOURCE_COVERAGE_VIOLATIONS=0",
            "RFC_CLAIM_DISCIPLINE_VIOLATIONS=0",
            "PROJECT_BRAIN_DOC_AUTHORITY_VIOLATIONS=0",
        ],
    },
    "dev_library_check": {
        "argv": [sys.executable, "scripts/dev_library_scan.py", "--check"],
        "contains": ["CHECK_OK TICKLETRUNK valid"],
    },

    "project_brain_doc_authority": {
        "argv": [sys.executable, "scripts/project_brain_doc_authority_check.py", "--json"],
        "contains": [
            "PROJECT_BRAIN_DOC_AUTHORITY=PASS",
            "TOP_LEVEL_FILES=26",
            "ACTIVE_SPEC_FILES=8",
        ],
    },
    "status_ledger_check": {
        "argv": [sys.executable, "scripts/lucidota_status_ledger.py", "--check"],
        "contains": ["CHECK_OK status ledger valid"],
    },
    "pytest_focused": {
        "argv": [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_rfc_program_check.py",
            "tests/test_dev_library_scan_wrapper.py",
            "tests/test_instruction_authority_registry.py",
            "tests/test_slop_audit_law.py",
            "tests/test_project_brain_doc_authority.py",
            "tests/test_rfc_claim_discipline.py",
            "tests/test_lucidota_progress.py",
            "tests/test_absurd_remaining_worker_contract_alignment.py",
            "-q",
        ],
        "contains": ["passed"],
    },
}


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def run_command(name: str) -> dict[str, Any]:
    spec = COMMANDS[name]
    proc = subprocess.run(
        spec["argv"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
        check=False,
    )
    text = proc.stdout + proc.stderr
    missing = [needle for needle in spec["contains"] if needle not in text]
    return {
        "name": name,
        "argv": spec["argv"],
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-2000:],
        "required_substrings_missing": missing,
        "passed": proc.returncode == 0 and not missing,
    }


def load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def subject_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(s.get("id")): s for s in registry.get("subjects", [])}


def evaluate_requirement(req: dict[str, Any], subjects: dict[str, dict[str, Any]], command_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing_paths = []
    for raw in req.get("evidence_paths", []):
        p = resolve_path(str(raw))
        if not p.exists():
            missing_paths.append(str(raw))

    missing_subjects = []
    weak_subjects = []
    for sid in req.get("rfc_ids", []):
        s = subjects.get(str(sid))
        if not s:
            missing_subjects.append(str(sid))
            continue
        rfc = resolve_path(str(s.get("rfc_path", "")))
        if not rfc.exists():
            missing_subjects.append(str(sid))
        if str(s.get("status")) not in {"draft", "reviewed", "implemented", "verified"}:
            weak_subjects.append(f"{sid}:{s.get('status')}")

    failed_commands = []
    for cname in req.get("command_checks", []):
        if not command_results.get(cname, {}).get("passed"):
            failed_commands.append(cname)

    proven = not missing_paths and not missing_subjects and not weak_subjects and not failed_commands
    return {
        "id": req.get("id"),
        "title": req.get("title"),
        "claim": req.get("claim"),
        "proven": proven,
        "missing_evidence_paths": missing_paths,
        "missing_rfc_subjects_or_files": missing_subjects,
        "weak_rfc_statuses": weak_subjects,
        "failed_command_checks": failed_commands,
        "evidence_paths": req.get("evidence_paths", []),
        "rfc_ids": req.get("rfc_ids", []),
        "command_checks": req.get("command_checks", []),
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# LUCIDOTA Goal Completion Audit",
        "",
        "<!--",
        "DEV NOTE (anti-slop provenance, 2026-05-24): A file/report/taxonomy is proof of the truth it actually declares: this generated report exists, has this title, and asserts this scoped audit content. It is not proof of unrelated external facts, not permission to rename its scope as \"the manual,\" and not evidence that the operator personally coined a local label. Names are integral to their own statements and contexts; do not erase, swap, or flatten them. When an agent says \"this is the user's rule,\" it must cite direct operator instructions or accepted doctrine/RFC/receipt lineage, state exactly how many source docs were consulted and why those docs were chosen, and distinguish direct evidence from inference or compression. If the system lacks receipts for a factual CLAIM, classify it as hypothesis, observation, inference, suggestion, or confidence-rated candidate instead.",
        "-->",
        "",
        f"Status: {'PASS' if report['passed'] else 'FAIL'}",
        f"Generated: {report['generated_at']}",
        "",
        "This audit maps the user's DONE criteria to current files, RFCs, verifier output, and receipts. It is intentionally stricter than a file list.",
        "",
        "## Summary",
        "",
        f"- Requirements: {report['requirement_count']}",
        f"- Proven requirements: {report['proven_requirement_count']}",
        f"- Unproven requirements: {report['unproven_requirement_count']}",
        f"- Missing evidence paths: {report['missing_evidence_count']}",
        "",
        "## Command Evidence",
        "",
    ]
    for name, result in report["command_results"].items():
        lines.append(f"- `{name}`: {'PASS' if result['passed'] else 'FAIL'}")
    lines += ["", "## Requirement Matrix", "", "| ID | Status | Requirement | Evidence |", "|---|---|---|---|"]
    for row in report["requirements"]:
        status = "PASS" if row["proven"] else "FAIL"
        evidence = "; ".join(row.get("evidence_paths", [])[:4])
        if len(row.get("evidence_paths", [])) > 4:
            evidence += "; …"
        lines.append(f"| {row['id']} | {status} | {row['title']} | `{evidence}` |")
    lines += ["", "## Notes", "", "A PASS here does not mean the system is metaphysically finished forever. It means the current requested RFC/organization plan is backed by current evidence and repeatable checks.", ""]
    AUDIT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    errors: list[str] = []
    if not MATRIX.exists():
        errors.append(f"missing_matrix:{rel(MATRIX)}")
        matrix = {"requirements": []}
    else:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        if matrix.get("schema") != "lucidota.goal_requirement_matrix.v1":
            errors.append("bad_matrix_schema")
    registry = load_registry()
    subjects = subject_map(registry)

    command_names = sorted({c for req in matrix.get("requirements", []) for c in req.get("command_checks", [])})
    command_results = {name: run_command(name) for name in command_names}
    requirement_results = [evaluate_requirement(req, subjects, command_results) for req in matrix.get("requirements", [])]

    missing_evidence_count = sum(len(r["missing_evidence_paths"]) for r in requirement_results)
    unproven = [r for r in requirement_results if not r["proven"]]
    passed = not errors and not unproven and missing_evidence_count == 0 and len(requirement_results) >= 20

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / f"lucidota_goal_audit_{stamp()}.json"
    report = {
        "schema": "lucidota.goal_audit_report.v1",
        "generated_at": now_z(),
        "passed": passed,
        "matrix_path": rel(MATRIX),
        "audit_doc_path": rel(AUDIT_MD),
        "requirement_count": len(requirement_results),
        "proven_requirement_count": len(requirement_results) - len(unproven),
        "unproven_requirement_count": len(unproven),
        "missing_evidence_count": missing_evidence_count,
        "errors": errors,
        "command_results": command_results,
        "requirements": requirement_results,
    }
    report["report_path"] = rel(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_markdown(report)

    print("LUCIDOTA_GOAL_AUDIT=" + ("PASS" if passed else "FAIL"))
    print(f"REQUIREMENTS={len(requirement_results)}")
    print(f"UNPROVEN_REQUIREMENTS={len(unproven)}")
    print(f"EVIDENCE_MISSING={missing_evidence_count}")
    print(f"REPORT_PATH={rel(report_path)}")
    print(f"AUDIT_DOC={rel(AUDIT_MD)}")
    for err in errors:
        print(err)
    for row in unproven:
        print("unproven:" + str(row["id"]))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
