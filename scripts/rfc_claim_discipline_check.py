#!/usr/bin/env python3
"""Enforce RFC claim discipline from the operator's #1/#2 corrections.

ABBA3^5 is treated here as a local operator audit instruction, not as an
established external field term. The checker is intentionally structural: it
forces each subject RFC to declare the five lenses before the prose can be
mistaken for self-justifying authority.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from spine_common import ROOT, receipt, rel  # noqa: E402

REGISTRY = ROOT / "00_PROJECT_BRAIN" / "RFCS" / "RFC_SUBJECT_REGISTRY.json"
SCHEMA = "lucidota.rfc_claim_discipline.v1"
OUT_ROOT = "05_OUTPUTS/rfcs"
OPERATOR_AUDIT_LABEL = "ABBA3^5"
OPERATOR_AUDIT_LABEL_STATUS = "local_operator_instruction_not_external_domain_term"

REQUIRED_LENSES = {
    "claim_state": re.compile(r"Claim-state\s*:", re.I),
    "provenance_count_and_reason": re.compile(r"Provenance-count-and-reason\s*:", re.I),
    "naming_integrity": re.compile(r"Naming-integrity\s*:", re.I),
    "reuse_before_reinvention": re.compile(r"Reuse-before-reinvention\s*:", re.I),
    "operational_proportionality": re.compile(r"Operational-proportionality\s*:", re.I),
}
SECTION_RE = re.compile(r"^##\s+0\.\s+Claim Discipline", re.M)
LOCAL_LABEL_RE = re.compile(r"ABBA3\^5.*local operator instruction.*not an established external", re.I | re.S)


def load_subject_rfcs() -> list[dict[str, Any]]:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return [s for s in data.get("subjects", []) if str(s.get("status")) in {"draft", "reviewed", "implemented", "verified", "active"}]


def audit_rfc(subject: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / str(subject["rfc_path"])
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    present = sorted(name for name, pattern in REQUIRED_LENSES.items() if pattern.search(text))
    missing = sorted(set(REQUIRED_LENSES) - set(present))
    has_section = bool(SECTION_RE.search(text))
    local_label = bool(LOCAL_LABEL_RE.search(text))
    return {
        "subject_id": subject.get("id"),
        "title": subject.get("title"),
        "path": rel(path),
        "exists": path.exists(),
        "has_claim_discipline_section": has_section,
        "declares_abba3_5_local": local_label,
        "present_lenses": present,
        "missing_lenses": missing,
    }


def build_report() -> dict[str, Any]:
    rows = [audit_rfc(subject) for subject in load_subject_rfcs()]
    missing_section = [row["path"] for row in rows if not row["has_claim_discipline_section"]]
    missing_lenses = [row for row in rows if row["missing_lenses"]]
    missing_local = [row["path"] for row in rows if not row["declares_abba3_5_local"]]
    errors = []
    errors += [f"missing_claim_discipline:{path}" for path in missing_section]
    errors += [f"missing_abba3_5_local_label:{path}" for path in missing_local]
    for row in missing_lenses:
        errors.append(f"missing_required_lenses:{row['path']}:{','.join(row['missing_lenses'])}")
    return {
        "schema": SCHEMA,
        "passed": not errors and len(rows) == 20,
        "operator_audit_label": OPERATOR_AUDIT_LABEL,
        "operator_audit_label_status": OPERATOR_AUDIT_LABEL_STATUS,
        "required_lenses": sorted(REQUIRED_LENSES),
        "subject_rfc_count": len(rows),
        "rfc_rows": rows,
        "rfcs_missing_claim_discipline": missing_section,
        "rfcs_missing_required_lenses": missing_lenses,
        "rfcs_missing_local_label_statement": missing_local,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check RFC claim discipline gates")
    parser.add_argument("--json", action="store_true", help="write JSON receipt")
    args = parser.parse_args()
    report = build_report()
    if args.json:
        receipt("rfc_claim_discipline", report, root=OUT_ROOT)
    print("RFC_CLAIM_DISCIPLINE=" + ("PASS" if report["passed"] else "FAIL"))
    print(f"SUBJECT_RFC_COUNT={report['subject_rfc_count']}")
    print(f"MISSING_CLAIM_DISCIPLINE={len(report['rfcs_missing_claim_discipline'])}")
    print(f"MISSING_REQUIRED_LENSES={len(report['rfcs_missing_required_lenses'])}")
    for err in report["errors"]:
        print(err)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
