#!/usr/bin/env python3
"""Verify LUCIDOTA RFC program structure and source evidence."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import project_brain_doc_authority_check
import rfc_claim_discipline_check

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "00_PROJECT_BRAIN" / "RFCS" / "RFC_SUBJECT_REGISTRY.json"
BOUNDARY_FORBIDDEN = [
    "00_PROJECT_BRAIN/414_PRIMITIVE_CRIES",
    "06_SCHEMA/015_root414_packets.sql",
    "04_RUNTIME/root414_knowledge.sqlite",
    "BOOKS/ROOT414_GAME_GRADING_SCHEMA.json",
    "ahoy_sim",
    "06_SCHEMA/ahoy",
    "05_OUTPUTS/ahoy",
    "tests/fixtures/ahoy",
    "scripts/legacy/ahoy",
]
BOUNDARY_GLOBS = [
    "scripts/lucidota_414_*.py",
    "tests/test_ahoy_*.py",
]

REQUIRED_DOCS = [
    ROOT / "00_PROJECT_BRAIN" / "ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md",
    ROOT / "00_PROJECT_BRAIN" / "ACTIVE_SPEC/02_EXECUTION_SPINE.md",
    ROOT / "00_PROJECT_BRAIN" / "ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md",
    ROOT / "00_PROJECT_BRAIN" / "ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md",
    ROOT / "00_PROJECT_BRAIN" / "ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md",
    ROOT / "00_PROJECT_BRAIN" / "RFCS" / "RFC-000-MASTER-THESIS-PROGRAM.md",
    ROOT / "00_PROJECT_BRAIN" / "RFCS" / "RFC-001-SYSTEM-THESIS.md",
    ROOT / "00_PROJECT_BRAIN" / "RFCS" / "SOURCES.md",
]

RFC_DRAFT_STATUSES = {"active", "draft", "reviewed", "implemented", "verified"}
PLACEHOLDER_STATUS_FRAGMENTS = ("seeded",)
MIN_SUBJECT_RFC_WORDS = 750
MIN_SUBJECT_RFC_URLS = 2
RFC_DRAFT_SECTION_FAMILIES = {
    "status": ("Status:",),
    "thesis": ("## 1. Thesis", "## Thesis"),
    "source_anchors": ("## 2. Sources", "Source Anchors", "External Source", "External sources"),
    "whole_system_interaction": (
        "Whole-System Interaction",
        "Whole-system Interaction",
        "Whole System Interaction",
    ),
    "whole_system_benefit": ("Benefit to the Whole System", "Benefit"),
    "correctness_argument": ("Correctness Argument", "Correctness"),
    "falsifiers": ("Falsifiers",),
    "filesystem_runtime": (
        "Filesystem / Runtime Consequences",
        "Filesystem",
        "Runtime Consequences",
    ),
}


def resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def contains_any(text: str, needles: tuple[str, ...]) -> bool:
    folded = text.casefold()
    return any(needle.casefold() in folded for needle in needles)


def main() -> int:
    errors: list[str] = []
    if not REGISTRY.exists():
        errors.append(f"missing_registry:{REGISTRY.relative_to(ROOT)}")
        subjects = []
    else:
        try:
            data = json.loads(REGISTRY.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid_registry_json:{exc}")
            data = {"subjects": []}
        subjects = data.get("subjects", [])
        if data.get("schema") != "lucidota.rfc_subject_registry.v1":
            errors.append("bad_registry_schema")

    for p in REQUIRED_DOCS:
        if not p.exists() or p.stat().st_size == 0:
            errors.append(f"missing_required_doc:{p.relative_to(ROOT)}")

    if len(subjects) < 20:
        errors.append(f"too_few_subjects:{len(subjects)}")

    boundary_violations: list[str] = []
    for raw in BOUNDARY_FORBIDDEN:
        if (ROOT / raw).exists():
            boundary_violations.append(raw)
    for pattern in BOUNDARY_GLOBS:
        for p in ROOT.glob(pattern):
            boundary_violations.append(str(p.relative_to(ROOT)))
    for raw in boundary_violations:
        errors.append(f"boundary_violation:{raw}")

    claim_discipline_report = rfc_claim_discipline_check.build_report()
    doc_authority_report = project_brain_doc_authority_check.build_report()
    claim_discipline_violations = len(claim_discipline_report.get("errors", []))
    doc_authority_violations = len(doc_authority_report.get("errors", []))
    if claim_discipline_violations:
        errors.append(f"rfc_claim_discipline_violations:{claim_discipline_violations}")
    if doc_authority_violations:
        errors.append(f"project_brain_doc_authority_violations:{doc_authority_violations}")

    missing_evidence = 0
    missing_rfc_files = 0
    placeholder_subjects = 0
    draft_section_violations = 0
    depth_violations = 0
    source_coverage_violations = 0
    drafts_checked = 0
    seen_ids: set[str] = set()
    for idx, subject in enumerate(subjects):
        for key in ("id", "title", "rfc_path", "local_evidence", "status", "requires_deep_audit"):
            if key not in subject:
                errors.append(f"subject_{idx}_missing_key:{key}")
        sid = str(subject.get("id", ""))
        if sid in seen_ids:
            errors.append(f"duplicate_subject_id:{sid}")
        seen_ids.add(sid)
        if not subject.get("requires_deep_audit"):
            errors.append(f"subject_not_deep_audit:{sid}")
        rfc_path = resolve_path(str(subject.get("rfc_path", "")))
        if not rfc_path.exists():
            missing_rfc_files += 1
            errors.append(f"missing_rfc_file:{sid}:{subject.get('rfc_path')}")
        status = str(subject.get("status", ""))
        if any(fragment in status for fragment in PLACEHOLDER_STATUS_FRAGMENTS):
            placeholder_subjects += 1
            errors.append(f"placeholder_subject_status:{sid}:{status}")
        for raw in subject.get("local_evidence", []):
            if not resolve_path(str(raw)).exists():
                missing_evidence += 1
                errors.append(f"missing_evidence:{sid}:{raw}")
        if status in RFC_DRAFT_STATUSES:
            drafts_checked += 1
            if not rfc_path.exists():
                draft_section_violations += 1
                errors.append(f"draft_missing_rfc:{sid}:{subject.get('rfc_path')}")
                continue
            text = rfc_path.read_text(encoding="utf-8", errors="ignore")
            for family, needles in RFC_DRAFT_SECTION_FAMILIES.items():
                if not contains_any(text, needles):
                    draft_section_violations += 1
                    errors.append(f"draft_missing_section_family:{sid}:{family}")
            word_count = len(text.split())
            if word_count < MIN_SUBJECT_RFC_WORDS:
                depth_violations += 1
                errors.append(f"rfc_too_short:{sid}:{word_count}<{MIN_SUBJECT_RFC_WORDS}")
            url_count = len(re.findall(r"https?://", text))
            if url_count < MIN_SUBJECT_RFC_URLS:
                source_coverage_violations += 1
                errors.append(f"rfc_external_sources_too_few:{sid}:{url_count}<{MIN_SUBJECT_RFC_URLS}")
            required_local_refs = min(2, len(subject.get("local_evidence", [])))
            local_refs = sum(1 for raw in subject.get("local_evidence", []) if str(raw) in text)
            if local_refs < required_local_refs:
                source_coverage_violations += 1
                errors.append(f"rfc_local_evidence_refs_too_few:{sid}:{local_refs}<{required_local_refs}")

    print("RFC_PROGRAM_CHECK=" + ("PASS" if not errors else "FAIL"))
    print(f"SUBJECTS={len(subjects)}")
    print(f"MISSING_EVIDENCE={missing_evidence}")
    print(f"RFC_FILES_MISSING={missing_rfc_files}")
    print(f"SEEDED_OR_PLACEHOLDER_SUBJECTS={placeholder_subjects}")
    print(f"BOUNDARY_VIOLATIONS={len(boundary_violations)}")
    print(f"DRAFTS_CHECKED={drafts_checked}")
    print(f"DRAFT_SECTION_VIOLATIONS={draft_section_violations}")
    print(f"RFC_DEPTH_VIOLATIONS={depth_violations}")
    print(f"RFC_SOURCE_COVERAGE_VIOLATIONS={source_coverage_violations}")
    print(f"RFC_CLAIM_DISCIPLINE_VIOLATIONS={claim_discipline_violations}")
    print(f"PROJECT_BRAIN_DOC_AUTHORITY_VIOLATIONS={doc_authority_violations}")
    for err in errors:
        print(err)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
