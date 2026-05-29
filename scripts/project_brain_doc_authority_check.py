#!/usr/bin/env python3
"""Check 00_PROJECT_BRAIN document/file authority shape.

This is not a metaphysical truth engine. It verifies the concrete filesystem
claim the operator challenged: top-level doc sprawl is counted, active specs live
in a named folder, and files cannot be casually renamed into "the manual" or an
arbitrary minimum-doc set.
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

PROJECT_BRAIN = ROOT / "00_PROJECT_BRAIN"
ACTIVE_SPEC = PROJECT_BRAIN / "ACTIVE_SPEC"
OUT_ROOT = "05_OUTPUTS/project_brain_doc_authority"
SCHEMA = "lucidota.project_brain_doc_authority.v1"

ACTIVE_SPEC_FILES = {
    "00_PROJECT_BRAIN/ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md": "identity_and_claim_state_law",
    "00_PROJECT_BRAIN/ACTIVE_SPEC/02_EXECUTION_SPINE.md": "execution_spine",
    "00_PROJECT_BRAIN/ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md": "custody_etl_pipeline",
    "00_PROJECT_BRAIN/ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md": "dev_library_reuse_law",
    "00_PROJECT_BRAIN/ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md": "component_authority_map",
    "00_PROJECT_BRAIN/ACTIVE_SPEC/06_BARE_STEEL_DOCTRINE.md": "bare_steel_doctrine",
    "00_PROJECT_BRAIN/ACTIVE_SPEC/07_WORKING_REALITY_LAW.md": "working_reality_law",
    "00_PROJECT_BRAIN/ACTIVE_SPEC/08_BOARD_EFFECT_TOURNAMENT_LAW.md": "board_effect_tournament_law",
}

FORBIDDEN_TOP_LEVEL_ALIASES = {
    "ROOT_DOCTRINE.md",
    "MAIN_SPINE.md",
    "FULL_ETL_PIPELINE.md",
    "DEV_LIBRARY.md",
    "RUNTIME_ORGANS.md",
}

# Top-level files are allowed only with a scoped role. This stops "whatever file I
# happened to read" from becoming the manual, doctrine, source of truth, etc.
TOP_LEVEL_ROLE_BY_FILE = {
    "ACTIVE_INSTRUCTION_INDEX.md": "instruction_source_index",
    "BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md": "workflow_hygiene_pseudolaw",
    "CANONICAL_FINISHED_PRODUCT_MAP.md": "supporting_product_map",
    "CODEX_PROMPTING_GUIDE_LUCIDOTA_POLICY.json": "codex_prompting_policy_machine_receipt",
    "CODEX_PROMPTING_GUIDE_LUCIDOTA_POLICY.md": "codex_prompting_policy_human_summary",
    "DOYOURJOB": "operator_instruction_artifact",
    "DURABLE_WORKFLOW_DECISION.json": "workflow_decision_record",
    "FILESYSTEM_TREE_INDEX_CURRENT.md": "generated_filesystem_tree_snapshot",
    "LUCIDOTA_CANONICAL_MANIFEST.md": "supporting_historical_manifest",
    "LUCIDOTA_GAPS_ACTION_WORKFLOWS.md": "supporting_gap_workflows",
    "LUCIDOTA_OPS_MANUAL.md": "operator_manual",
    "POSTGRES_AUDIT_CURRENT.md": "generated_postgres_audit_snapshot",
    "PROJECT_2501_ADMIN_PROMPT.md": "project2501_admin_prompt",
    "PROJECT_2501_CORE_CONTRACT.md": "project2501_core_contract",
    "READTHISFIRST_CURRENT.md": "generated_current_readme",
    "RUVECTOR_ABSURD_SONA_RIVERML_NOTES.md": "supporting_research_notes",
    "SCRIPTS_MANIFEST_V2.md": "supporting_script_manifest",
    "STATUS_LEDGER.md": "status_ledger",
    "TICKLETRUNK.json": "dev_library_machine_manifest",
    "TICKLETRUNK.md": "dev_library_human_manifest",
    "canonical_graph_write_allowlist.json": "graph_write_allowlist",
    "claude_code_claw_runtime_registry.json": "runtime_registry",
    "gpu_model_runtime_registry.json": "runtime_registry",
    "instruction_authority_registry.json": "instruction_authority_registry",
    "rust_port_candidacy_registry.json": "port_candidacy_registry",
    "spine_authority_registry.json": "spine_authority_registry",
}

MINIMUM_DOC_RE = re.compile(r"minimum[- ]canonical doc set|MINIMUM CANONICAL DOC|Minimum-Doc Rule", re.I)
MANUAL_TITLE_RE = re.compile(r"^#\s+.*\bmanual\b", re.I | re.M)
ALLOWED_MANUAL_TITLE_PATHS = {"00_PROJECT_BRAIN/LUCIDOTA_OPS_MANUAL.md"}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def top_level_files() -> list[Path]:
    return sorted(p for p in PROJECT_BRAIN.iterdir() if p.is_file())


def recursive_files() -> list[Path]:
    return sorted(p for p in PROJECT_BRAIN.rglob("*") if p.is_file())


def markdown_and_json_sources() -> list[Path]:
    return [p for p in recursive_files() if p.suffix.lower() in {".md", ".json"} or p.name == "DOYOURJOB"]


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def text_hits(paths: list[Path], pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for path in paths:
        text = read_text(path)
        for match in pattern.finditer(text):
            hits.append({"path": rel(path), "line": line_number(text, match.start()), "match": match.group(0)[:120]})
    return hits


def manual_title_hits(paths: list[Path]) -> list[dict[str, Any]]:
    hits = text_hits([p for p in paths if p.suffix.lower() == ".md"], MANUAL_TITLE_RE)
    return [hit for hit in hits if hit["path"] not in ALLOWED_MANUAL_TITLE_PATHS]


def build_report() -> dict[str, Any]:
    top = top_level_files()
    rec = recursive_files()
    top_names = {p.name for p in top}
    active_paths = {rel(p) for p in ACTIVE_SPEC.glob("*.md")} if ACTIVE_SPEC.exists() else set()
    expected_active = set(ACTIVE_SPEC_FILES)
    source_paths = markdown_and_json_sources()

    unmapped = sorted(name for name in top_names if name not in TOP_LEVEL_ROLE_BY_FILE)
    stale_aliases = sorted(name for name in FORBIDDEN_TOP_LEVEL_ALIASES if (PROJECT_BRAIN / name).exists())
    missing_active = sorted(expected_active - active_paths)
    extra_active = sorted(active_paths - expected_active)
    minimum_hits = text_hits(source_paths, MINIMUM_DOC_RE)
    manual_hits = manual_title_hits(source_paths)

    errors = []
    errors += [f"unmapped_top_level_file:{name}" for name in unmapped]
    errors += [f"stale_top_level_alias:{name}" for name in stale_aliases]
    errors += [f"missing_active_spec:{path}" for path in missing_active]
    errors += [f"extra_active_spec:{path}" for path in extra_active]
    errors += [f"minimum_doc_language:{hit['path']}:{hit['line']}" for hit in minimum_hits]
    errors += [f"unauthorized_manual_title:{hit['path']}:{hit['line']}" for hit in manual_hits]

    return {
        "schema": SCHEMA,
        "passed": not errors,
        "project_brain_path": rel(PROJECT_BRAIN),
        "top_level_file_count": len(top),
        "top_level_dir_count": len([p for p in PROJECT_BRAIN.iterdir() if p.is_dir()]),
        "recursive_file_count": len(rec),
        "active_spec_count": len(active_paths),
        "top_level_roles": {f"00_PROJECT_BRAIN/{name}": role for name, role in sorted(TOP_LEVEL_ROLE_BY_FILE.items())},
        "active_spec_files": ACTIVE_SPEC_FILES,
        "unmapped_top_level_files": unmapped,
        "stale_top_level_aliases": stale_aliases,
        "missing_active_specs": missing_active,
        "extra_active_specs": extra_active,
        "minimum_doc_language_hits": minimum_hits,
        "unauthorized_manual_title_hits": manual_hits,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 00_PROJECT_BRAIN document authority shape")
    parser.add_argument("--json", action="store_true", help="write JSON receipt")
    args = parser.parse_args()
    report = build_report()
    if args.json:
        receipt("project_brain_doc_authority", report, root=OUT_ROOT)
    print("PROJECT_BRAIN_DOC_AUTHORITY=" + ("PASS" if report["passed"] else "FAIL"))
    print(f"TOP_LEVEL_FILES={report['top_level_file_count']}")
    print(f"ACTIVE_SPEC_FILES={report['active_spec_count']}")
    print(f"RECURSIVE_FILES={report['recursive_file_count']}")
    for err in report["errors"]:
        print(err)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
