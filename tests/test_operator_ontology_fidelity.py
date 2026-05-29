#!/usr/bin/env python3
"""Operator ontology fidelity fixture/extraction guard.

This is a fixture_check when no extraction output is supplied. When an extraction
output path is supplied with --extraction-output or OPERATOR_ONTOLOGY_EXTRACTION_OUTPUT,
it also checks that output for softened/renamed substitutes.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "05_OUTPUTS" / "contracts" / "operator_ontology_labels.json"
OUT_DIR = ROOT / "05_OUTPUTS" / "contracts"

REQUIRED = [
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


def load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def flattened_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value


def fixture_check() -> dict[str, Any]:
    data = load_contract()
    labels = data.get("required_exact_labels", [])
    missing = [label for label in REQUIRED if label not in labels]
    duplicates = len(labels) != len(set(labels))
    forbidden = data.get("forbidden_softened_substitutes", {})
    promoted = []
    for canonical, substitutes in forbidden.items():
        if canonical not in labels:
            missing.append(canonical)
        promoted.extend(sub for sub in substitutes if sub in labels)
    passed = not missing and not duplicates and not promoted
    return {
        "mode": "fixture_check",
        "passed": passed,
        "required_label_count": len(REQUIRED),
        "fixture_label_count": len(labels),
        "missing": missing,
        "duplicates": duplicates,
        "softened_substitutes_promoted_in_fixture": promoted,
    }


def extraction_output_check(path: Path) -> dict[str, Any]:
    data = load_contract()
    output = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else path.read_text(encoding="utf-8")
    text = flattened_text(output)
    forbidden_hits = []
    for canonical, substitutes in data.get("forbidden_softened_substitutes", {}).items():
        for substitute in substitutes:
            if substitute in text and canonical not in text:
                forbidden_hits.append({"canonical": canonical, "softened_substitute": substitute})
    return {
        "mode": "extraction_output_check",
        "path": str(path),
        "passed": not forbidden_hits,
        "forbidden_softening_hits": forbidden_hits,
    }


def test_fixture_required_labels_exact() -> None:
    result = fixture_check()
    assert not result["missing"], f"missing exact Operator labels: {result['missing']}"
    assert not result["duplicates"], "duplicate labels in ontology fixture"


def test_fixture_forbidden_substitutes_not_promoted() -> None:
    result = fixture_check()
    assert not result["softened_substitutes_promoted_in_fixture"], result


def test_optional_extraction_output_does_not_soften_labels() -> None:
    path = os.environ.get("OPERATOR_ONTOLOGY_EXTRACTION_OUTPUT")
    if not path:
        return
    result = extraction_output_check(Path(path))
    assert result["passed"], result


def main() -> int:
    parser = argparse.ArgumentParser(description="Operator ontology fidelity fixture/extraction guard")
    parser.add_argument("--extraction-output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    checks = [fixture_check()]
    if args.extraction_output:
        checks.append(extraction_output_check(args.extraction_output))
    passed = all(check["passed"] for check in checks)
    report = {
        "schema": "lucidota.operator_ontology_fidelity_check.v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "overall_status": "PASS" if passed else "FAIL",
        "checks": checks,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = args.report or OUT_DIR / f"operator_ontology_fidelity_fixture_check_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["overall_status"], "mode": "+".join(check["mode"] for check in checks), "report": str(report_path.relative_to(ROOT))}, indent=2, sort_keys=True))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
