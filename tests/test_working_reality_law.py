from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.working_reality_record import (
    build_working_reality_record,
    validate_working_reality_record,
)

ROOT = Path(__file__).resolve().parents[1]


def test_working_reality_law_is_active_canon() -> None:
    law = (ROOT / "00_PROJECT_BRAIN" / "ACTIVE_SPEC" / "07_WORKING_REALITY_LAW.md").read_text(encoding="utf-8")
    index = (ROOT / "00_PROJECT_BRAIN" / "ACTIVE_INSTRUCTION_INDEX.md").read_text(encoding="utf-8")
    core = (ROOT / "00_PROJECT_BRAIN" / "PROJECT_2501_CORE_CONTRACT.md").read_text(encoding="utf-8")

    assert "Ontology is not Truth." in law
    assert "Ontology is a structured hypothesis ecology." in law
    assert "OPERATOR = abductive chooser of working reality under receipt discipline." in law
    assert "REALITY" in law and "EVIDENCE" in law and "HYPOTHESIS" in law and "WORKING REALITY" in law
    assert "ACTIVE_SPEC/07_WORKING_REALITY_LAW.md" in index
    assert "working reality" in core.lower()


def test_official_ontology_declares_humility_contract_and_terms() -> None:
    ontology = json.loads((ROOT / "OFFICIAL_ONTOLOGY.json").read_text(encoding="utf-8"))
    terms = set(ontology["extension_terms"])
    contract = ontology["ontology_humility_contract"]

    assert {"REALITY", "WORKING_REALITY", "CONTRADICTION"}.issubset(terms)
    assert ontology["active_term_count_with_extensions"] == 48
    assert contract["ontology_is_truth"] is False
    assert contract["layer_order"] == ["REALITY", "EVIDENCE", "HYPOTHESIS", "WORKING_REALITY"]
    assert contract["operator_role"] == "abductive_chooser_of_working_reality_under_receipt_discipline"
    assert contract["rejected_hypotheses_preserved"] is True


def test_build_working_reality_record_preserves_future_replay_contract() -> None:
    record = build_working_reality_record(
        evidence=["receipt://05_OUTPUTS/example.json", "hash://abc123"],
        hypothesis="Graph writes are blocked by admission policy.",
        working_reality="Treat graph writes as blocked until a verified allow-gate exists.",
        move="Stage graph candidates instead of claiming canonical write.",
        result="PASS",
    )

    assert record["schema"] == "lucidota.working_reality.record.v1"
    assert record["layers"]["reality"]["claim"] == "exists_beyond_system_possession"
    assert record["layers"]["evidence"]["refs"] == ["receipt://05_OUTPUTS/example.json", "hash://abc123"]
    assert record["hypothesis"] == "Graph writes are blocked by admission policy."
    assert record["working_reality"].startswith("Treat graph writes as blocked")
    assert record["record_for_future"] is True
    assert validate_working_reality_record(record) == []


def test_working_reality_cli_writes_receipt_and_optional_db_is_off_by_default() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/working_reality_record.py",
            "--evidence",
            "receipt://05_OUTPUTS/example.json",
            "--hypothesis",
            "Ontology is map, not territory.",
            "--working-reality",
            "Operate on the current map while preserving contradiction.",
            "--move",
            "Record the move with replayable evidence.",
            "--result",
            "PASS",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stderr
    assert "WORKING_REALITY_RECORD=PASS" in proc.stdout
    receipt_line = next(line for line in proc.stdout.splitlines() if line.startswith("RECEIPT_PATH="))
    receipt = json.loads((ROOT / receipt_line.split("=", 1)[1]).read_text(encoding="utf-8"))
    assert receipt["execute_performed"] is False
    assert receipt["record"]["record_for_future"] is True
    assert receipt["record"]["canonical_graph_writes_performed"] is False
