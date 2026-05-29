#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def receipt_path(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("RECEIPT_PATH="):
            return ROOT / line.split("=", 1)[1]
    raise AssertionError(stdout)


def test_ontology_staging_receipt_conforms_to_json_schema_contract(tmp_path: Path) -> None:
    src = tmp_path / "claim.md"
    src.write_text("ENTITY has ATTRIBUTE. Evidence supports CLAIM. Event happens over TIME in 2026.", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "scripts/ontology_staging_contract.py", "--source-file", str(src), "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ONTOLOGY_STAGE=SCHEMA_VALID" in proc.stdout
    data = json.loads(receipt_path(proc.stdout).read_text(encoding="utf-8"))
    assert data["schema_valid"] is True
    assert data["schema_errors"] == []
    assert isinstance(data["confidence"], int)
    assert 0 <= data["confidence"] <= 10000
    assert data["direct_truth_promotion_performed"] is False
