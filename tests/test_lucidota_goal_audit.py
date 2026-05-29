import subprocess
import sys
from pathlib import Path


def test_lucidota_goal_audit_passes_against_requirement_matrix():
    result = subprocess.run(
        [sys.executable, "scripts/lucidota_goal_audit.py"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LUCIDOTA_GOAL_AUDIT=PASS" in result.stdout
    assert "UNPROVEN_REQUIREMENTS=0" in result.stdout
    assert "EVIDENCE_MISSING=0" in result.stdout


def test_lucidota_goal_audit_includes_project_brain_doc_authority_gate():
    result = subprocess.run(
        [sys.executable, "scripts/lucidota_goal_audit.py"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    audit_doc = (Path(__file__).resolve().parents[1] / "00_PROJECT_BRAIN" / "RFCS" / "GOAL_COMPLETION_AUDIT.md").read_text(encoding="utf-8")
    assert "project_brain_doc_authority" in audit_doc
    assert "rfc_claim_discipline" in audit_doc


def test_lucidota_goal_audit_includes_absurd_remaining_worker_contract_gate():
    result = subprocess.run(
        [sys.executable, "scripts/lucidota_goal_audit.py"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    audit_doc = (Path(__file__).resolve().parents[1] / "00_PROJECT_BRAIN" / "RFCS" / "GOAL_COMPLETION_AUDIT.md").read_text(encoding="utf-8")
    assert "absurd_remaining_worker_contract_alignment" in audit_doc
