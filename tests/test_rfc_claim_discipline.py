import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_check():
    return subprocess.run(
        [sys.executable, "scripts/rfc_claim_discipline_check.py", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
    )


def _receipt(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("RECEIPT_PATH="):
            return json.loads((ROOT / line.split("=", 1)[1]).read_text(encoding="utf-8"))
    raise AssertionError(stdout)


def test_rfc_claim_discipline_check_passes_all_20_subject_rfcs():
    proc = _run_check()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RFC_CLAIM_DISCIPLINE=PASS" in proc.stdout
    data = _receipt(proc.stdout)
    assert data["schema"] == "lucidota.rfc_claim_discipline.v1"
    assert data["subject_rfc_count"] == 20
    assert data["rfcs_missing_claim_discipline"] == []
    assert data["rfcs_missing_required_lenses"] == []


def test_claim_discipline_names_abba3_5_as_local_operator_audit_not_external_term():
    proc = _run_check()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = _receipt(proc.stdout)
    assert data["operator_audit_label"] == "ABBA3^5"
    assert data["operator_audit_label_status"] == "local_operator_instruction_not_external_domain_term"
    for row in data["rfc_rows"]:
        assert row["declares_abba3_5_local"] is True


def test_claim_discipline_enforces_the_five_user_corrections():
    proc = _run_check()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = _receipt(proc.stdout)
    expected = {
        "claim_state",
        "provenance_count_and_reason",
        "naming_integrity",
        "reuse_before_reinvention",
        "operational_proportionality",
    }
    assert set(data["required_lenses"]) == expected
    for row in data["rfc_rows"]:
        assert set(row["present_lenses"]) == expected
