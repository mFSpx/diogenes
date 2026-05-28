from __future__ import annotations
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
AGG = "05_OUTPUTS/golden_path/golden_path_single_instruction_20260517T233140Z.json"

def test_status_ledger_broad_myth_upgrade_fails():
    p = subprocess.run([sys.executable, "scripts/status_ledger_evidence_gate.py", "--evidence", AGG, "--claim", "Whole LUCIDOTA system complete", "--proposed-status", "verified"], cwd=ROOT, text=True, capture_output=True)
    assert p.returncode == 4
    assert "STATUS_LEDGER_EVIDENCE_GATE=FAIL" in p.stdout

def test_status_ledger_scoped_evidence_claim_passes():
    p = subprocess.run([sys.executable, "scripts/status_ledger_evidence_gate.py", "--evidence", AGG, "--claim", "Golden Path v1 single-instruction no-materialization lane crowned", "--proposed-status", "verified"], cwd=ROOT, text=True, capture_output=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "STATUS_LEDGER_EVIDENCE_GATE=PASS" in p.stdout
