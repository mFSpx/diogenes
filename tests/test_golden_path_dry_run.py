#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cep_full_e2e  # noqa: E402


def test_cep_full_e2e_dry_run_links_without_mutation_flags():
    seen: list[list[str]] = []

    def fake_runner(cmd: list[str]) -> dict[str, object]:
        seen.append(cmd)
        joined = " ".join(cmd)
        if "absurd_kernel_authorization.py" in joined:
            return {
                "cmd": joined,
                "rc": 2,
                "stdout_tail": '{"required": true, "ok": false, "error_kind": "missing_kernel_authorization"}',
                "stderr_tail": "",
            }
        if "graph_promotion_gate.py" in joined:
            return {"cmd": joined, "rc": 0, "stdout_tail": "REPORT_PATH=05_OUTPUTS/graph/probe.json\nGRAPH_GATE_ALLOWED=true\n", "stderr_tail": ""}
        if "graph_canonical_mutation_detector.py" in joined:
            return {"cmd": joined, "rc": 0, "stdout_tail": "REPORT_PATH=05_OUTPUTS/graph/mutation_detector.json\n", "stderr_tail": ""}
        if "lucidota_status_ledger.py" in joined:
            return {"cmd": joined, "rc": 0, "stdout_tail": "CHECK_OK status ledger valid\n", "stderr_tail": ""}
        return {"cmd": joined, "rc": 0, "stdout_tail": "REPORT_PATH=05_OUTPUTS/absurd/fixture.json\n", "stderr_tail": ""}

    result = cep_full_e2e.run_dry_run(fake_runner, operator_instruction="fixture instruction")

    assert result["status"] == "PASS_DRY_RUN"
    assert result["lineage_id"] == cep_full_e2e.stable_lineage_id("fixture instruction")
    assert result["execute_performed"] is False
    assert result["db_writes_performed"] is False
    assert result["canonical_graph_writes_performed"] is False
    assert result["link_status"]["kernel_authz_decision_exists"] == "REAL_DENIAL"
    assert result["receipt_refs"]
    assert any("graph_promotion_gate.py" in " ".join(cmd) and "--dry-run" in cmd for cmd in seen)


def test_graph_promotion_gate_exposes_pure_dry_run_flag():
    proc = subprocess.run(
        [sys.executable, "scripts/graph_promotion_gate.py", "gate", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--dry-run" in proc.stdout
    assert "--execute" in proc.stdout
