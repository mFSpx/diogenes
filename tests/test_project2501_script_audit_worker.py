from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_schema_defines_script_audit_run_and_preserves_graph_gate() -> None:
    schema = (ROOT / "06_SCHEMA" / "115_project2501_script_audit_worker.sql").read_text(encoding="utf-8")
    assert "lucidota_control.script_audit_run" in schema
    assert "lucidota_control.script_manifest" in schema
    assert "lucidota_control.corpse_manifest" in schema
    assert "canonical_graph_writes_performed boolean NOT NULL DEFAULT false" in schema
    assert "operator_feature_authority_required boolean NOT NULL DEFAULT true" in schema


def test_classify_legacy_script_corpses_without_deleting() -> None:
    from project2501_script_audit_worker import classify_script

    target = ROOT / "scripts" / "legacy" / "lucidota_big_board.py"
    assert target.exists()
    before = target.read_bytes()

    result = classify_script(target, max_callers=8)

    assert result["script_manifest"]["verdict"] == "CORPSE_MANIFEST"
    assert result["script_manifest"]["survival_score"] <= 35
    assert result["corpse_manifest"]["original_path"] == "scripts/legacy/lucidota_big_board.py"
    assert result["corpse_manifest"]["source_sha256"]
    assert result["canonical_graph_writes_performed"] is False
    assert target.read_bytes() == before


def test_classify_active_tested_script_keeps_or_wraps_with_receipt_contract() -> None:
    from project2501_script_audit_worker import classify_script

    result = classify_script(ROOT / "scripts" / "project2501_board_worker.py", max_callers=16)

    manifest = result["script_manifest"]
    assert manifest["verdict"] in {"KEEP", "WRAP"}
    assert manifest["survival_score"] >= 45
    assert manifest["receipt_contract"]["writes_receipt"] is True
    assert "tests/test_project2501_board_worker.py" in manifest["caller"]
    assert result["corpse_manifest"] is None


def test_cli_dry_run_writes_receipt_without_db_writes() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/project2501_script_audit_worker.py",
            "classify",
            "--path",
            "scripts/project2501_board_worker.py",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PROJECT2501_SCRIPT_AUDIT=PASS" in proc.stdout
    report_path = next(line.split("=", 1)[1] for line in proc.stdout.splitlines() if line.startswith("REPORT_PATH="))
    receipt = json.loads((ROOT / report_path).read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["execute_performed"] is False
    assert receipt["canonical_graph_writes_performed"] is False


def test_select_batch_candidates_prioritizes_active_high_loc_scripts() -> None:
    from project2501_script_audit_worker import select_batch_candidates

    candidates = select_batch_candidates(limit=4, min_nonblank_loc=400, max_files=900)

    assert candidates
    assert len(candidates) <= 4
    assert all(c["script_path"].startswith("scripts/") for c in candidates)
    assert all("/legacy/" not in c["script_path"] for c in candidates)
    assert all(c["nonblank_loc"] >= 400 for c in candidates)
    assert [c["nonblank_loc"] for c in candidates] == sorted((c["nonblank_loc"] for c in candidates), reverse=True)


def test_build_batch_result_emits_summary_and_preserves_graph_gate() -> None:
    from project2501_script_audit_worker import build_batch_result

    result = build_batch_result(
        paths=[
            ROOT / "scripts" / "project2501_board_worker.py",
            ROOT / "scripts" / "project2501_script_audit_worker.py",
        ],
        execute=False,
        max_callers=16,
    )

    assert result["schema"] == "lucidota.project2501.script_audit.batch.v1"
    assert result["status"] == "PASS"
    assert len(result["classifications"]) == 2
    assert result["summary"]["total"] == 2
    assert result["watch_metric"]["metric_key"] == "project2501_script_audit_batch"
    assert result["canonical_graph_writes_performed"] is False


def test_batch_cli_dry_run_writes_receipt_without_db_writes() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/project2501_script_audit_worker.py",
            "batch",
            "--limit",
            "2",
            "--min-nonblank-loc",
            "400",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PROJECT2501_SCRIPT_AUDIT=PASS" in proc.stdout
    report_path = next(line.split("=", 1)[1] for line in proc.stdout.splitlines() if line.startswith("REPORT_PATH="))
    receipt = json.loads((ROOT / report_path).read_text(encoding="utf-8"))
    assert receipt["schema"] == "lucidota.project2501.script_audit.batch.v1"
    assert receipt["execute_performed"] is False
    assert receipt["summary"]["total"] <= 2
    assert receipt["canonical_graph_writes_performed"] is False
