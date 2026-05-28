import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_check():
    return subprocess.run(
        [sys.executable, "scripts/project_brain_doc_authority_check.py", "--json"],
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


def test_project_brain_doc_authority_check_passes_and_counts_scope():
    proc = _run_check()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PROJECT_BRAIN_DOC_AUTHORITY=PASS" in proc.stdout
    data = _receipt(proc.stdout)
    assert data["schema"] == "lucidota.project_brain_doc_authority.v1"
    assert data["top_level_file_count"] == 26
    assert data["active_spec_count"] == 8
    assert data["recursive_file_count"] >= 125
    assert data["unmapped_top_level_files"] == []


def test_active_spec_files_are_in_dedicated_folder_not_top_level_aliases():
    forbidden = [
        "ROOT_DOCTRINE.md",
        "MAIN_SPINE.md",
        "FULL_ETL_PIPELINE.md",
        "DEV_LIBRARY.md",
        "RUNTIME_ORGANS.md",
    ]
    for name in forbidden:
        assert not (ROOT / "00_PROJECT_BRAIN" / name).exists(), name
    expected = {
        "00_PROJECT_BRAIN/ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md",
        "00_PROJECT_BRAIN/ACTIVE_SPEC/02_EXECUTION_SPINE.md",
        "00_PROJECT_BRAIN/ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md",
        "00_PROJECT_BRAIN/ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md",
        "00_PROJECT_BRAIN/ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md",
        "00_PROJECT_BRAIN/ACTIVE_SPEC/06_BARE_STEEL_DOCTRINE.md",
        "00_PROJECT_BRAIN/ACTIVE_SPEC/07_WORKING_REALITY_LAW.md",
        "00_PROJECT_BRAIN/ACTIVE_SPEC/08_BOARD_EFFECT_TOURNAMENT_LAW.md",
    }
    for raw in expected:
        assert (ROOT / raw).exists(), raw


def test_no_minimum_doc_set_language_or_unowned_manual_drift():
    proc = _run_check()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = _receipt(proc.stdout)
    assert data["minimum_doc_language_hits"] == []
    assert data["unauthorized_manual_title_hits"] == []
