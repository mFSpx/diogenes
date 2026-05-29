from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def report_path(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            return ROOT / line.split("=", 1)[1]
    raise AssertionError(stdout)


def test_surface_instruction_compiler_language_not_button_telemetry() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/surface_instruction_compile_dry_run.py",
            "--surface-id",
            "pytest_surface_compiler",
            "--operator-action",
            "selected",
            "--target-ref",
            "design_atom:test",
            "--evidence-refs",
            "tests/test_surface_instruction_compiler.py",
            "--artifact-refs",
            "07_SURFACES/generated/marrow_loop_status.html",
            "--current-object-state",
            '{"state":"pytest"}',
            "--allowed-effect",
            "stage_conversation_instruction_only",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    data = json.loads(report_path(proc.stdout).read_text())
    envelope = data["command_envelope"]
    assert data["db_writes_performed"] is False
    assert data["graph_writes_performed"] is False
    assert data["canonical_mutation_allowed"] is False
    assert data["conversation_required"] is True
    assert envelope["primary_signal"] == "plain_language_operator_instruction"
    assert envelope["button_telemetry_primary"] is False
    assert "button_pressed" not in json.dumps(data).lower()
    assert "Tell LUCIDOTA" in data["plain_language_instruction"]
    assert "do not mutate canonical graph" in data["plain_language_instruction"]

if __name__ == "__main__":
    test_surface_instruction_compiler_language_not_button_telemetry()
    print("SURFACE_INSTRUCTION_COMPILER_TEST=PASS")
