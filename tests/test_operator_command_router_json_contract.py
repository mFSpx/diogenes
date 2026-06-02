from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_operator_command_router_json_stdout_is_pure(tmp_path):
    src = tmp_path / "drop"
    src.mkdir()
    (src / "note.md").write_text("Alice saw Evidence.")
    receipt_dir = tmp_path / "receipts"
    base_dir = tmp_path / "cases"

    proc = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT / "scripts" / "operator_command_router.py"),
            "--raw-command",
            "create case from folder and build packet",
            "--case-id",
            "operator-case",
            "--source-folder",
            str(src),
            "--base-dir",
            str(base_dir),
            "--receipt-dir",
            str(receipt_dir),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASSED"
    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert "OPERATOR_ROUTE=" not in proc.stdout
    assert proc.stderr == "" or "WARNING:" in proc.stderr
