from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_luci_attempt_engine_json_stdout_is_pure():
    proc = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT / "scripts" / "luci_attempt_engine.py"),
            "--synthetic",
            "--text",
            "fix one small broken thing and prove it",
            "--run-id",
            "pytest-attempt-engine-json",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["schema"] == "lucidota.luci_attempt_engine.receipt.v1"
    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert "RECEIPT_PATH=" not in proc.stdout
    assert proc.stderr == "" or "WARNING:" in proc.stderr
