import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_saturation_cli_emits_receipt_and_runs_async_loop(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/resource_governor.py",
            "--root",
            str(tmp_path),
            "test-saturation",
            "--duration-sec",
            "1",
            "--initial-workers",
            "20",
            "--max-workers",
            "50",
            "--rate-limit-every",
            "4",
            "--base-latency-ms",
            "20",
        ],
        text=True,
        capture_output=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    report_line = next(line for line in proc.stdout.splitlines() if line.startswith("REPORT_PATH="))
    receipt = json.loads((tmp_path / report_line.split("=", 1)[1]).read_text())
    assert receipt["schema"] == "lucidota.resource_governor.saturation_test.v1"
    assert receipt["report"]["launched"] > 0
    assert receipt["report"]["rate_limited"] > 0
    assert receipt["report"]["final_workers"] >= 1
