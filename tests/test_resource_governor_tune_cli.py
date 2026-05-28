import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tune_cli_updates_dials(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/resource_governor.py",
            "--root",
            str(tmp_path),
            "tune",
            "--mode",
            "AGGRESSIVE",
            "--max-cloud-workers",
            "80",
            "--max-db-connections",
            "18",
            "--target-api-latency-ms",
            "700",
        ],
        text=True,
        capture_output=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    dials = json.loads((tmp_path / "05_OUTPUTS/runtime/governor_dials.json").read_text())
    assert dials["GLOBAL_MODE"] == "AGGRESSIVE"
    assert dials["MAX_CLOUD_WORKERS"] == 80
    assert dials["MAX_DB_CONNECTIONS"] == 18
    assert dials["TARGET_API_LATENCY_MS"] == 700
