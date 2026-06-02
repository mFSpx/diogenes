from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dev_journey_decision_points_json_stdout_is_pure():
    proc = subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT / "scripts" / "dev_journey_decision_points.py"),
            "--source",
            "scripts/dev_journey_decision_points.py",
            "--max-points",
            "4",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["truth_status"] == "training_candidates_only"
    assert proc.stdout.lstrip().startswith("{")
    assert proc.stdout.rstrip().endswith("}")
    assert "DEV_JOURNEY_DECISION_POINTS=PASS" not in proc.stdout
    assert "RECEIPT_PATH=" not in proc.stdout
    assert proc.stderr == "" or "WARNING:" in proc.stderr
