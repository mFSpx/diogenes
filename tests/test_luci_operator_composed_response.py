from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_luci_operate_json_includes_multi_lane_composition():
    proc = subprocess.run(
        [str(ROOT / "luci"), "operate", "--text", "make my system work and get my shit ingested", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    visible = payload["visible_response"]
    assert visible["summary"].startswith("Indy_READs:")
    assert "map:" in visible["summary"].lower()
    assert "math:" in visible["summary"].lower()
    assert "quote:" in visible["summary"].lower()
    assert "improve:" in visible["summary"].lower()
    assert "review:" in visible["summary"].lower()
    assert visible["segments"][0]["lane"] == "fast"
    assert visible["segments"][-1]["lane"] == "review"
    assert proc.stderr == "" or "WARNING:" in proc.stderr


def test_luci_operate_human_output_shows_composed_summary():
    proc = subprocess.run(
        [str(ROOT / "luci"), "operate", "--text", "make my system work and get my shit ingested"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Indy_READs:" in proc.stdout
    assert "map:" in proc.stdout.lower()
    assert "math:" in proc.stdout.lower()
    assert "quote:" in proc.stdout.lower()
    assert "improve:" in proc.stdout.lower()
    assert "review:" in proc.stdout.lower()
    assert "NEXT=" in proc.stdout
