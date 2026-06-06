from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_luci_percyphon_current_json_reads_live_surface() -> None:
    proc = subprocess.run(
        [str(ROOT / "luci"), "percyphon", "current", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    row = payload[0]
    assert row["authority"] == "procedural_scaffold_candidate_not_truth"
    assert row["slots"]
    assert "slot_001" in row
    assert "slot_128" in row


def test_luci_percyphon_matrix_json_reads_live_surface() -> None:
    proc = subprocess.run(
        [str(ROOT / "luci"), "percyphon", "matrix", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert payload
    assert payload[0]["authority"] == "procedural_scaffold_candidate_not_truth"
    assert "packet" in payload[0]
    assert payload[0]["packet"]["slot_count"] == 129


def test_luci_percyphon_emit_json_writes_live_surface() -> None:
    seed = "percyphon-test-seed-20260604"
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "percyphon",
            "emit",
            "--seed",
            seed,
            "--villager",
            "operator",
            "--villager",
            "scribe",
            "--fluid-slots",
            "100",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["authority"] == "procedural_scaffold_candidate_not_truth"
    assert payload["seed"] == "operator|scribe"
    assert payload["source"] == "Runtime"

    current = subprocess.run(
        [str(ROOT / "luci"), "percyphon", "current", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    rows = json.loads(current.stdout)
    assert any(row.get("seed") == "operator|scribe" for row in rows)
