#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_claw_one_shot_returns_indy_reads_voice():
    proc = subprocess.run(
        [str(ROOT / "claw"), "say hi in one sentence"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Indy_READs:" in proc.stdout
    assert "here to help you with your software engineering tasks and questions" in proc.stdout


def test_claw_json_one_shot_returns_indy_reads_voice():
    proc = subprocess.run(
        [str(ROOT / "claw"), "--output-format=json", "say hi in one sentence"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["message"].startswith("Indy_READs:")
    assert "Indy_READs:" not in proc.stderr
