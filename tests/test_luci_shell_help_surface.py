#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_luci_shell_help_mentions_unified_rails():
    proc = subprocess.run(
        [str(ROOT / "luci"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "luci operator-route --raw-command" in proc.stdout
    assert "luci provider groq-chat --prompt" in proc.stdout
    assert "luci model admission [--run-diogenes-gate]" in proc.stdout
    assert "luci learn --text TEXT [--run-id ID] [--artifact PATH] [--candidate-kind KIND] [--json]" in proc.stdout
