#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_luci_shell_wrapper_exposes_provider_class():
    proc = subprocess.run(
        [
            str(ROOT / "luci"),
            "provider",
            "groq-chat",
            "--prompt",
            "say hello in one short sentence",
            "--max-tokens",
            "16",
            "--execute",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
    assert payload["provider"] == "groq"
    assert payload["api_key_env_used"] == "GROQ_API_KEY"
