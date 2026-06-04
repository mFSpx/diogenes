#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_provider_and_capability_registry_shell_aliases_are_live() -> None:
    provider = subprocess.run(
        [str(ROOT / "luci"), "provider", "registry", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    capability = subprocess.run(
        [str(ROOT / "luci"), "capability", "registry", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    provider_payload = json.loads(provider.stdout)
    capability_payload = json.loads(capability.stdout)
    assert isinstance(provider_payload, list) and provider_payload
    assert isinstance(capability_payload, list) and capability_payload
    assert "provider_key" in provider_payload[0]
    assert "capability_key" in capability_payload[0]
