from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_provider_registry_raw_shell_alias_is_live() -> None:
    proc = subprocess.run([str(ROOT / "luci"), "provider", "registry", "raw", "--json"], cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["source_url"].endswith("/provider_registry?order=provider_key.asc&limit=50")
    assert payload["payload"]


def test_capability_registry_raw_shell_alias_is_live() -> None:
    proc = subprocess.run([str(ROOT / "luci"), "capability", "registry", "raw", "--json"], cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["source_url"].endswith("/capability_registry?order=updated_at.desc&limit=50")
    assert payload["payload"]
