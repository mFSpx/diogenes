from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_api_workflow_registry_raw_shell_alias_is_live() -> None:
    proc = subprocess.run([str(ROOT / "luci"), "api", "workflow", "registry", "raw", "--json"], cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["source_url"].endswith("/api_workflow_registry?order=updated_at.desc&limit=50")
    assert payload["payload"]
