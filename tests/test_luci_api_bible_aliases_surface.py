from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_luci_api_bible_manuals_edges_nodes_and_subtree_aliases_are_live() -> None:
    for args, expected_key in [
        (["api", "bible", "manuals", "--json"], "manual_id"),
        (["api", "bible", "route", "catalog", "--json"], "route_id"),
        (["api", "bible", "edges", "--json"], "edge_id"),
        (["api", "bible", "nodes", "--manual-id", "RUNTIME_GOVERNOR", "--json"], "node_id"),
        (["api", "bible", "subtree", "--root-id", "1.0.0", "--json"], "node_id"),
    ]:
        proc = subprocess.run([str(ROOT / "luci"), *args], cwd=ROOT, text=True, capture_output=True, check=True)
        payload = json.loads(proc.stdout)
        assert payload["source_url"].startswith("http://127.0.0.1:3000/api_bible_")
        assert payload["payload"], (args, payload)
        first = payload["payload"][0]
        assert expected_key in first, (args, first)


def test_luci_api_bible_help_mentions_alias_namespace() -> None:
    proc = subprocess.run([str(ROOT / "luci"), "--help"], cwd=ROOT, text=True, capture_output=True, check=True)
    assert "luci api bible manuals [--json] [--base-url URL]" in proc.stdout
    assert "luci api bible route catalog [--json] [--base-url URL]" in proc.stdout
    assert "luci api bible edges [--json] [--base-url URL]" in proc.stdout
    assert "luci api bible nodes --manual-id ID [--json] [--base-url URL]" in proc.stdout
    assert "luci api bible subtree --root-id ID [--json] [--base-url URL]" in proc.stdout
