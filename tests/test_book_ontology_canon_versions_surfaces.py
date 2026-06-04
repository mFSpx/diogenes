#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_book_registry_family_and_canon_versions_are_live() -> None:
    book = subprocess.run([str(ROOT / "luci"), "book", "source", "--json"], cwd=ROOT, text=True, capture_output=True, check=True)
    book_payload = json.loads(book.stdout)
    assert isinstance(book_payload, list)

    canon = subprocess.run([str(ROOT / "luci"), "canon", "versions", "--json"], cwd=ROOT, text=True, capture_output=True, check=True)
    canon_payload = json.loads(canon.stdout)
    assert isinstance(canon_payload, list) and canon_payload
    assert "node_id" in canon_payload[0] or "canon_id" in canon_payload[0] or "version_id" in canon_payload[0]
