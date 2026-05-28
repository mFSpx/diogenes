#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from script_bucket_manifest import BUCKETS, build_manifest  # noqa: E402


def test_script_bucket_manifest_is_non_destructive_and_uses_required_buckets() -> None:
    payload = build_manifest(ROOT / "scripts")
    assert payload["schema"] == "lucidota.script_bucket_manifest.v1"
    assert payload["non_destructive"] is True
    assert set(payload["counts"]) == set(BUCKETS)
    assert sum(payload["counts"].values()) == len(payload["items"])


def test_current_script_with_legacy_counterpart_stays_active_not_merge_duplicate() -> None:
    payload = build_manifest(ROOT / "scripts")
    by_path = {item["path"]: item for item in payload["items"]}
    assert by_path["scripts/boring_beast.py"]["bucket"] == "KEEP_ACTIVE"
    assert by_path["scripts/legacy/boring_beast.py"]["bucket"] == "QUARANTINE_LEGACY"
    assert any(
        "legacy_duplicate_counterpart_quarantined" in reason
        for reason in by_path["scripts/boring_beast.py"]["reasons"]
    )
