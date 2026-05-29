from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lucidota_security_quarantine_gate import scan  # type: ignore
from phase05_allowlisted_ingest import is_excluded_by_manifest  # type: ignore


def test_security_gate_excluded_sample_limit_is_configurable(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("API_KEY=abc123456789012345\n", encoding="utf-8")
    (tmp_path / "credentials.json").write_text('{"client_secret":"abc123456789012345"}', encoding="utf-8")
    report = scan([tmp_path], max_files=10, mode="brain_archaeology_ingest", excluded_sample_limit=10)
    assert report["clean_manifest"] is True
    assert report["included_findings_count"] == 0
    assert report["excluded_findings_count"] == 2
    assert len(report["excluded_findings_sample"]) == 2
    assert report["secret_values_printed"] is False


def test_phase05_respects_manifest_exclusion_patterns() -> None:
    assert is_excluded_by_manifest(ROOT / ".env", ["*.env", "**/.env"])
    assert is_excluded_by_manifest(ROOT / "subdir" / "credentials.json", ["**/*credential*.json"])
    assert not is_excluded_by_manifest(ROOT / "safe_notes.txt", ["*.env", "**/*credential*.json"])
