from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_deterministic_purge_schema_installs_native_triggers_and_functions() -> None:
    schema = (ROOT / "06_SCHEMA" / "115_deterministic_bare_steel_purge.sql").read_text(encoding="utf-8")
    assert "fn_auto_alias_staging_packet" in schema
    assert "trg_pre_stage_hygiene" in schema
    assert "fn_high_entropy_noise" in schema
    assert "trg_component_high_entropy_quarantine" in schema
    assert "fn_exact_sha_duplicate_alias_candidates" in schema
    assert "fn_kleene_k3" in schema
    assert "ternary_valency_summary" in schema
