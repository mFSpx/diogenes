from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_exact_sha_bulk_dedupe_alias_lives_in_postgres() -> None:
    schema = (ROOT / "06_SCHEMA" / "118_exact_sha_bulk_dedupe_alias.sql").read_text(encoding="utf-8")
    assert "fn_exact_sha_bulk_alias_materialize" in schema
    assert "lucidota_go.exact_sha_bulk_alias_batch" in schema
    assert "DEDUP_ONLY_NOT_FOR_LLM_EXTRACTIONS" in schema
    assert "postgres_exact_sha_bulk_alias_ledgered.v1" in schema
    assert "graph_promotion_packet" in schema
    assert "graph_promotion_materialization" in schema
    assert "graph_materialization_helper_receipt" in schema
    assert "enforce_graph_promotion_path" not in schema


def test_exact_sha_bulk_dedupe_alias_has_post_batch_integrity_checks() -> None:
    schema = (ROOT / "06_SCHEMA" / "118_exact_sha_bulk_dedupe_alias.sql").read_text(encoding="utf-8")
    assert "batch_bad_policy" in schema
    assert "batch_bad_guard" in schema
    assert "remaining_exact_sha_alias_targets" in schema
    assert "inserted_distinct_sources" in schema
