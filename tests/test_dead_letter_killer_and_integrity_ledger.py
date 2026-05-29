from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dead_letter_killer_schema_adds_native_purge_trigger() -> None:
    schema = (ROOT / "06_SCHEMA" / "119_dead_letter_killer_and_integrity_ledger.sql").read_text(encoding="utf-8")
    assert "fn_dead_letter_killer_purge" in schema
    assert "BEFORE INSERT ON lucidota_control.absurd_queue_dead_letter" in schema
    assert "NEW.resolved := true" in schema
    assert "NEW.last_seen_at := now()" in schema
    assert "JSON_MALFORMAT" in schema
    assert "helper_01_rc_2" in schema
    assert "NEW.context := COALESCE(NEW.context, '{}'::jsonb)" in schema


def test_system_integrity_ledger_view_counts_and_verdicts() -> None:
    schema = (ROOT / "06_SCHEMA" / "119_dead_letter_killer_and_integrity_ledger.sql").read_text(encoding="utf-8")
    assert "CREATE OR REPLACE VIEW lucidota_go.v_system_integrity_ledger" in schema
    assert "CREATE EXTENSION IF NOT EXISTS postgres_fdw" in schema
    assert "CREATE SERVER IF NOT EXISTS lucidota_storage_fdw" in schema
    assert "IMPORT FOREIGN SCHEMA lucidota_go" in schema
    assert "total_canonical_items" in schema
    assert "total_journal_stages" in schema
    assert "unconsumed_loss_signals" in schema
    assert "unresolved_dead_letters" in schema
    assert "PASS_COMPLETE" in schema
    assert "PASS_DEAD_LETTERS_PURGED" in schema
    assert "FAIL_ORPHAN_JOURNAL_ROWS" in schema
