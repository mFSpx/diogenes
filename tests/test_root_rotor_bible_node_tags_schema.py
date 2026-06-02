from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "06_SCHEMA" / "146_root_rotor_bible_node_tags.sql"


def test_bible_node_tags_migration_adds_kind_tags_and_api_projection() -> None:
    schema = SCHEMA.read_text(encoding="utf-8")
    assert "ALTER TABLE lucidota_canon.bible_nodes" in schema
    assert "ADD COLUMN IF NOT EXISTS node_kind text" in schema
    assert "ADD COLUMN IF NOT EXISTS ontology_tags text[]" in schema
    assert "bible_nodes_kind_check" in schema
    assert "bible_nodes_ontology_tags_not_empty" in schema
    assert "'node_kind', node_row.node_kind" in schema
    assert "'ontology_tags', node_row.ontology_tags" in schema
    assert "CREATE TABLE IF NOT EXISTS lucidota_canon.api_route_catalog" in schema
    assert "CREATE OR REPLACE VIEW lucidota_canon.api_bible_route_catalog" in schema
    assert "node_kind" in schema
    assert "ontology_tags" in schema


def test_bible_node_tags_migration_backfills_from_payload_and_path() -> None:
    schema = SCHEMA.read_text(encoding="utf-8")
    assert "payload::jsonb->>'node_kind'" in schema
    assert "payload::jsonb->'ontology_tags'" in schema
    assert "source_refs->>0 LIKE '06_SCHEMA/%'" in schema
    assert "source_refs->>0 LIKE 'scripts/%'" in schema
