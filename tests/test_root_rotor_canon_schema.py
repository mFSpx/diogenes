from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "06_SCHEMA" / "144_canonical_technical_bible.sql"


def read_schema() -> str:
    return SCHEMA.read_text(encoding="utf-8")


def test_root_rotor_schema_defines_versioned_canon_tables() -> None:
    schema = read_schema()
    assert "CREATE EXTENSION IF NOT EXISTS pgcrypto" in schema
    assert "CREATE SCHEMA IF NOT EXISTS lucidota_canon" in schema
    assert "CREATE TABLE IF NOT EXISTS lucidota_canon.bible_nodes" in schema
    assert "CREATE TABLE IF NOT EXISTS lucidota_canon.bible_dependencies" in schema
    assert "CREATE TABLE IF NOT EXISTS lucidota_canon.bible_history" in schema
    assert "node_sort_key integer[] NOT NULL" in schema
    assert "manual_id text NOT NULL" in schema
    assert "node_kind text NOT NULL" in schema
    assert "ontology_tags text[] NOT NULL" in schema
    assert "status text NOT NULL" in schema
    assert "bible_nodes_kind_check" in schema
    assert "bible_nodes_ontology_tags_not_empty" in schema
    assert "CHECK (status IN ('verified', 'review_required', 'deprecated', 'draft'))" in schema
    assert "UNIQUE (node_id, version)" in schema


def test_root_rotor_schema_hashes_material_fields_and_snapshots_history() -> None:
    schema = read_schema()
    assert "lucidota_canon.fn_bible_node_material" in schema
    assert "jsonb_build_object" in schema
    for field in [
        "node_id",
        "parent_id",
        "manual_id",
        "node_kind",
        "title",
        "payload",
        "payload_format",
        "ontology_tags",
        "source_refs",
        "evidence_hashes",
        "dependencies",
        "affects_nodes",
        "status",
    ]:
        assert f"'{field}'" in schema
    assert "encode(digest(lucidota_canon.fn_bible_node_material(NEW)::text, 'sha256'), 'hex')" in schema
    assert "old_row jsonb NOT NULL" in schema
    assert "to_jsonb(OLD)" in schema
    assert "NEW.version := OLD.version + 1" in schema
    assert "NEW.previous_hash := OLD.hash_current" in schema


def test_root_rotor_schema_exposes_postgrest_safe_rpc_surface() -> None:
    schema = read_schema()
    assert "CREATE OR REPLACE FUNCTION lucidota_canon.get_subtree" in schema
    assert "WITH RECURSIVE manual_tree" in schema
    assert "ORDER BY node_sort_key" in schema
    assert "CREATE OR REPLACE VIEW lucidota_canon.api_bible_nodes" in schema
    assert "node_kind" in schema
    assert "ontology_tags" in schema
    assert "CREATE OR REPLACE VIEW lucidota_canon.api_bible_edges" in schema
    assert "CREATE OR REPLACE VIEW lucidota_canon.api_bible_manuals" in schema
    assert "fn_mark_bible_blast_radius" in schema
    assert "status = 'review_required'" in schema
