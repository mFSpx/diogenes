from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_balanced_ternary_schema_adds_native_graph_valency_columns() -> None:
    schema = (ROOT / "06_SCHEMA" / "114_balanced_ternary_valency.sql").read_text(encoding="utf-8")
    assert "lucidota_go.graph_item" in schema
    assert "lucidota_go.graph_edge" in schema
    assert "ternary_valency integer NOT NULL DEFAULT 0" in schema
    assert "CHECK (ternary_valency IN (1, 0, -1))" in schema


def test_graph_materializer_persists_candidate_ternary_valency() -> None:
    source = (ROOT / "scripts" / "graph_promotion_materialize.py").read_text(encoding="utf-8")
    assert "06_SCHEMA/114_balanced_ternary_valency.sql" in source
    assert "def ternary_valency" in source
    assert "graph_payload[\"ternary_valency\"] = valency" in source
    assert "location_real_at_added, ternary_valency, payload" in source
