from __future__ import annotations

from pathlib import Path


def test_graph_materializers_handle_packet_dedupe_idempotently():
    promotion = Path("scripts/graph_promotion_materialize.py").read_text(encoding="utf-8")
    edge = Path("scripts/graph_edge_materialize.py").read_text(encoding="utf-8")

    assert "ON CONFLICT (packet_dedupe_key)" in promotion
    assert "ON CONFLICT (packet_dedupe_key)" in edge
    assert "operator_uuid" in edge
