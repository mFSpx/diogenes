from __future__ import annotations

import json
from pathlib import Path
from importlib import import_module

mod = import_module("scripts.graph_deferred_promotion_packets")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_build_deferred_packets_groups_ontologies_without_raw_bodies(tmp_path: Path):
    go = tmp_path / "go.jsonl"
    body = tmp_path / "body.json"
    body.write_text('{"text":"raw body must stay referenced"}', encoding="utf-8")
    rows = [
        {
            "schema": "lucidota.go_packet.v1",
            "ontology": "GO25",
            "chunk_ref": "book.c0001",
            "input_hash": "abc",
            "body_path": str(body),
            "route": "CHECK",
            "route_reason": "treelite_lane_slow",
            "terms": ["ALPHA", "BETA"],
            "packet": {"claims": ["ALPHA"], "evidence_ref": str(body)},
        },
        {
            "schema": "lucidota.go_packet.v1",
            "ontology": "GCI_O_75",
            "chunk_ref": "book.c0001",
            "input_hash": "abc",
            "body_path": str(body),
            "route": "CHECK",
            "route_reason": "treelite_lane_slow",
            "terms": ["ALPHA"],
            "packet": {"contradiction_gate": True},
        },
        {
            "schema": "lucidota.go_packet.v1",
            "ontology": "O414",
            "chunk_ref": "book.c0001",
            "input_hash": "abc",
            "body_path": str(body),
            "route": "CHECK",
            "route_reason": "treelite_lane_slow",
            "terms": ["ALPHA"],
            "packet": {"requires_absurd": True},
        },
    ]
    _write_jsonl(go, rows)
    embed_receipt = tmp_path / "embed_receipt.json"
    embed_receipt.write_text(json.dumps({"status": "PASS", "row_count": 1}), encoding="utf-8")
    out = tmp_path / "packets.jsonl"
    receipt = tmp_path / "receipt.json"

    result = mod.build_deferred_packets(
        go_packets_path=go,
        output_path=out,
        receipt_path=receipt,
        evidence_refs=[str(embed_receipt)],
        max_chunks=25,
    )

    assert result["status"] == "PASS"
    assert result["chunks_seen"] == 1
    assert result["packets_written"] == 1
    packet = json.loads(out.read_text(encoding="utf-8").strip())
    assert packet["schema"] == "lucidota.graph.deferred_promotion_packet.v1"
    assert packet["chunk_ref"] == "book.c0001"
    assert packet["canonical_graph_writes_performed"] is False
    assert packet["candidate_kind"] == "node"
    assert packet["candidate_payload"]["ontologies"] == ["GCI_O_75", "GO25", "O414"]
    assert str(body) in packet["evidence_refs"]
    assert str(embed_receipt) in packet["evidence_refs"]
    serialized = json.dumps(packet)
    assert "raw body must stay referenced" not in serialized


def test_cli_writes_receipt_and_limits_chunks(tmp_path: Path):
    go = tmp_path / "go.jsonl"
    _write_jsonl(
        go,
        [
            {"ontology": "GO25", "chunk_ref": "a", "input_hash": "1", "body_path": "body-a", "route": "CHECK", "route_reason": "x", "terms": []},
            {"ontology": "GO25", "chunk_ref": "b", "input_hash": "2", "body_path": "body-b", "route": "CHECK", "route_reason": "x", "terms": []},
        ],
    )
    out = tmp_path / "out.jsonl"
    receipt = tmp_path / "receipt.json"
    code = mod.main([
        "--go-packets", str(go),
        "--output", str(out),
        "--receipt", str(receipt),
        "--max-chunks", "1",
        "--json",
    ])
    assert code == 0
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["packets_written"] == 1
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1
