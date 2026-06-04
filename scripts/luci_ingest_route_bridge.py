#!/usr/bin/env python3
"""Bounded ingest -> sheet -> Treelite/algo -> Tokio pubsub/ABSURD bridge.

This is the refs-only proof seam for live ingest: read bounded chunk refs, let the
sheet layer be first authority, route cheap with Treelite/algo scores, emit GO
ontology packets, emit Tokio pubsub EventRefs, and send only CHECK/PANIC/DEEP to
ABSURD slowlane. Bodies are written to CAS-ish body files and never embedded in
events.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.edge_grail_treelite_router import route_edge_packet

ONTOLOGIES = ("GO25", "GCI_O_75", "O414")
BODY_KEYS = {"text", "body", "raw", "raw_text", "prompt", "response"}


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path, limit: int) -> Iterable[dict[str, Any]]:
    seen = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        yield json.loads(line)
        seen += 1
        if limit > 0 and seen >= limit:
            return


def keywords(text: str, max_terms: int = 12) -> list[str]:
    stop = {"the", "and", "that", "with", "from", "this", "should", "because", "after", "before", "operator"}
    out: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", text.lower()):
        if token in stop or token in out:
            continue
        out.append(token.upper())
        if len(out) >= max_terms:
            break
    return out


def sheet_first_decision(sheet_manifest: dict[str, Any]) -> dict[str, Any]:
    sheets = {s.get("id"): s for s in sheet_manifest.get("sheets", [])}
    sheet = sheets.get("next_work_batch") or {}
    query = str(sheet.get("query") or "")
    return {
        "status": "PASS" if sheet else "FAIL",
        "sheet": "next_work_batch",
        "database_object": sheet.get("database_object", ""),
        "query_hash": sha256(query.encode()).hexdigest(),
        "principle": "sheet_layer_before_algos_before_models",
        "next_action": "PROBE",
    }


def write_body(body_dir: Path, chunk: dict[str, Any]) -> Path:
    body_dir.mkdir(parents=True, exist_ok=True)
    chunk_ref = str(chunk.get("chunk_ref") or chunk.get("id") or "chunk")
    text = str(chunk.get("text") or chunk.get("content") or "")
    digest = str(chunk.get("text_sha256") or sha256(text.encode()).hexdigest())
    path = body_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', chunk_ref)[:120]}_{digest[:12]}.json"
    path.write_text(
        json.dumps(
            {
                "schema": "lucidota.ingest_route_bridge.body_ref.v1",
                "chunk_ref": chunk_ref,
                "text_sha256": digest,
                "text": text,
                "source": chunk.get("book_id") or chunk.get("source_path") or "",
                "created_at": now_z(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def build_go_packets(chunk: dict[str, Any], body_path: Path, route: dict[str, Any], sheet: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(chunk.get("text") or chunk.get("content") or "")
    base = {
        "chunk_ref": str(chunk.get("chunk_ref") or chunk.get("id") or ""),
        "input_hash": str(chunk.get("text_sha256") or sha256(text.encode()).hexdigest()),
        "body_path": str(body_path),
        "route": route["route"],
        "route_reason": route["reason"],
        "sheet_action": sheet["next_action"],
        "terms": keywords(text),
        "created_at": now_z(),
    }
    return [
        {
            "schema": "lucidota.go_packet.v1",
            "ontology": "GO25",
            **base,
            "packet": {
                "claims": base["terms"][:5],
                "evidence_ref": str(body_path),
                "uncertainty": "chunk_scoped",
            },
        },
        {
            "schema": "lucidota.go_packet.v1",
            "ontology": "GCI_O_75",
            **base,
            "packet": {
                "control_lane": "FASTLANE" if route["route"] == "FAST" else "SLOWLANE",
                "interposer": "Treelite+bounded_algos",
                "contradiction_gate": route["route"] in {"CHECK", "PANIC", "DEEP"},
            },
        },
        {
            "schema": "lucidota.go_packet.v1",
            "ontology": "O414",
            **base,
            "packet": {
                "deep_reading_lens": "Root414/Indy_READs novel-like ontology lane",
                "manual_bridge": True,
                "requires_absurd": route["route"] in {"CHECK", "PANIC", "DEEP"},
            },
        },
    ]


def strip_bodies(obj: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if k not in BODY_KEYS}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def run_bridge(chunks_path: Path, receipt_path: Path, body_dir: Path, max_chunks: int) -> dict[str, Any]:
    sheet_manifest = load_json(ROOT / "04_RUNTIME" / "lucidota_sheet_manifest.json")
    routing_manifest = load_json(ROOT / "04_RUNTIME" / "lucidota_integrated_routing_manifest.json")
    sheet = sheet_first_decision(sheet_manifest)

    packets: list[dict[str, Any]] = []
    pubsub_events: list[dict[str, Any]] = []
    absurd_events: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []

    for idx, chunk in enumerate(iter_jsonl(chunks_path, max_chunks), 1):
        body_path = write_body(body_dir, chunk)
        scores = chunk.get("algo_scores") if isinstance(chunk.get("algo_scores"), dict) else {}
        packet = {
            "event_id": f"ingest-route-{idx}",
            "input_hash": str(chunk.get("text_sha256") or ""),
            "body_path": str(body_path),
            "source_type": "book_chunk",
            "algo_scores": scores,
            "danger_flag": bool(scores.get("danger_flag")),
        }
        route = route_edge_packet(packet, artifact_path=str(body_path))
        route_rows.append(route)
        chunk_packets = build_go_packets(chunk, body_path, route, sheet)
        packets.extend(chunk_packets)

        event = {
            "schema": "lucidota.tokio_pubsub.event_ref.v1",
            "executor": "tokio_bounded_pubsub",
            "topic": "go_packet",
            "event_id": packet["event_id"],
            "route": route["route"],
            "body_path": str(body_path),
            "packet_refs": [p["ontology"] for p in chunk_packets],
            "content_hash": packet["input_hash"],
            "preview": str(chunk.get("chunk_ref") or "")[:256],
            "created_at": now_z(),
        }
        pubsub_events.append(event)
        needs_absurd = route["route"] in {"PANIC", "DEEP"} or bool(scores.get("danger_flag")) or int(scores.get("contradiction_count", 0) or 0) > 0
        if needs_absurd:
            absurd_events.append(
                {
                    "schema": "lucidota.absurd.slowlane_event_ref.v1",
                    "executor": "ABSURD_slowlane_receipt_gate",
                    "event_id": packet["event_id"],
                    "route": route["route"],
                    "reason": route["reason"],
                    "body_path": str(body_path),
                    "go_packet_ontologies": [p["ontology"] for p in chunk_packets],
                    "content_hash": packet["input_hash"],
                    "created_at": now_z(),
                }
            )

    out_dir = receipt_path.parent
    go_path = out_dir / "ingest_route_go_packets.jsonl"
    pubsub_path = out_dir / "ingest_route_tokio_pubsub.jsonl"
    absurd_path = out_dir / "ingest_route_absurd_slowlane.jsonl"
    route_path = out_dir / "ingest_route_treelite_routes.jsonl"
    write_jsonl(go_path, packets)
    write_jsonl(pubsub_path, pubsub_events)
    write_jsonl(absurd_path, absurd_events)
    write_jsonl(route_path, route_rows)

    receipt = {
        "schema": "lucidota.ingest_route_bridge.receipt.v1",
        "status": "PASS" if pubsub_events and packets and sheet["status"] == "PASS" else "FAIL",
        "created_at": now_z(),
        "chunks_path": str(chunks_path),
        "chunks_processed": len(pubsub_events),
        "sheet_first": sheet,
        "routing_manifest": strip_bodies(routing_manifest),
        "go_packet_jsonl": str(go_path),
        "tokio_pubsub_jsonl": str(pubsub_path),
        "absurd_slowlane_jsonl": str(absurd_path),
        "treelite_routes_jsonl": str(route_path),
        "packets_written": len(pubsub_events),
        "go_packet_rows_written": len(packets),
        "pubsub_events_written": len(pubsub_events),
        "absurd_events_written": len(absurd_events),
        "body_policy": "refs_not_bodies",
    }
    receipt["receipt_hash"] = sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(prog="luci-ingest-route-bridge")
    ap.add_argument("--chunks", default="04_RUNTIME/BOOK_READER_LORA/chunks/chunks_500tok.jsonl")
    ap.add_argument("--receipt", default="05_OUTPUTS/runtime/ingest_route_bridge_latest.json")
    ap.add_argument("--body-dir", default="04_RUNTIME/ingest_route_bridge/bodies")
    ap.add_argument("--max-chunks", type=int, default=25)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    receipt = run_bridge(Path(args.chunks), Path(args.receipt), Path(args.body_dir), args.max_chunks)
    print(json.dumps(receipt, sort_keys=True) if args.json else json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
