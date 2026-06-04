#!/usr/bin/env python3
"""Build deferred graph-promotion packets from GO/GCI/O414 route packets.

This is the safe bridge after sheet/RunPod ingest and before any canonical
graph mutation. It emits small evidence-backed packet JSONL suitable for
`graph_promotion_gate.py gate --dry-run` / packet staging, while preserving the
hard rule that raw bodies stay in CAS/body refs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

SCHEMA = "lucidota.graph.deferred_promotion_packet.v1"
RECEIPT_SCHEMA = "lucidota.graph.deferred_promotion_packet.receipt.v1"


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def iter_jsonl(path: Path):
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{line_no}: row must be object")
        yield obj


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _base_group(row: dict[str, Any]) -> dict[str, Any]:
    chunk_ref = str(row.get("chunk_ref") or row.get("id") or "")
    input_hash = str(row.get("input_hash") or "")
    route = str(row.get("route") or "CHECK")
    reason = str(row.get("route_reason") or "")
    return {
        "chunk_ref": chunk_ref,
        "input_hash": input_hash,
        "route": route,
        "route_reason": reason,
        "body_path": str(row.get("body_path") or ""),
        "ontologies": [],
        "terms": [],
        "evidence_refs": [],
        "requires_absurd": False,
    }


def packet_from_group(group: dict[str, Any], extra_evidence_refs: list[str]) -> dict[str, Any]:
    evidence_refs = list(group["evidence_refs"])
    for ref in extra_evidence_refs:
        _append_unique(evidence_refs, ref)
    candidate_payload = {
        "term": group["chunk_ref"],
        "label": "deferred_go_packet_chunk",
        "chunk_ref": group["chunk_ref"],
        "input_hash": group["input_hash"],
        "route": group["route"],
        "route_reason": group["route_reason"],
        "ontologies": sorted(group["ontologies"]),
        "terms": group["terms"][:24],
        "requires_absurd": bool(group["requires_absurd"]),
    }
    return {
        "schema": SCHEMA,
        "created_at": now_z(),
        "candidate_kind": "node",
        "authority_class": "operator_authored_assertion",
        "decision": "defer",
        "rationale": "Deferred graph packet from sheet-first RunPod/GO route bridge; canonical materialization remains gated.",
        "chunk_ref": group["chunk_ref"],
        "input_hash": group["input_hash"],
        "body_path": group["body_path"],
        "evidence_refs": evidence_refs,
        "candidate_payload": candidate_payload,
        "canonical_graph_writes_performed": False,
        "db_writes_performed": False,
        "body_policy": "refs_not_bodies",
        "packet_hash": sha_obj(candidate_payload),
    }


def build_deferred_packets(
    *,
    go_packets_path: Path,
    output_path: Path,
    receipt_path: Path,
    evidence_refs: list[str] | None = None,
    max_chunks: int = 0,
) -> dict[str, Any]:
    evidence_refs = list(evidence_refs or [])
    groups: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    rows_seen = 0
    for row in iter_jsonl(go_packets_path):
        rows_seen += 1
        chunk_ref = str(row.get("chunk_ref") or row.get("id") or "")
        if not chunk_ref:
            continue
        if chunk_ref not in groups:
            if max_chunks > 0 and len(groups) >= max_chunks:
                continue
            groups[chunk_ref] = _base_group(row)
        group = groups[chunk_ref]
        _append_unique(group["ontologies"], str(row.get("ontology") or ""))
        for term in row.get("terms") or []:
            _append_unique(group["terms"], str(term))
        body_path = str(row.get("body_path") or "")
        _append_unique(group["evidence_refs"], body_path)
        packet = row.get("packet") if isinstance(row.get("packet"), dict) else {}
        _append_unique(group["evidence_refs"], str(packet.get("evidence_ref") or ""))
        if packet.get("requires_absurd") or packet.get("contradiction_gate"):
            group["requires_absurd"] = True

    packets = [packet_from_group(group, evidence_refs) for group in groups.values()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in packets), encoding="utf-8")
    out_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS" if packets else "FAIL",
        "created_at": now_z(),
        "go_packets_path": str(go_packets_path),
        "go_rows_seen": rows_seen,
        "chunks_seen": len(groups),
        "packets_written": len(packets),
        "output_path": str(output_path),
        "output_sha256": out_hash,
        "canonical_graph_writes_performed": False,
        "db_writes_performed": False,
        "body_policy": "refs_not_bodies",
        "evidence_refs": evidence_refs,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="graph-deferred-promotion-packets")
    p.add_argument("--go-packets", type=Path, default=Path("05_OUTPUTS/runtime/ingest_route_go_packets.jsonl"))
    p.add_argument("--output", type=Path, default=Path("05_OUTPUTS/graph/deferred_promotion_packets.jsonl"))
    p.add_argument("--receipt", type=Path, default=Path("05_OUTPUTS/graph/deferred_promotion_packets_receipt.json"))
    p.add_argument("--evidence-ref", action="append", default=[])
    p.add_argument("--max-chunks", type=int, default=0)
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_deferred_packets(
        go_packets_path=args.go_packets,
        output_path=args.output,
        receipt_path=args.receipt,
        evidence_refs=args.evidence_ref,
        max_chunks=args.max_chunks,
    )
    print(json.dumps(receipt, sort_keys=True) if args.json else json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
