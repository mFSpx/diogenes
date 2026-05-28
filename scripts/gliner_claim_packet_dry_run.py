#!/usr/bin/env python3
"""Bench-only GLiNER span -> ClaimPacket dry-run layer.

Takes existing GLiNER/proof-hoard extraction behavior and emits claim candidates
with evidence refs. These are not graph truth and are not written to canonical
graph tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "05_OUTPUTS" / "tech_bench"
sys.path.insert(0, str(ROOT / "ALGOS"))
from gliner_zero_shot_extractor import extract, parse_labels  # type: ignore  # proof-hoard adapter copy boundary


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def confidence_bps(score: float) -> int:
    return max(0, min(10000, int(round(float(score) * 10000))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--text", default="Operator routes KORPUS through Chrono-Ledger. Command Envelope Protocol preserves instruction provenance.")
    ap.add_argument("--artifact-uuid", default="bench-artifact-uuid")
    ap.add_argument("--component-uuid", default="bench-component-uuid")
    ap.add_argument("--labels", default=str(ROOT / "05_OUTPUTS/contracts/operator_ontology_labels.json"))
    ap.add_argument("--extractor-version", default="bench.v1")
    ap.add_argument("--model-hash", default="missing_local_gliner_model")
    ap.add_argument("--authority-class", default="model_computed_finding")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    labels = parse_labels(args.labels)
    extraction = extract(args.text, labels, no_fallback=False)
    packets = []
    for idx, span in enumerate(extraction.get("spans", []), start=1):
        packet = {
            "claim_packet_id": f"claim_packet_bench_{idx:04d}_{sha256_text(json.dumps(span, sort_keys=True))[:12]}",
            "artifact_uuid": args.artifact_uuid,
            "component_uuid": args.component_uuid,
            "source_span_id": f"{args.component_uuid}:{span['start']}:{span['end']}",
            "evidence_refs": [{
                "artifact_uuid": args.artifact_uuid,
                "component_uuid": args.component_uuid,
                "start_char": span["start"],
                "end_char": span["end"],
                "matched_text_sha256": sha256_text(span["text"]),
            }],
            "claim_text": f"Detected {span['label']} mention: {span['text']}",
            "label": span["label"],
            "matched_text": span["text"],
            "extractor_name": "gliner_zero_shot_extractor",
            "extractor_version": args.extractor_version,
            "model_hash": args.model_hash,
            "confidence_bps": confidence_bps(span.get("score", 0)),
            "authority_class": args.authority_class,
            "review_state": "candidate_unreviewed",
            "graph_promotion_status": "not_promoted",
        }
        packets.append(packet)
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_ROOT / stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "lucidota.tech_bench.claim_packet_dry_run.v1",
        "generated_at": now_iso(),
        "mode": "dry_run",
        "text_sha256": sha256_text(args.text),
        "extractor_backend": extraction.get("backend"),
        "extractor_install_instruction": extraction.get("install_instruction"),
        "claim_packets": packets,
        "claim_packet_count": len(packets),
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "canonical_graph_mutation": False,
        "blockers": [] if packets else ["no_spans_detected_in_sample"],
    }
    out = out_dir / "claim_packet_dry_run_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    print(f"REPORT_PATH={out.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
