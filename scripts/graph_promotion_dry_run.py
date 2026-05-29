#!/usr/bin/env python3
"""Graph promotion dry-run scaffold.

Writes a report only. Performs no DB writes and no graph mutation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "graph"
SCHEMA = ROOT / "06_SCHEMA" / "034_graph_promotion_pipeline.sql"
GRAPH_CORE = ROOT / "06_SCHEMA" / "016_go_graph_core.sql"

REQUIRED_SQL_TOKENS = [
    "graph_promotion_packet",
    "graph_promotion_decision",
    "graph_promotion_journal_requirement",
    "evidence_refs",
    "authority_class",
]
CANONICAL_TABLES = ["graph_item", "graph_edge", "graph_journal", "staging_packet"]
FORBIDDEN_MUTATION = re.compile(r"\b(INSERT\s+INTO\s+lucidota_go\.graph_(item|edge)|UPDATE\s+lucidota_go\.graph_|DELETE\s+FROM\s+lucidota_go\.graph_)\b", re.I)


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_obj(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_candidate_packets(path_raw: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    if not path_raw:
        return [], []
    path = Path(path_raw)
    blockers: list[str] = []
    if not path.exists() or not path.is_file():
        return [], [f"candidate_file_missing:{path_raw}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [f"candidate_file_invalid_json:{type(exc).__name__}:{exc}"]
    packets = data.get("claim_packets", []) if isinstance(data, dict) else []
    if not isinstance(packets, list):
        return [], ["candidate_file_claim_packets_not_array"]
    candidates: list[dict[str, Any]] = []
    for idx, packet in enumerate(packets, start=1):
        evidence_refs = packet.get("evidence_refs", []) if isinstance(packet, dict) else []
        if not evidence_refs:
            blockers.append(f"claim_packet_{idx}_missing_evidence_refs")
        candidates.append({
            "source_system": "claim_packet_dry_run",
            "candidate_kind": "node",
            "candidate_payload": {
                "claim_packet_id": packet.get("claim_packet_id"),
                "claim_text": packet.get("claim_text"),
                "label": packet.get("label"),
                "matched_text_sha256": hashlib.sha256(str(packet.get("matched_text", "")).encode()).hexdigest(),
                "review_state": packet.get("review_state"),
                "graph_promotion_status": "candidate_dry_run_only",
            },
            "evidence_refs": evidence_refs,
            "authority_class": packet.get("authority_class", "model_computed_finding"),
            "source_report_path": str(path),
        })
    if not candidates:
        blockers.append("candidate_file_no_claim_packets")
    return candidates, blockers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--candidate-kind", default="doctrine")
    ap.add_argument("--authority-class", default="operator_authored_assertion")
    ap.add_argument("--candidate-file", help="Optional JSON claim/packet/report file to convert into dry-run promotion candidates. No DB or graph writes.")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    blockers = []
    schema_text = SCHEMA.read_text(encoding="utf-8") if SCHEMA.exists() else ""
    graph_text = GRAPH_CORE.read_text(encoding="utf-8") if GRAPH_CORE.exists() else ""
    if not SCHEMA.exists(): blockers.append("graph_promotion_schema_missing")
    missing_tokens = [t for t in REQUIRED_SQL_TOKENS if t not in schema_text]
    if missing_tokens: blockers.append("graph_promotion_schema_missing_tokens:" + ",".join(missing_tokens))
    confirmed_tables = [t for t in CANONICAL_TABLES if re.search(rf"CREATE\s+TABLE\s+(IF\s+NOT\s+EXISTS\s+)?lucidota_go\.{t}\b", graph_text, re.I)]
    if "graph_journal" not in confirmed_tables:
        blockers.append("graph_journal_not_confirmed")
    forbidden = sorted(set(m.group(0) for m in FORBIDDEN_MUTATION.finditer(schema_text)))
    if forbidden:
        blockers.append("forbidden_direct_graph_mutation_in_schema")
    candidate_packets, candidate_blockers = load_candidate_packets(args.candidate_file)
    blockers.extend(candidate_blockers)
    sample_candidate = {
        "source_system": "graph_promotion_dry_run",
        "candidate_kind": args.candidate_kind,
        "candidate_payload": {"claim": "No direct graph mutation; use promotion packets."},
        "evidence_refs": ["00_PROJECT_BRAIN/GRAPH_PROMOTION_PIPELINE.md", "06_SCHEMA/034_graph_promotion_pipeline.sql"],
        "authority_class": args.authority_class,
    }
    dry_run_candidates = candidate_packets or [sample_candidate]
    decision = {
        "decision": "defer",
        "decided_by": "graph_promotion_dry_run",
        "rationale": "Dry-run report only; no graph mutation allowed in scaffold phase.",
        "evidence_refs": sample_candidate["evidence_refs"],
        "operator_confirmed": False,
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "dry_run",
        "schema_path": str(SCHEMA.relative_to(ROOT)),
        "confirmed_graph_tables": confirmed_tables,
        "sample_candidate": sample_candidate,
        "sample_candidate_sha256": sha256_obj(sample_candidate),
        "candidate_file": args.candidate_file,
        "dry_run_candidates": dry_run_candidates,
        "dry_run_candidate_count": len(dry_run_candidates),
        "dry_run_candidate_sha256s": [sha256_obj(candidate) for candidate in dry_run_candidates],
        "sample_decision": decision,
        "graph_writes_performed": False,
        "db_writes_performed": False,
        "forbidden_direct_graph_mutation_matches": forbidden,
        "blockers": blockers,
    }
    out = out_dir / f"graph_promotion_dry_run_{ts()}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    print(f"REPORT_PATH={out.relative_to(ROOT)}")
    return 0 if not blockers else 1

if __name__ == "__main__":
    raise SystemExit(main())
