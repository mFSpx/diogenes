#!/usr/bin/env python3
"""Document parse span -> GLiNER-style claim packet bridge.

This is executable production-path plumbing, not graph truth:
- reads parsed document spans from lucidota_korpus.document_parse_* tables
- extracts exact matched text from parser Markdown output using stored char ranges
- runs the proof-hoard GLiNER instrument (or its literal fallback)
- writes candidate claim packets to lucidota_korpus.document_claim_packet only with --execute
- never writes canonical graph tables
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "06_SCHEMA/050_document_claim_packet_bridge.sql"
OUT = ROOT / "05_OUTPUTS/claim_packets"
sys.path.insert(0, str(ROOT / "ALGOS"))
from gliner_zero_shot_extractor import extract, parse_labels  # type: ignore


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def db_url(args: argparse.Namespace) -> str:
    return args.database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def confidence_bps(score: float) -> int:
    return max(0, min(10000, int(round(float(score) * 10000))))


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"document_claim_packet_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def load_markdown_text(output_markdown_path: str) -> str:
    p = ROOT / output_markdown_path
    if not p.exists():
        p = Path(output_markdown_path)
    return p.read_text(encoding="utf-8", errors="replace")


def fetch_spans(conn: psycopg.Connection[Any], limit: int, only_unclaimed: bool) -> list[dict[str, Any]]:
    predicate = ""
    if only_unclaimed:
        predicate = """
          AND NOT EXISTS (
            SELECT 1 FROM lucidota_korpus.document_claim_packet cp
            WHERE cp.span_uuid = s.span_uuid
              AND cp.extractor_name = 'gliner_zero_shot_extractor'
          )
        """
    sql = f"""
      SELECT
        r.run_uuid::text,
        r.source_path,
        r.source_sha256,
        r.output_markdown_path,
        r.parser_name,
        r.parser_version,
        s.span_uuid::text,
        s.page_number,
        s.block_index,
        s.start_char,
        s.end_char,
        s.span_sha256,
        s.label AS parser_span_label,
        s.provenance
      FROM lucidota_korpus.document_parse_span s
      JOIN lucidota_korpus.document_parse_run r ON r.run_uuid = s.run_uuid
      WHERE r.status = 'parsed'
        AND COALESCE(r.output_markdown_path, '') <> ''
        {predicate}
      ORDER BY r.created_at DESC, s.page_number, s.block_index
      LIMIT %s
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (limit,))
        return list(cur.fetchall())


def packet_from_span(row: dict[str, Any], span: dict[str, Any], extractor_version: str, model_hash: str, authority_class: str, extractor_backend: str) -> dict[str, Any]:
    source_span_id = f"{row['run_uuid']}:{row['span_uuid']}:{span['start']}:{span['end']}"
    evidence_refs = [{
        "run_uuid": row["run_uuid"],
        "span_uuid": row["span_uuid"],
        "source_path": row["source_path"],
        "source_sha256": row["source_sha256"],
        "parser": f"{row['parser_name']}:{row['parser_version']}",
        "page_number": row["page_number"],
        "block_index": row["block_index"],
        "parser_span_start_char": row["start_char"],
        "parser_span_end_char": row["end_char"],
        "extractor_span_start_char": span["start"],
        "extractor_span_end_char": span["end"],
        "matched_text_sha256": sha256_text(span["text"]),
    }]
    return {
        "run_uuid": row["run_uuid"],
        "span_uuid": row["span_uuid"],
        "source_span_id": source_span_id,
        "evidence_refs": evidence_refs,
        "claim_text": f"Detected {span['label']} mention: {span['text']}",
        "label": span["label"],
        "matched_text": span["text"],
        "matched_text_sha256": sha256_text(span["text"]),
        "extractor_name": "gliner_zero_shot_extractor",
        "extractor_version": extractor_version,
        "extractor_backend": extractor_backend,
        "model_hash": model_hash,
        "confidence_bps": confidence_bps(span.get("score", 0.0)),
        "authority_class": authority_class,
        "review_state": "candidate_unreviewed",
        "graph_promotion_status": "not_promoted",
        "truth_status": "not_truth_claim_candidate",
        "detail": {
            "script": "scripts/document_claim_packet_worker.py",
            "parser_span_label": row.get("parser_span_label"),
            "parser_provenance": row.get("provenance") or {},
        },
    }


def insert_packet(cur: psycopg.Cursor[Any], packet: dict[str, Any]) -> bool:
    cur.execute(
        """
        INSERT INTO lucidota_korpus.document_claim_packet(
          run_uuid, span_uuid, source_span_id, evidence_refs,
          claim_text, label, matched_text, matched_text_sha256,
          extractor_name, extractor_version, extractor_backend, model_hash,
          confidence_bps, authority_class, review_state, graph_promotion_status,
          truth_status, detail
        )
        VALUES (
          %s::uuid, %s::uuid, %s, %s::jsonb,
          %s, %s, %s, %s,
          %s, %s, %s, %s,
          %s, %s, %s, %s,
          %s, %s::jsonb
        )
        ON CONFLICT (source_span_id, extractor_name, extractor_version, label, matched_text_sha256)
        DO NOTHING
        RETURNING claim_packet_uuid::text
        """,
        (
            packet["run_uuid"], packet["span_uuid"], packet["source_span_id"], json.dumps(packet["evidence_refs"]),
            packet["claim_text"], packet["label"], packet["matched_text"], packet["matched_text_sha256"],
            packet["extractor_name"], packet["extractor_version"], packet["extractor_backend"], packet["model_hash"],
            packet["confidence_bps"], packet["authority_class"], packet["review_state"], packet["graph_promotion_status"],
            packet["truth_status"], json.dumps(packet["detail"]),
        ),
    )
    return cur.fetchone() is not None


def init_schema(args: argparse.Namespace) -> int:
    if not args.execute:
        write_report("init_schema_dry_run", {"action": "init_schema", "execute_performed": False, "schema": rel(SCHEMA)})
        return 0
    with psycopg.connect(db_url(args)) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA.read_text(encoding="utf-8"))
        conn.commit()
    write_report("init_schema_execute", {"action": "init_schema", "execute_performed": True, "schema": rel(SCHEMA)})
    return 0


def extract_packets(args: argparse.Namespace) -> int:
    labels = parse_labels(args.labels)
    packets: list[dict[str, Any]] = []
    blockers: list[str] = []
    spans_seen = 0
    with psycopg.connect(db_url(args)) as conn:
        rows = fetch_spans(conn, args.limit, only_unclaimed=args.only_unclaimed)
        for row in rows:
            spans_seen += 1
            try:
                md = load_markdown_text(str(row["output_markdown_path"]))
            except Exception as exc:
                blockers.append(f"markdown_unreadable:{row['output_markdown_path']}:{type(exc).__name__}")
                continue
            start = int(row["start_char"])
            end = int(row["end_char"])
            local_text = md[start:end] if 0 <= start <= end <= len(md) else ""
            if not local_text.strip():
                blockers.append(f"empty_or_invalid_span:{row['span_uuid']}")
                continue
            extraction = extract(local_text, labels, model=args.model, threshold=args.threshold, allow_remote_model=False, no_fallback=False)
            backend = str(extraction.get("backend") or "unknown")
            for span in extraction.get("spans", []):
                packets.append(packet_from_span(row, span, args.extractor_version, args.model_hash, args.authority_class, backend))

        inserted = 0
        if args.execute and packets:
            with conn.cursor() as cur:
                for packet in packets:
                    if insert_packet(cur, packet):
                        inserted += 1
            conn.commit()
        report = {
            "action": "extract",
            "execute_performed": bool(args.execute),
            "db_writes_performed": bool(args.execute),
            "graph_writes_performed": False,
            "canonical_graph_mutation": False,
            "spans_seen": spans_seen,
            "packets_prepared": len(packets),
            "packets_inserted": inserted,
            "extractor_name": "gliner_zero_shot_extractor",
            "extractor_version": args.extractor_version,
            "authority_class": args.authority_class,
            "review_state": "candidate_unreviewed",
            "truth_status": "not_truth_claim_candidate",
            "only_unclaimed": args.only_unclaimed,
            "sample_packets": packets[:5],
            "blockers": blockers,
        }
    write_report("extract_execute" if args.execute else "extract_dry_run", report)
    print(f"PACKETS_PREPARED={len(packets)}")
    print(f"PACKETS_INSERTED={inserted if args.execute else 0}")
    return 0 if packets or not args.require_packets else 2


def audit(args: argparse.Namespace) -> int:
    with psycopg.connect(db_url(args)) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT count(*) AS n FROM lucidota_korpus.document_claim_packet")
            total = cur.fetchone()["n"]
            cur.execute("SELECT label, count(*) AS n FROM lucidota_korpus.document_claim_packet GROUP BY label ORDER BY n DESC, label LIMIT 20")
            by_label = list(cur.fetchall())
            cur.execute("SELECT review_state, graph_promotion_status, count(*) AS n FROM lucidota_korpus.document_claim_packet GROUP BY review_state, graph_promotion_status ORDER BY 1,2")
            states = list(cur.fetchall())
    write_report("audit", {"action": "audit", "total_claim_packets": total, "by_label": by_label, "states": states, "graph_writes_performed": False})
    print(f"CLAIM_PACKETS_TOTAL={total}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge document parser spans into GLiNER claim packets")
    parser.add_argument("--database-url")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-schema")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=init_schema)
    p = sub.add_parser("extract")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--only-unclaimed", action="store_true", default=True)
    p.add_argument("--labels", default=str(ROOT / "05_OUTPUTS/contracts/operator_ontology_labels.json"))
    p.add_argument("--extractor-version", default="document_claim_packet_worker.v1")
    p.add_argument("--model", default=os.environ.get("GLINER_MODEL_PATH"))
    p.add_argument("--model-hash", default=os.environ.get("GLINER_MODEL_HASH", "missing_local_gliner_model"))
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--authority-class", default="model_computed_finding")
    p.add_argument("--require-packets", action="store_true")
    p.set_defaults(func=extract_packets)
    p = sub.add_parser("audit")
    p.set_defaults(func=audit)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
