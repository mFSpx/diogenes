#!/usr/bin/env python3
"""Attach TRACER-lite epistemic labels to document claim packets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
SCHEMA0 = ROOT / "06_SCHEMA/039_dbos_real_work_loop.sql"
SCHEMA = ROOT / "06_SCHEMA/057_tracer_claim_packet_bridge.sql"
OUT = ROOT / "05_OUTPUTS/tracer"
LABELS = {"quote","compression","inference","abduction","speculation","operator_prior","heuristic","contradiction","falsification_target","PFM"}

def stamp() -> str: return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
def state_db(a: argparse.Namespace) -> str: return a.state_database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"
def storage_db(a: argparse.Namespace) -> str: return a.storage_database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"
def rel(p: Path | str) -> str:
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True); p=OUT/f"tracer_claim_packet_bridge_{name}_{stamp()}.json"; payload.setdefault("generated_at", now()); payload["report_path"]=rel(p); p.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str)); print(f"REPORT_PATH={rel(p)}"); return p

def sha_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def init_schema(args: argparse.Namespace) -> int:
    if args.execute:
        with psycopg.connect(state_db(args)) as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA0.read_text(encoding="utf-8"))
                cur.execute(SCHEMA.read_text(encoding="utf-8"))
            conn.commit()
    write_report("init_schema_execute" if args.execute else "init_schema_dry_run", {"execute_performed": bool(args.execute), "schema": rel(SCHEMA)})
    return 0


def fetch_packets(args: argparse.Namespace) -> list[dict[str, Any]]:
    with psycopg.connect(storage_db(args), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            if args.claim_packet_uuid:
                cur.execute("SELECT claim_packet_uuid::text, source_span_id, evidence_refs, claim_text, label, matched_text, confidence_bps, authority_class FROM lucidota_korpus.document_claim_packet WHERE claim_packet_uuid=%s::uuid", (args.claim_packet_uuid,))
            else:
                cur.execute("SELECT claim_packet_uuid::text, source_span_id, evidence_refs, claim_text, label, matched_text, confidence_bps, authority_class FROM lucidota_korpus.document_claim_packet ORDER BY created_at DESC LIMIT %s", (args.limit,))
            return [dict(r) for r in cur.fetchall()]


def infer_label(packet: dict[str, Any]) -> str:
    text = f"{packet.get('claim_text','')} {packet.get('matched_text','')}".lower()
    if "detected" in text:
        return "inference"
    if '"' in text or "quote" in text:
        return "quote"
    return "heuristic"


def attach(args: argparse.Namespace) -> int:
    if args.label and args.label not in LABELS:
        write_report("blocked", {"execute_performed": False, "blockers": ["invalid_tracer_label"]}); return 2
    packets = fetch_packets(args)
    prepared = []
    for p in packets:
        label = args.label or infer_label(p)
        source_span = {"source_span_id": p["source_span_id"], "evidence_refs": p["evidence_refs"], "matched_text": p["matched_text"], "claim_text": p["claim_text"]}
        prepared.append({"packet": p, "label": label, "source_span": source_span})
    inserted = 0
    if args.execute:
        with psycopg.connect(state_db(args)) as conn:
            with conn.cursor() as cur:
                for item in prepared:
                    p = item["packet"]
                    cur.execute(
                        """
                        INSERT INTO lucidota_control.tracer_lite_label(packet_ref,label,source_span,authority_class,confidence_bps,detail)
                        VALUES (%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                        RETURNING trace_uuid::text
                        """,
                        (f"document_claim_packet:{p['claim_packet_uuid']}", item["label"], json.dumps(item["source_span"]), p["authority_class"], p["confidence_bps"], json.dumps({"script": "scripts/tracer_claim_packet_bridge.py"})),
                    )
                    trace_uuid = cur.fetchone()[0]
                    span_sha = sha_obj(item["source_span"])
                    cur.execute(
                        """
                        INSERT INTO lucidota_control.tracer_claim_packet_bridge(claim_packet_uuid,trace_uuid,label,source_span,source_span_sha256,authority_class,detail)
                        VALUES (%s::uuid,%s::uuid,%s,%s::jsonb,%s,%s,%s::jsonb)
                        ON CONFLICT (claim_packet_uuid, label, source_span_sha256) DO NOTHING
                        RETURNING bridge_uuid::text
                        """,
                        (p["claim_packet_uuid"], trace_uuid, item["label"], json.dumps(item["source_span"]), span_sha, p["authority_class"], json.dumps({"script": "scripts/tracer_claim_packet_bridge.py"})),
                    )
                    if cur.fetchone() is not None:
                        inserted += 1
            conn.commit()
    report = {"action": "attach", "execute_performed": bool(args.execute), "db_writes_performed": bool(args.execute), "graph_writes_performed": False, "packets_seen": len(packets), "labels_prepared": len(prepared), "labels_inserted": inserted, "sample": prepared[:5], "blockers": [] if prepared else ["no_claim_packets_found"]}
    write_report("attach_execute" if args.execute else "attach_dry_run", report)
    print(f"TRACER_LABELS_PREPARED={len(prepared)}")
    print(f"TRACER_LABELS_INSERTED={inserted if args.execute else 0}")
    return 0 if prepared else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Bridge claim packets to TRACER-lite labels")
    p.add_argument("--state-database-url"); p.add_argument("--storage-database-url")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("init-schema"); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=init_schema)
    sp = sub.add_parser("attach"); sp.add_argument("--execute", action="store_true"); sp.add_argument("--claim-packet-uuid"); sp.add_argument("--label"); sp.add_argument("--limit", type=int, default=25); sp.set_defaults(func=attach)
    args = p.parse_args(); return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
