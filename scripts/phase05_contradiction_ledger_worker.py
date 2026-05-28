#!/usr/bin/env python3
"""Extract Phase 0.5 contradiction/boundary records from design atoms and operator laws."""
from __future__ import annotations
import argparse, hashlib, json, os, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "06_SCHEMA/099_phase05_contradiction_ledger.sql"
OUT = ROOT / "05_OUTPUTS/phase05"
PATTERNS = [
    ("generated_not_policy_mutable", re.compile(r"generated\s*(?:!=|does\s+not\s+mean|is\s+not)\s*policy[- ]mutable", re.I)),
    ("retrieved_not_verified", re.compile(r"retrieved\s*(?:!=|does\s+not\s+mean|is\s+not)\s*verified", re.I)),
    ("repeated_not_preferred", re.compile(r"repeated\s*(?:!=|does\s+not\s+mean|is\s+not)\s*preferred", re.I)),
    ("surface_not_ui", re.compile(r"surface\s*(?:!=|does\s+not\s+mean|is\s+not)\s*UI", re.I)),
    ("graph_path_not_evidence", re.compile(r"graph\s+path\s*(?:!=|does\s+not\s+mean|is\s+not)\s*evidence", re.I)),
    ("must_not_boundary", re.compile(r"\b(must\s+not|do\s+not|forbidden|rejected|blocked|cannot\s+bypass|no\s+direct)\b", re.I)),
]
OPERATOR_LAWS = [
    ("generated_not_policy_mutable", "Generated does not mean policy-mutable."),
    ("retrieved_not_verified", "Retrieved does not mean verified."),
    ("repeated_not_preferred", "Repeated does not mean preferred."),
    ("surface_not_ui", "Surface is not UI; conversation is the UI."),
    ("graph_path_not_evidence", "Graph paths are maps to evidence, not evidence themselves."),
]

def now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
def stamp() -> str: return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def rel(p: Path | str) -> str:
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def db_url(a: argparse.Namespace) -> str: return a.database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"
def idem(*parts: Any) -> str: return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
def report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"phase05_contradiction_ledger_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now()); payload["report_path"] = rel(p)
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(p)}")
    return p

def candidates(cur) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cur.execute("""SELECT atom_uuid::text, atom_kind, title, normalized_claim, raw_excerpt
                   FROM lucidota_archaeology.design_atom ORDER BY created_at DESC LIMIT 5000""")
    for row in cur.fetchall():
        text = "\n".join(str(row.get(k) or "") for k in ("title", "normalized_claim", "raw_excerpt"))
        for key, rx in PATTERNS:
            if rx.search(text):
                out.append({"source_kind":"design_atom","source_uuid":row["atom_uuid"],"source_ref":f"lucidota_archaeology.design_atom:{row['atom_uuid']}","boundary_key":key,"contradiction_text":(row.get("normalized_claim") or row.get("title") or text)[:2000],"evidence_refs":[f"design_atom:{row['atom_uuid']}"],"idempotency_key":idem("design_atom", row["atom_uuid"], key)})
    cur.execute("""SELECT blueprint_uuid::text, workflow_key, title, purpose FROM lucidota_archaeology.workflow_blueprint ORDER BY created_at DESC LIMIT 1000""")
    for row in cur.fetchall():
        text = "\n".join(str(row.get(k) or "") for k in ("workflow_key", "title", "purpose"))
        for key, rx in PATTERNS:
            if rx.search(text):
                out.append({"source_kind":"workflow_blueprint","source_uuid":row["blueprint_uuid"],"source_ref":f"lucidota_archaeology.workflow_blueprint:{row['blueprint_uuid']}","boundary_key":key,"contradiction_text":(row.get("purpose") or row.get("title") or text)[:2000],"evidence_refs":[f"workflow_blueprint:{row['blueprint_uuid']}"],"idempotency_key":idem("workflow_blueprint", row["blueprint_uuid"], key)})
    for key, text in OPERATOR_LAWS:
        out.append({"source_kind":"operator_law","source_uuid":None,"source_ref":"operator_law:hard_law", "boundary_key":key,"contradiction_text":text,"evidence_refs":["00_PROJECT_BRAIN/STATUS_LEDGER.md"],"idempotency_key":idem("operator_law", key, text)})
    # de-dupe preserving order
    seen=set(); uniq=[]
    for c in out:
        if c["idempotency_key"] not in seen:
            uniq.append(c); seen.add(c["idempotency_key"])
    return uniq

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url")
    ap.add_argument("--init-schema", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    action = "extract" if a.extract else "init_schema" if a.init_schema else "check"
    payload: dict[str, Any] = {"action": action, "execute_performed": bool(a.execute), "db_writes_performed": False, "graph_writes_performed": False, "blockers": []}
    with psycopg.connect(db_url(a), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            if a.init_schema or a.extract:
                cur.execute(SCHEMA.read_text())
            if a.extract:
                rows = candidates(cur)
                payload["candidate_count"] = len(rows)
                if a.execute:
                    inserted = 0
                    for c in rows:
                        cur.execute("""INSERT INTO lucidota_archaeology.contradiction_ledger
                            (source_kind, source_uuid, source_ref, boundary_key, contradiction_text, evidence_refs, authority_class, review_state, idempotency_key, detail)
                            VALUES (%s,%s::uuid,%s,%s,%s,%s::jsonb,'operator_doctrine','candidate',%s,%s::jsonb)
                            ON CONFLICT(idempotency_key) DO NOTHING RETURNING contradiction_uuid""",
                            (c["source_kind"], c["source_uuid"], c["source_ref"], c["boundary_key"], c["contradiction_text"], json.dumps(c["evidence_refs"]), c["idempotency_key"], json.dumps({"detector":"pattern+operator_law"})))
                        if cur.fetchone(): inserted += 1
                    payload.update({"inserted": inserted, "db_writes_performed": True})
                payload["sample"] = rows[:10]
            if a.init_schema and a.execute:
                payload["db_writes_performed"] = True
        conn.commit()
    payload["status"] = "PASS" if not payload["blockers"] else "FAIL"
    report(action + ("_execute" if a.execute else "_dry_run"), payload)
    print("CONTRADICTION_LEDGER=" + payload["status"])
    return 0 if payload["status"] == "PASS" else 4
if __name__ == "__main__": raise SystemExit(main())
