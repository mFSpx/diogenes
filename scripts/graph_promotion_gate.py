#!/usr/bin/env python3
"""Runtime graph promotion gate: evidence + authority + preflight before packet creation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from spine_authority_checker import decide_authority

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = [ROOT/"06_SCHEMA/034_graph_promotion_pipeline.sql", ROOT/"06_SCHEMA/044_graph_promotion_policy_roles.sql", ROOT/"06_SCHEMA/059_graph_promotion_gate_runtime.sql"]
OUT = ROOT / "05_OUTPUTS/graph"
AUTH = {"raw_evidence","operator_authored_assertion","operator_defined_label","deterministic_metric","statistical_finding","model_computed_finding","stream_ml_finding","graph_inferred_relation","operator_confirmed_finding","canonical_doctrine","external_action_authorized"}
def now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def stamp() -> str: return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def db(a: argparse.Namespace) -> str: return a.database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"
def sha_obj(o: Any) -> str: return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",",":"), default=str).encode()).hexdigest()
def rel(p: Path | str) -> str:
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def refs(raw: list[str]) -> list[str]:
    out=[]
    for r in raw:
        if r.strip().startswith("["):
            try: out += [str(x) for x in json.loads(r)]; continue
            except Exception: pass
        out += [x.strip() for x in r.split(",") if x.strip()]
    return out
def payload(raw: str) -> dict[str, Any]:
    s=raw.lstrip()
    if not s.startswith("{") and len(raw) < 240 and Path(raw).exists():
        return json.loads(Path(raw).read_text())
    data=json.loads(raw)
    if not isinstance(data, dict): raise ValueError("payload must be object")
    return data
def write_report(name: str, d: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True); p=OUT/f"graph_promotion_gate_{name}_{stamp()}.json"; d.setdefault("generated_at", now()); d["report_path"]=rel(p); p.write_text(json.dumps(d, indent=2, sort_keys=False, default=str)); print(f"REPORT_PATH={rel(p)}"); return p

def init_schema(a: argparse.Namespace) -> int:
    if a.execute:
        with psycopg.connect(db(a)) as c:
            with c.cursor() as cur:
                for s in SCHEMAS: cur.execute(s.read_text())
            c.commit()
    write_report("init_schema_execute" if a.execute else "init_schema_dry_run", {"execute_performed": bool(a.execute), "schemas": [rel(s) for s in SCHEMAS]}); return 0

def gate(a: argparse.Namespace) -> int:
    blockers=[]; ev=refs(a.evidence_ref)
    try: pl=payload(a.candidate_payload_json)
    except Exception as exc: pl={}; blockers.append(f"invalid_payload:{exc}")
    if not ev: blockers.append("evidence_refs_required")
    dry_run = bool(getattr(a, "dry_run", False) or not getattr(a, "execute", False))
    requested_effect = "materialize_canonical_graph" if a.materialize else "stage_graph_packet"
    requested_lane = "graph_promotion_execute" if a.execute else "graph_promotion_dry_run"
    authority_decision = decide_authority(
        authority_class=a.authority_class,
        effect=requested_effect,
        lane=requested_lane,
        evidence_refs=ev,
        operator_override=False,
    )
    if not authority_decision["allowed"]:
        blockers.extend(["authority_decision:" + b for b in authority_decision["blockers"]])
        if "unknown_authority" in authority_decision["blockers"]:
            blockers.append("invalid_authority_class")
    if dry_run and a.materialize:
        blockers.append("dry_run_materialize_refused")
    if a.materialize:
        if not a.execute:
            blockers.append("materialize_requires_execute")
        if not a.operator_confirmed:
            blockers.append("materialize_requires_operator_confirmed")
        if not a.command_envelope_uuid:
            blockers.append("materialize_requires_command_envelope_uuid")
        ledger = subprocess.run([sys.executable, "scripts/lucidota_status_ledger.py", "--check"], cwd=ROOT, text=True, capture_output=True)
        if ledger.returncode != 0:
            blockers.append("materialize_requires_status_ledger_check_pass")
        blockers.append("canonical_materialization_disabled_for_hardening_sprint")
    result={"action":"gate","dry_run":dry_run,"execute_performed":bool(a.execute and not dry_run),"db_writes_performed":False,"graph_writes_performed":False,"canonical_graph_writes_performed":False,"evidence_refs":ev,"authority_class":a.authority_class,"authority_decision":authority_decision,"command_envelope_uuid":a.command_envelope_uuid,"materialize_requested":bool(a.materialize),"candidate_payload_sha256":sha_obj(pl),"blockers":blockers,"preflight":None,"packet_uuid":None,"decision_uuid":None,"gate_uuid":None}
    if blockers:
        write_report("blocked", result); return 2
    with psycopg.connect(db(a)) as c:
        with c.cursor() as cur:
            cur.execute("SELECT lucidota_go.graph_promotion_preflight(%s,%s,%s,%s,%s::jsonb,%s::uuid)", (a.role_name, a.decision, bool(a.materialize), bool(a.operator_confirmed), json.dumps(ev), a.command_envelope_uuid))
            pre=cur.fetchone()[0]; result["preflight"]=pre
            allowed=bool(pre.get("allowed"))
            if a.execute and allowed:
                cur.execute("""INSERT INTO lucidota_go.graph_promotion_packet(source_system,candidate_kind,candidate_payload,evidence_refs,authority_class,detail)
                    VALUES (%s,%s,%s::jsonb,%s::jsonb,%s,%s::jsonb)
                    ON CONFLICT (packet_dedupe_key) DO UPDATE SET detail=lucidota_go.graph_promotion_packet.detail || EXCLUDED.detail
                    RETURNING packet_uuid::text""", (a.source_system,a.candidate_kind,json.dumps(pl),json.dumps(ev),a.authority_class,json.dumps({"script":"scripts/graph_promotion_gate.py"})))
                packet=cur.fetchone()[0]
                cur.execute("""INSERT INTO lucidota_go.graph_promotion_decision(packet_uuid,decision,decided_by,rationale,evidence_refs,operator_confirmed,command_envelope_uuid)
                    VALUES (%s::uuid,%s,'graph_promoter',%s,%s::jsonb,%s,%s::uuid) RETURNING decision_uuid::text""", (packet,a.decision,a.rationale,json.dumps(ev),bool(a.operator_confirmed),a.command_envelope_uuid))
                decision=cur.fetchone()[0]; result.update({"packet_uuid":packet,"decision_uuid":decision,"db_writes_performed":True})
            if not dry_run:
                cur.execute("""INSERT INTO lucidota_go.graph_promotion_gate_audit(candidate_payload_sha256,decision,materialize_requested,preflight,allowed,packet_uuid,decision_uuid,detail)
                    VALUES (%s,%s,%s,%s::jsonb,%s,%s::uuid,%s::uuid,%s::jsonb) RETURNING gate_uuid::text""", (result["candidate_payload_sha256"],a.decision,bool(a.materialize),json.dumps(pre),allowed,result["packet_uuid"],result["decision_uuid"],json.dumps({"script":"scripts/graph_promotion_gate.py"})))
                result["gate_uuid"]=cur.fetchone()[0]
                result["db_writes_performed"] = True
        c.commit()
    if not result["preflight"].get("allowed"): result["blockers"] += list(result["preflight"].get("blockers", []))
    write_report("dry_run" if dry_run else ("execute" if a.execute else "preflight"), result)
    print("GRAPH_GATE_ALLOWED=" + ("true" if not result["blockers"] else "false"))
    if result["packet_uuid"]: print(f"PACKET_UUID={result['packet_uuid']}")
    return 0 if not result["blockers"] else 2

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--database-url"); sub=p.add_subparsers(dest="cmd", required=True)
    sp=sub.add_parser("init-schema"); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=init_schema)
    sp=sub.add_parser("gate"); mode=sp.add_mutually_exclusive_group(); mode.add_argument("--dry-run", action="store_true"); mode.add_argument("--execute", action="store_true"); sp.add_argument("--source-system", default="graph_promotion_gate"); sp.add_argument("--candidate-kind", choices=["node","edge","property","doctrine","workflow","other"], default="node"); sp.add_argument("--candidate-payload-json", required=True); sp.add_argument("--evidence-ref", action="append", default=[]); sp.add_argument("--authority-class", default="operator_authored_assertion"); sp.add_argument("--role-name", default="graph_promoter"); sp.add_argument("--decision", default="defer"); sp.add_argument("--rationale", default="Graph gate packet staging"); sp.add_argument("--operator-confirmed", action="store_true"); sp.add_argument("--materialize", action="store_true"); sp.add_argument("--command-envelope-uuid"); sp.set_defaults(func=gate)
    a=p.parse_args(); return a.func(a)
if __name__ == "__main__": raise SystemExit(main())
