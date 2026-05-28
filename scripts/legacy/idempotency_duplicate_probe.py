#!/usr/bin/env python3
"""Prove normalized idempotency duplicate suppression in the DBOS queue."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/"06_SCHEMA/060_idempotency_duplicate_suppression.sql"
OUT=ROOT/"05_OUTPUTS/dbos"
def stamp() -> str: return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def db(a: argparse.Namespace) -> str: return a.database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"
def norm(k: str) -> str: return "".join(ch for ch in "-".join(k.strip().lower().split()) if ch.isalnum() or ch in "-_.:")[:180]
def rel(p: Path | str) -> str:
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def write_report(name: str, d: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True); p=OUT/f"idempotency_duplicate_probe_{name}_{stamp()}.json"; d.setdefault("generated_at", now()); d["report_path"]=rel(p); p.write_text(json.dumps(d, indent=2, sort_keys=False, default=str)); print(f"REPORT_PATH={rel(p)}"); return p
def run(cmd: list[str]) -> dict[str, Any]:
    proc=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=120); return {"command":" ".join(cmd),"returncode":proc.returncode,"stdout":proc.stdout[-2000:],"stderr":proc.stderr[-2000:]}
def init_schema(a: argparse.Namespace) -> int:
    if a.execute:
        with psycopg.connect(db(a)) as c:
            with c.cursor() as cur: cur.execute(SCHEMA.read_text())
            c.commit()
    write_report("init_schema_execute" if a.execute else "init_schema_dry_run", {"execute_performed": bool(a.execute)}); return 0
def probe(a: argparse.Namespace) -> int:
    raw1=f" Round4 Duplicate Key {stamp()} "; raw2=raw1.lower().strip()
    payload1={"target_number":5,"target_name":"Idempotency and duplicate suppression","idempotency_key":raw1,"handler":"noop","message":"first"}
    payload2={**payload1,"idempotency_key":raw2,"message":"duplicate"}
    cmds=[run([sys.executable,"scripts/boring_beast.py","init-schema","--execute"]),
          run([sys.executable,"scripts/boring_beast.py","enqueue","--execute","--payload-json",json.dumps(payload1)]),
          run([sys.executable,"scripts/boring_beast.py","enqueue","--execute","--payload-json",json.dumps(payload2)])]
    inserted_flags=["INSERTED_NEW=true" in c["stdout"] for c in cmds]
    passed=inserted_flags[1] is True and inserted_flags[2] is False
    if a.execute:
        with psycopg.connect(db(a)) as c:
            with c.cursor() as cur:
                cur.execute("INSERT INTO lucidota_control.idempotency_attempt_audit(scope,raw_key,normalized_key,first_ref,duplicate,detail) VALUES ('boring_beast',%s,%s,%s,false,%s::jsonb)", (raw1,norm(raw1),"first_enqueue",json.dumps({"script":"idempotency_duplicate_probe.py"})))
                cur.execute("INSERT INTO lucidota_control.idempotency_attempt_audit(scope,raw_key,normalized_key,first_ref,duplicate,detail) VALUES ('boring_beast',%s,%s,%s,true,%s::jsonb)", (raw2,norm(raw2),"first_enqueue",json.dumps({"script":"idempotency_duplicate_probe.py"})))
            c.commit()
    report={"action":"probe","execute_performed":bool(a.execute),"db_writes_performed":bool(a.execute),"graph_writes_performed":False,"normalized_key":norm(raw1),"commands":cmds,"pass":passed,"blockers":[] if passed else ["duplicate_suppression_failed"]}
    write_report("execute" if a.execute else "dry_run", report); print("DUPLICATE_SUPPRESSION="+("PASS" if passed else "FAIL")); return 0 if passed else 2
def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--database-url"); sub=p.add_subparsers(dest="cmd", required=True)
    sp=sub.add_parser("init-schema"); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=init_schema)
    sp=sub.add_parser("probe"); sp.add_argument("--execute", action="store_true"); sp.set_defaults(func=probe)
    a=p.parse_args(); return a.func(a)
if __name__=="__main__": raise SystemExit(main())
