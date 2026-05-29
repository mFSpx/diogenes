#!/usr/bin/env python3
"""Apply/check DBOS queue schema hardening v2."""
from __future__ import annotations
import argparse,json,os
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/"06_SCHEMA/061_dbos_queue_hardening_v2.sql"; OUT=ROOT/"05_OUTPUTS/dbos"
def stamp(): return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def now(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def db(a): return a.database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def write(name,d): OUT.mkdir(parents=True,exist_ok=True); p=OUT/f"dbos_queue_hardening_{name}_{stamp()}.json"; d.setdefault("generated_at",now()); d["report_path"]=rel(p); p.write_text(json.dumps(d,indent=2,sort_keys=False,default=str)); print(f"REPORT_PATH={rel(p)}"); return p
def main():
    p=argparse.ArgumentParser(); p.add_argument("--database-url"); p.add_argument("--execute",action="store_true"); a=p.parse_args()
    with psycopg.connect(db(a), row_factory=dict_row) as c:
        with c.cursor() as cur:
            if a.execute:
                cur.execute(SCHEMA.read_text())
            cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='lucidota_control' AND table_name='dbos_queue_job' AND column_name='payload_sha256') AS exists")
            has_payload_sha = bool(cur.fetchone()["exists"])
            if has_payload_sha:
                cur.execute("SELECT count(*) AS n FROM lucidota_control.dbos_queue_job WHERE payload_sha256 IS NULL OR payload_sha256 !~ '^[0-9a-f]{64}$'")
                bad_hash=int(cur.fetchone()["n"])
            else:
                bad_hash=-1
            cur.execute("SELECT count(*) AS n FROM lucidota_control.dbos_queue_job WHERE attempt_count < 0 OR attempt_count > max_attempts")
            bad_attempt=int(cur.fetchone()["n"])
            cur.execute("SELECT count(*) AS n FROM pg_trigger WHERE tgname='trg_set_dbos_queue_job_hardening_fields'")
            trigger_count=int(cur.fetchone()["n"])
        c.commit()
    blockers=[] 
    if bad_hash == -1: blockers.append("payload_sha256_column_missing")
    elif bad_hash: blockers.append(f"bad_payload_sha256_rows={bad_hash}")
    if bad_attempt: blockers.append(f"bad_attempt_bound_rows={bad_attempt}")
    if trigger_count<1: blockers.append("hardening_trigger_missing")
    write("execute" if a.execute else "dry_run", {"execute_performed":bool(a.execute),"bad_payload_sha256_rows":bad_hash,"bad_attempt_bound_rows":bad_attempt,"hardening_trigger_count":trigger_count,"blockers":blockers})
    print("DBOS_QUEUE_HARDENING="+("PASS" if not blockers else "BLOCKED"))
    return 0 if not blockers else 2
if __name__=="__main__": raise SystemExit(main())
