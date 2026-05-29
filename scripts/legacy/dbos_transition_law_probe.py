#!/usr/bin/env python3
"""Apply and probe table-driven DBOS transition law."""
from __future__ import annotations
import argparse,json,os
from datetime import datetime,timezone
from pathlib import Path
import psycopg
ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/"06_SCHEMA/064_queue_transition_law_v2.sql"; OUT=ROOT/"05_OUTPUTS/dbos"
def stamp(): return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def now(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def db(a): return a.database_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def write(name,d): OUT.mkdir(parents=True,exist_ok=True); p=OUT/f"dbos_transition_law_probe_{name}_{stamp()}.json"; d.setdefault("generated_at",now()); d["report_path"]=rel(p); p.write_text(json.dumps(d,indent=2,sort_keys=False,default=str)); print(f"REPORT_PATH={rel(p)}"); return p
def main():
    p=argparse.ArgumentParser(); p.add_argument("--database-url"); p.add_argument("--execute",action="store_true"); a=p.parse_args()
    cases=[("queued","running","worker",True),("running","queued","auditor",False),("failed","queued","foreman",True),("succeeded","queued","worker",False),("dead_lettered","dead_lettered","auditor",True)]
    rows=[]
    with psycopg.connect(db(a)) as c:
        with c.cursor() as cur:
            if a.execute: cur.execute(SCHEMA.read_text())
            for old,new,role,expected in cases:
                cur.execute("SELECT lucidota_control.dbos_queue_transition_allowed(%s,%s,%s)",(old,new,role)); got=bool(cur.fetchone()[0]); rows.append({"old":old,"new":new,"role":role,"expected":expected,"got":got,"pass":got==expected})
            cur.execute("SELECT to_regclass('lucidota_control.dbos_queue_transition_policy')")
            if cur.fetchone()[0] is None:
                policy_count=0
            else:
                cur.execute("SELECT count(*) FROM lucidota_control.dbos_queue_transition_policy"); policy_count=cur.fetchone()[0]
        c.commit()
    passed=all(r["pass"] for r in rows)
    write("execute" if a.execute else "dry_run", {"execute_performed":bool(a.execute),"policy_count":policy_count,"cases":rows,"pass":passed,"blockers":[] if passed else ["transition_case_failed"]})
    print("TRANSITION_LAW="+("PASS" if passed else "FAIL")); return 0 if passed else 2
if __name__=="__main__": raise SystemExit(main())
