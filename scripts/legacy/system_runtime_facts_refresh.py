#!/usr/bin/env python3
"""System-wide runtime facts refresh from live DB/daemon evidence."""
from __future__ import annotations
import argparse,json,os,subprocess
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/status'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def sdb(a): return a.state_database_url or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def kdb(a): return a.storage_database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def q(cur,sql): cur.execute(sql); return dict(cur.fetchone())
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'system_runtime_facts_refresh_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print('REPORT_PATH='+rel(p))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--state-database-url'); ap.add_argument('--storage-database-url'); ap.add_argument('--execute',action='store_true'); a=ap.parse_args(); facts={}; blockers=[]
 with psycopg.connect(sdb(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   facts['dbos_queue']=q(cur,"SELECT count(*) total, count(*) FILTER (WHERE status='queued') queued, count(*) FILTER (WHERE status='succeeded') succeeded, count(*) FILTER (WHERE status='failed') failed FROM lucidota_control.dbos_queue_job")
   facts['conversation_command']=q(cur,"SELECT count(*) total, count(*) FILTER (WHERE status='executed') executed FROM lucidota_control.conversation_command")
 with psycopg.connect(kdb(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   facts['chrono']=q(cur,"SELECT count(*) temporal_claims, count(DISTINCT file_uuid) files_covered FROM lucidota_korpus.temporal_claim")
   facts['graph']=q(cur,"SELECT (SELECT count(*) FROM lucidota_go.graph_item) graph_items, (SELECT count(*) FROM lucidota_go.graph_edge) graph_edges, (SELECT count(*) FROM lucidota_go.graph_promotion_materialization) materializations")
 chrono_proc=subprocess.run(['scripts/check_chrono_ledger_service.sh'],cwd=ROOT,text=True,capture_output=True)
 facts['chrono_service_rc']=chrono_proc.returncode
 if chrono_proc.returncode!=0: blockers.append('chrono_service_check_failed')
 if a.execute:
  with psycopg.connect(sdb(a)) as conn:
   with conn.cursor() as cur:
    cur.execute("""INSERT INTO lucidota_control.runtime_status_fact(subsystem,fact_key,fact_value,evidence_refs) VALUES ('system','runtime_facts_refresh',%s::jsonb,%s::jsonb)
                   ON CONFLICT(subsystem,fact_key) DO UPDATE SET fact_value=EXCLUDED.fact_value,evidence_refs=EXCLUDED.evidence_refs,derived_at=now()""",(json.dumps(facts),json.dumps({'script':'scripts/system_runtime_facts_refresh.py','chrono_stdout_tail':chrono_proc.stdout[-1000:]})))
   conn.commit()
 payload={'action':'system_runtime_facts_refresh','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'facts':facts,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(payload); print('SYSTEM_RUNTIME_FACTS='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
