#!/usr/bin/env python3
"""Export DBOS queue metrics into lucidota_control.runtime_status_fact."""
from __future__ import annotations
import argparse,json,os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/dbos'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'dbos_queue_metrics_exporter_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def collect(cur):
 cur.execute('SELECT queue_name,status::text,count(*) AS n FROM lucidota_control.dbos_queue_job GROUP BY queue_name,status::text ORDER BY queue_name,status::text')
 by_queue={}
 for r in cur.fetchall(): by_queue.setdefault(r['queue_name'],{})[r['status']]=int(r['n'])
 cur.execute('SELECT status::text,count(*) AS n FROM lucidota_control.dbos_queue_job GROUP BY status::text')
 by_status={r['status']:int(r['n']) for r in cur.fetchall()}
 cur.execute('SELECT count(*) AS n FROM lucidota_control.dbos_queue_dead_letter WHERE resolved=false')
 dlq=int(cur.fetchone()['n'])
 cur.execute("SELECT count(*) AS n FROM lucidota_control.dbos_queue_job WHERE status IN ('running','leased') AND coalesce(last_heartbeat_at,locked_at,updated_at) < now() - interval '5 minutes'")
 stale=int(cur.fetchone()['n'])
 return {'by_queue':by_queue,'by_status':by_status,'unresolved_dead_letters':dlq,'stale_inflight_jobs':stale}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--execute',action='store_true'); a=ap.parse_args(); fact_uuid=None
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   metrics=collect(cur)
   if a.execute:
    cur.execute('''INSERT INTO lucidota_control.runtime_status_fact(subsystem,fact_key,fact_value,evidence_refs)
                   VALUES ('dbos_queue','queue_metrics',%s::jsonb,%s::jsonb)
                   ON CONFLICT(subsystem,fact_key) DO UPDATE SET fact_value=EXCLUDED.fact_value,evidence_refs=EXCLUDED.evidence_refs,derived_at=now()
                   RETURNING fact_uuid::text''',(json.dumps(metrics),json.dumps(['scripts/dbos_queue_metrics_exporter.py','lucidota_control.dbos_queue_job'])))
    fact_uuid=cur.fetchone()['fact_uuid']
  conn.commit()
 report={'action':'export_metrics','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'fact_uuid':fact_uuid,'metrics':metrics,'status':'PASS'}
 write(report); print('DBOS_QUEUE_METRICS=PASS'); return 0
if __name__=='__main__': raise SystemExit(main())
