#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os
from datetime import datetime,timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/status'
def ts(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def one(cur,sql): cur.execute(sql); return dict(cur.fetchone())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--execute',action='store_true'); a=ap.parse_args(); blockers=[]; facts={}
 with psycopg.connect(os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state', row_factory=dict_row) as c:
  with c.cursor() as cur:
   facts['dbos']=one(cur,"SELECT count(*) jobs,count(*) FILTER (WHERE status='running' AND coalesce(lease_expires_at,now())<now()) stale_running FROM lucidota_control.dbos_queue_job")
   facts['runtime_fact']=one(cur,"SELECT count(*) facts FROM lucidota_control.runtime_status_fact")
 with psycopg.connect(os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage', row_factory=dict_row) as c:
  with c.cursor() as cur:
   facts['chrono']=one(cur,"SELECT count(*) claims,count(DISTINCT file_uuid) files_covered FROM lucidota_korpus.temporal_claim")
   facts['graph']=one(cur,"SELECT (SELECT count(*) FROM lucidota_go.graph_item) items,(SELECT count(*) FROM lucidota_go.graph_edge) edges")
   cur.execute('''WITH ranked AS (SELECT file_uuid, row_number() over(partition by file_uuid order by trust_weight desc,candidate_timestamp asc,created_at desc) rn FROM lucidota_korpus.temporal_claim WHERE file_uuid IS NOT NULL AND coalesce(invalid,false)=false) SELECT count(*) v FROM (SELECT file_uuid,count(*) FROM ranked WHERE rn=1 GROUP BY file_uuid HAVING count(*)<>1) x'''); facts['ranking_violations']=int(cur.fetchone()['v'])
 if int(facts['dbos']['stale_running']): blockers.append('STALE_RUNNING_JOBS')
 if facts['ranking_violations']!=0: blockers.append('CHRONO_RANKING_VIOLATIONS')
 if int(facts['chrono']['claims'])<1 or int(facts['graph']['items'])<1: blockers.append('CORE_FACTS_MISSING')
 payload={'action':'system_state_desync_detect','execute_performed':bool(a.execute),'facts':facts,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'system_state_desync_detector_{ts()}.json'; payload['generated_at']=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print('REPORT_PATH='+rel(p)); print('SYSTEM_STATE_DESYNC='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
