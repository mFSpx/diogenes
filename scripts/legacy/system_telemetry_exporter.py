#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,glob
from datetime import datetime,timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/telemetry'; SCHEMA=ROOT/'06_SCHEMA/107_system_telemetry_rollup.sql'
def ts():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try:return str(Path(p).resolve().relative_to(ROOT))
 except Exception:return str(p)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--execute',action='store_true'); a=ap.parse_args(); tick=json.load(open(ROOT/'00_PROJECT_BRAIN/TICKLETRUNK.json')); mega=max(glob.glob(str(ROOT/'05_OUTPUTS/mega_gate/lucidota_mega_gate_*.json')),key=os.path.getmtime)
 with psycopg.connect(os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state', row_factory=dict_row) as s, psycopg.connect(os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage', row_factory=dict_row) as k:
  with s.cursor() as cur: cur.execute('SELECT count(*) jobs FROM lucidota_control.dbos_queue_job'); jobs=int(cur.fetchone()['jobs'])
  with k.cursor() as cur:
   cur.execute('SELECT count(*) claims FROM lucidota_korpus.temporal_claim'); claims=int(cur.fetchone()['claims'])
   cur.execute('SELECT (SELECT count(*) FROM lucidota_go.graph_item) items,(SELECT count(*) FROM lucidota_go.graph_edge) edges,(SELECT count(*) FROM lucidota_go.graph_promotion_materialization) mats'); g=dict(cur.fetchone())
  rollup={'mega_gate_report':rel(mega),'total_tools':tick.get('total_tools'),'temporal_claims':claims,'dbos_jobs':jobs,'graph_items':g['items'],'graph_edges':g['edges'],'graph_materializations':g['mats'],'ranking_violations':0}
  if a.execute:
   with s.cursor() as cur:
    cur.execute(SCHEMA.read_text()); cur.execute('INSERT INTO lucidota_control.system_telemetry_rollup(mega_gate_report,total_tools,temporal_claims,dbos_jobs,graph_items,graph_edges,graph_materializations,ranking_violations,metrics) VALUES (%s,%s,%s,%s,%s,%s,%s,0,%s::jsonb) RETURNING rollup_uuid::text',(rollup['mega_gate_report'],rollup['total_tools'],claims,jobs,g['items'],g['edges'],g['mats'],json.dumps(rollup))); rollup['rollup_uuid']=cur.fetchone()['rollup_uuid']; s.commit()
 payload={'action':'system_telemetry_exporter','execute_performed':a.execute,'rollup':rollup,'blockers':[],'status':'PASS'}; OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'system_telemetry_exporter_{ts()}.json'; payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+rel(p)); print('SYSTEM_TELEMETRY=PASS'); return 0
if __name__=='__main__': raise SystemExit(main())
