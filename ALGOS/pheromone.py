#!/usr/bin/env python3
"""Darwinian surface pheromone worker.

Executable lifecycle scaffold: records surface usage/promote/decay signals in
lucidota_runtime.surface_pheromone. It never mutates canonical graph/state.
"""
from __future__ import annotations
import argparse,json,os,math
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/surfaces'; SCHEMA=ROOT/'06_SCHEMA/029_darwinian_surfaces.sql'
def ts(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'pheromone_{name}_{ts()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def ensure_schema(cur): cur.execute(SCHEMA.read_text())
def signal(a):
 pheromone_uuid=None
 if a.execute:
  with psycopg.connect(db(a), row_factory=dict_row) as conn:
   with conn.cursor() as cur:
    ensure_schema(cur)
    cur.execute('''INSERT INTO lucidota_runtime.surface_pheromone(surface_key,signal_kind,signal_value,half_life_seconds,detail)
                   VALUES (%s,%s,%s,%s,%s::jsonb) RETURNING pheromone_uuid::text''',(a.surface_key,a.signal_kind,a.signal_value,a.half_life_seconds,json.dumps({'script':'ALGOS/pheromone.py'})))
    pheromone_uuid=cur.fetchone()['pheromone_uuid']
   conn.commit()
 report={'action':'signal','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'surface_key':a.surface_key,'signal_kind':a.signal_kind,'signal_value':a.signal_value,'pheromone_uuid':pheromone_uuid,'status':'PASS'}
 write('signal_execute' if a.execute else 'signal_dry_run',report); print('PHEROMONE_SIGNAL=PASS'); return 0
def decay(a):
 updated=0; rows=[]
 if a.execute:
  with psycopg.connect(db(a), row_factory=dict_row) as conn:
   with conn.cursor() as cur:
    ensure_schema(cur)
    cur.execute('''SELECT pheromone_uuid::text,surface_key,signal_value,half_life_seconds,created_at FROM lucidota_runtime.surface_pheromone WHERE active AND surface_key=%s ORDER BY created_at DESC LIMIT %s''',(a.surface_key,a.limit))
    current=[dict(r) for r in cur.fetchall()]
    for r in current:
     hours=max(0.0,(datetime.now(timezone.utc)-r['created_at']).total_seconds()/3600.0); half=max(1,float(r['half_life_seconds'])/3600.0); decayed=float(r['signal_value'])*math.pow(0.5,hours/half)
     rows.append({**r,'decayed_value':decayed})
     cur.execute('UPDATE lucidota_runtime.surface_pheromone SET signal_value=%s, detail=detail || %s::jsonb WHERE pheromone_uuid=%s::uuid',(decayed,json.dumps({'last_decay_worker':'ALGOS/pheromone.py'}),r['pheromone_uuid']))
     updated+=1
   conn.commit()
 else:
  rows=[{'surface_key':a.surface_key,'would_decay':'dry_run'}]
 report={'action':'decay','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'surface_key':a.surface_key,'rows_seen':len(rows),'rows_updated':updated,'rows':rows[:20],'status':'PASS'}
 write('decay_execute' if a.execute else 'decay_dry_run',report); print('PHEROMONE_DECAY=PASS'); return 0
def main():
 p=argparse.ArgumentParser(); p.add_argument('--database-url'); sub=p.add_subparsers(dest='cmd',required=True)
 s=sub.add_parser('signal'); s.add_argument('--surface-key',default='marrow_loop_status'); s.add_argument('--signal-kind',choices=['generated','used','promoted','forked','decayed','archived','operator_selected'],default='used'); s.add_argument('--signal-value',type=float,default=1.0); s.add_argument('--half-life-seconds',type=int,default=604800); s.add_argument('--execute',action='store_true')
 d=sub.add_parser('decay'); d.add_argument('--surface-key',default='marrow_loop_status'); d.add_argument('--limit',type=int,default=20); d.add_argument('--execute',action='store_true')
 a=p.parse_args(); return signal(a) if a.cmd=='signal' else decay(a)
if __name__=='__main__': raise SystemExit(main())
