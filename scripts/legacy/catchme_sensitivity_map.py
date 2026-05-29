#!/usr/bin/env python3
from __future__ import annotations
import argparse, fnmatch, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/'06_SCHEMA/046_catchme_sensitivity_map.sql'; OUT=ROOT/'05_OUTPUTS/catchme'
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def db(a): return a.database_url or os.environ.get('DBOS_SYSTEM_DATABASE_URL') or 'postgresql:///lucidota_state'
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def write(name,d): OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'catchme_{name}_{stamp()}.json'; d['report_path']=str(p.relative_to(ROOT)); p.write_text(json.dumps(d,indent=2,sort_keys=False)); print(f'REPORT_PATH={p.relative_to(ROOT)}'); return p
def init(a):
 if a.execute:
  with psycopg.connect(db(a)) as c:
   with c.cursor() as cur: cur.execute(SCHEMA.read_text())
   c.commit()
 write('init_schema_execute' if a.execute else 'init_schema_dry_run',{'execute_performed':bool(a.execute)}); return 0
def classify(a):
 path=rel(a.path); requested=a.requested_use; rules=[]
 with psycopg.connect(db(a)) as c:
  with c.cursor() as cur:
   cur.execute('SELECT rule_key,path_glob,sensitivity_class,consent_status,allowed_use,priority FROM lucidota_control.catchme_sensitivity_rule WHERE active ORDER BY priority ASC')
   rules=cur.fetchall()
 match=None
 for r in rules:
  if fnmatch.fnmatch(path,r[1]): match=r; break
 if match:
  key,glob,sens,consent,allowed_use,priority=match
 else:
  key=None; sens='private'; consent='requires_operator'; allowed_use='operator_review_only'
 allowed=(consent=='allowed') or (a.operator_approved and consent=='requires_operator')
 reason='allowed' if allowed else f'blocked:{consent}'
 report={'path_ref':path,'path_sha256':sha(path),'matched_rule_key':key,'sensitivity_class':sens,'consent_status':consent,'requested_use':requested,'operator_approved':bool(a.operator_approved),'allowed':allowed,'reason':reason,'execute_performed':False}
 if a.execute:
  with psycopg.connect(db(a)) as c:
   with c.cursor() as cur:
    cur.execute('''INSERT INTO lucidota_control.catchme_path_decision(path_sha256,path_ref,matched_rule_key,sensitivity_class,consent_status,requested_use,operator_approved,allowed,reason,detail)
      VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb) RETURNING decision_uuid::text''',(report['path_sha256'],path,key,sens,consent,requested,bool(a.operator_approved),allowed,reason,json.dumps({'script':'catchme_sensitivity_map.py'})))
    report['decision_uuid']=cur.fetchone()[0]
   c.commit()
  report['execute_performed']=True
 write('classify_execute' if a.execute else 'classify_dry_run',report); return 0 if allowed else 2
def main():
 p=argparse.ArgumentParser(); p.add_argument('--database-url'); sub=p.add_subparsers(dest='cmd',required=True)
 sp=sub.add_parser('init-schema'); sp.add_argument('--execute',action='store_true'); sp.set_defaults(func=init)
 sp=sub.add_parser('classify'); sp.add_argument('--execute',action='store_true'); sp.add_argument('--path',required=True); sp.add_argument('--requested-use',default='context_use'); sp.add_argument('--operator-approved',action='store_true'); sp.set_defaults(func=classify)
 a=p.parse_args(); return a.func(a)
if __name__=='__main__': raise SystemExit(main())
