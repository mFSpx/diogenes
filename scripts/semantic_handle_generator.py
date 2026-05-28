#!/usr/bin/env python3
"""Generate operator-owned fungible semantic handles for tools/workflows/artifacts."""
from __future__ import annotations
import argparse,hashlib,json,os,re
from datetime import datetime, timezone
from pathlib import Path
import psycopg
ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/'06_SCHEMA/101_semantic_handle_registry.sql'; OUT=ROOT/'05_OUTPUTS/semantic'; MAN=ROOT/'00_PROJECT_BRAIN/TICKLETRUNK.json'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def slug(s): return re.sub(r'[^a-z0-9]+','-',s.lower()).strip('-')[:80] or hashlib.sha256(s.encode()).hexdigest()[:16]
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'semantic_handle_generator_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print('REPORT_PATH='+rel(p))
def entries(limit):
 data=json.loads(MAN.read_text(encoding='utf-8')); out=[]
 for cat, vals in (data.get('toolboxes') or {}).items():
  for e in vals:
   name=e.get('name') or Path(e.get('relative_path','')).stem
   target=e.get('relative_path') or e.get('path') or name
   h=f"{cat.lower()}:{slug(name)}"
   out.append({'handle_key':h,'display_name':name,'target_ref':target,'target_kind':cat.lower(),'proof_hoard_role':e.get('proof_hoard_role') or 'unknown','evidence_refs':[target]})
   if len(out)>=limit: return out
 return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--limit',type=int,default=100); ap.add_argument('--execute',action='store_true'); a=ap.parse_args(); blockers=[]
 vals=entries(a.limit); inserted=0
 with psycopg.connect(db(a)) as conn:
  with conn.cursor() as cur:
   cur.execute(SCHEMA.read_text())
   if a.execute:
    for e in vals:
     cur.execute('''INSERT INTO lucidota_semantic.semantic_handle(handle_key,display_name,target_ref,target_kind,proof_hoard_role,handle_is_truth,evidence_refs,detail)
                    VALUES (%s,%s,%s,%s,%s,false,%s::jsonb,%s::jsonb)
                    ON CONFLICT(handle_key) DO UPDATE SET updated_at=now(), target_ref=EXCLUDED.target_ref, proof_hoard_role=EXCLUDED.proof_hoard_role RETURNING handle_uuid''',
                 (e['handle_key'],e['display_name'],e['target_ref'],e['target_kind'],e['proof_hoard_role'],json.dumps(e['evidence_refs']),json.dumps({'semantic_id_not_truth':True})))
     if cur.fetchone(): inserted += 1
  conn.commit()
 payload={'action':'semantic_handle_generate','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'candidate_count':len(vals),'inserted_or_updated':inserted,'sample':vals[:10],'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(payload); print('SEMANTIC_HANDLES='+payload['status']); return 0
if __name__=='__main__': raise SystemExit(main())
