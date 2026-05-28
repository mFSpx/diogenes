#!/usr/bin/env python3
"""Workflow Foundry runtime: design atoms -> invariant candidates."""
from __future__ import annotations
import argparse,hashlib,json,os,re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/'06_SCHEMA/094_workflow_foundry_runtime.sql'; OUT=ROOT/'05_OUTPUTS/workflow_foundry'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def sig(text): return hashlib.sha256(text.encode()).hexdigest()
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'workflow_foundry_runtime_{name}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
FAMILIES={'absurd':'ABSURD queue/workflow invariant','graph':'Graph promotion invariant','chrono':'Chrono-Ledger temporal invariant','surface':'Surface/CEP invariant','tickletrunk':'TICKLETRUNK proof-hoard invariant','ontology':'Operator ontology invariant'}
def family(text):
 low=text.lower()
 for k,v in FAMILIES.items():
  if k in low: return k,v
 return 'general','Recovered workflow invariant'
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--execute',action='store_true'); ap.add_argument('--limit',type=int,default=500); a=ap.parse_args(); inserted=0; candidates=[]
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   if a.execute: cur.execute(SCHEMA.read_text())
   cur.execute("SELECT atom_uuid::text, atom_kind, title, normalized_claim, tags FROM lucidota_archaeology.design_atom WHERE atom_kind IN ('workflow','operator_process','governance_rule','constraint','requirement') ORDER BY created_at DESC LIMIT %s",(a.limit,))
   rows=[dict(r) for r in cur.fetchall()]
   buckets=defaultdict(list)
   for r in rows:
    fam,_=family((r.get('title') or '')+' '+(r.get('normalized_claim') or ''))
    buckets[fam].append(r)
   for fam,atoms in buckets.items():
    if not atoms: continue
    title=FAMILIES.get(fam,'Recovered workflow invariant')
    text=f"{title}: {len(atoms)} design atoms repeatedly express this family; candidate requires operator review before canonization."
    key=sig(fam+'|'+','.join(sorted(a['atom_uuid'] for a in atoms))[:500])
    candidates.append({'invariant_key':key,'workflow_family':fam,'title':title,'invariant_text':text,'source_atom_uuids':[a['atom_uuid'] for a in atoms[:50]],'confidence_bps':min(8500,5000+len(atoms)*200)})
    if a.execute:
     cur.execute('''INSERT INTO lucidota_workflow_foundry.workflow_invariant_candidate(invariant_key,workflow_family,title,invariant_text,source_atom_uuids,confidence_bps,detail)
                    VALUES (%s,%s,%s,%s,%s::uuid[],%s,%s::jsonb)
                    ON CONFLICT(invariant_key) DO UPDATE SET detail=lucidota_workflow_foundry.workflow_invariant_candidate.detail || EXCLUDED.detail
                    RETURNING candidate_uuid::text,(xmax=0) AS inserted_new''',(key,fam,title,text,[x['atom_uuid'] for x in atoms[:50]],min(8500,5000+len(atoms)*200),json.dumps({'script':'scripts/workflow_foundry_runtime.py'})))
     if cur.fetchone()['inserted_new']: inserted+=1
   conn.commit()
 report={'action':'mine_invariants','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'atoms_seen':len(rows),'candidates_seen':len(candidates),'candidates_inserted':inserted,'candidates':candidates[:20]}
 write('execute' if a.execute else 'dry_run',report); print(f'WORKFLOW_INVARIANTS_INSERTED={inserted}'); return 0
if __name__=='__main__': raise SystemExit(main())
