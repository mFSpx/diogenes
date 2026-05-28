#!/usr/bin/env python3
"""Generate LongMemEval-style eval seeds after timeline/dev-pattern ingestion."""
from __future__ import annotations
import argparse,hashlib,json,os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; SCHEMA=ROOT/'06_SCHEMA/103_longmemeval_seed_generator.sql'; OUT=ROOT/'05_OUTPUTS/evals'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def key(*parts): return hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'longmemeval_seed_generator_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print('REPORT_PATH='+rel(p))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--execute',action='store_true'); ap.add_argument('--limit',type=int,default=40); a=ap.parse_args(); seeds=[]; inserted=0
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute(SCHEMA.read_text())
   cur.execute('SELECT atom_uuid::text, atom_kind, title, normalized_claim FROM lucidota_archaeology.design_atom ORDER BY created_at DESC LIMIT %s',(a.limit,))
   for r in cur.fetchall():
    claim=r.get('normalized_claim') or r.get('title') or ''
    cat='workflow_knowledge' if r.get('atom_kind') in ('workflow','operator_process') else 'operator_boundary' if r.get('atom_kind') in ('constraint','governance_rule','doctrine') else 'static_state'
    seeds.append({'category':cat,'question':f"What evidence supports this LUCIDOTA design claim: {claim[:180]}?",'expected_evidence_refs':[f"design_atom:{r['atom_uuid']}"],'source_kind':'design_atom','source_uuid':r['atom_uuid'],'idempotency_key':key(cat,r['atom_uuid'],claim[:80])})
   if a.execute:
    for s in seeds:
     cur.execute('''INSERT INTO lucidota_archaeology.longmem_eval_seed(category,question,expected_evidence_refs,source_kind,source_uuid,idempotency_key,detail)
                    VALUES (%s,%s,%s::jsonb,%s,%s::uuid,%s,%s::jsonb) ON CONFLICT(idempotency_key) DO NOTHING RETURNING seed_uuid''',(s['category'],s['question'],json.dumps(s['expected_evidence_refs']),s['source_kind'],s['source_uuid'],s['idempotency_key'],json.dumps({'script':'scripts/longmemeval_seed_generator.py'})))
     if cur.fetchone(): inserted += 1
  conn.commit()
 payload={'action':'longmemeval_seed_generate','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'seed_candidates':len(seeds),'inserted':inserted,'sample':seeds[:10],'blockers':[],'status':'PASS'}
 write(payload); print('LONGMEMEVAL_SEEDS=PASS'); return 0
if __name__=='__main__': raise SystemExit(main())
