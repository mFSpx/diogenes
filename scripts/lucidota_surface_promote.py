#!/usr/bin/env python3
"""Promote generated surfaces with immutable lineage.

Dry-run is default. Execute requires --execute and --confirm-promote; execution copies
an artifact to promoted/ and appends lineage metadata only. No canonical graph writes.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/surfaces'; SCHEMA=ROOT/'06_SCHEMA/092_surface_promotion_lineage_guard.sql'
def ts(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def file_sha(p:Path):
 if not p.exists(): return None
 h=hashlib.sha256();
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
def sha_obj(o:Any): return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'surface_promote_{name}_{ts()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dry-run',action='store_true',default=True); p.add_argument('--execute',action='store_true'); p.add_argument('--confirm-promote',action='store_true'); p.add_argument('--init-schema',action='store_true'); p.add_argument('--database-url'); p.add_argument('--surface',default='07_SURFACES/generated/marrow_loop_status.html'); p.add_argument('--sidecar'); p.add_argument('--target-dir',default='07_SURFACES/promoted'); p.add_argument('--evidence-ref',action='append',default=[])
 a=p.parse_args(); blockers=[]; src=(ROOT/a.surface).resolve(); target=(ROOT/a.target_dir/src.name).resolve(); side=(ROOT/a.sidecar).resolve() if a.sidecar else None
 if not src.exists(): blockers.append('surface_missing')
 if side and not side.exists(): blockers.append('sidecar_missing')
 evidence=[x for raw in a.evidence_ref for x in raw.split(',') if x.strip()]
 if not evidence: evidence=[rel(src)] if src.exists() else []
 lineage={'parent_surface_key':rel(src),'child_surface_key':rel(target),'lineage_kind':'promoted_from','evidence_refs':evidence,'source_sha256':file_sha(src),'sidecar_path':rel(side) if side else None}
 lineage_sha=sha_obj(lineage)
 copied=False; lineage_uuid=None
 if a.execute and not a.confirm_promote: blockers.append('execute_requires_confirm_promote')
 if a.execute and not blockers:
  target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,target); copied=True
  with psycopg.connect(db(a)) as conn:
   with conn.cursor() as cur:
    if a.init_schema: cur.execute(SCHEMA.read_text())
    cur.execute('''INSERT INTO lucidota_runtime.surface_lineage(parent_surface_key,child_surface_key,lineage_kind,evidence_refs,lineage_sha256,immutable,detail)
                   VALUES (%s,%s,'promoted_from',%s::jsonb,%s,true,%s::jsonb)
                   ON CONFLICT(lineage_sha256) WHERE lineage_sha256 IS NOT NULL DO UPDATE SET detail=lucidota_runtime.surface_lineage.detail || EXCLUDED.detail
                   RETURNING lineage_uuid::text''',(rel(src),rel(target),json.dumps(evidence),lineage_sha,json.dumps({'script':'scripts/lucidota_surface_promote.py','target_sha256':file_sha(target)})))
    lineage_uuid=cur.fetchone()[0]
   conn.commit()
 report={'mode':'execute' if a.execute else 'dry_run','execute_performed':bool(a.execute),'surface':rel(src),'source_sha256':file_sha(src),'target':rel(target),'target_exists_before':target.exists(),'copied':copied,'lineage':lineage,'lineage_sha256':lineage_sha,'lineage_uuid':lineage_uuid,'db_writes_performed':bool(lineage_uuid),'graph_writes_performed':False,'canonical_mutation_allowed':False,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write('execute' if a.execute else 'dry_run',report); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
