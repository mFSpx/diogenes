#!/usr/bin/env python3
"""Validate generated surface lineage records and artifacts."""
from __future__ import annotations
import argparse,json,os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/surfaces'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def exists(path):
 if not path: return False
 p=ROOT/path if not Path(path).is_absolute() else Path(path)
 return p.exists()
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'surface_lineage_validator_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print('REPORT_PATH='+rel(p))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); a=ap.parse_args(); blockers=[]; checked=[]
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute('''SELECT l.lineage_uuid::text,l.parent_surface_key,l.child_surface_key,l.lineage_kind,l.evidence_refs,l.detail,
                         p.artifact_path parent_artifact,p.sidecar_path parent_sidecar,c.artifact_path child_artifact,c.sidecar_path child_sidecar
                  FROM lucidota_runtime.surface_lineage l
                  LEFT JOIN lucidota_runtime.surface_artifact p ON p.surface_key=l.parent_surface_key
                  LEFT JOIN lucidota_runtime.surface_artifact c ON c.surface_key=l.child_surface_key
                  ORDER BY l.created_at DESC LIMIT 5000''')
   for r in cur.fetchall():
    probs=[]
    if not (r.get('evidence_refs') or []): probs.append('evidence_refs_missing')
    if r.get('parent_artifact') and not exists(r.get('parent_artifact')): probs.append('parent_artifact_missing')
    if r.get('child_artifact') and not exists(r.get('child_artifact')): probs.append('child_artifact_missing')
    if r.get('parent_sidecar') and not exists(r.get('parent_sidecar')): probs.append('parent_sidecar_missing')
    if r.get('child_sidecar') and not exists(r.get('child_sidecar')): probs.append('child_sidecar_missing')
    d=dict(r); d['problems']=probs; d['ok']=not probs; checked.append(d)
    if probs: blockers.append(f"{r['lineage_uuid']}:{','.join(probs)}")
 payload={'action':'surface_lineage_validate','lineage_checked':len(checked),'failures':len(blockers),'sample':checked[:20],'blockers':blockers[:100],'status':'PASS' if not blockers else 'FAIL'}
 write(payload); print('SURFACE_LINEAGE_VALIDATOR='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
