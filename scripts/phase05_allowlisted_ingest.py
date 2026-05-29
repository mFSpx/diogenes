#!/usr/bin/env python3
from __future__ import annotations
import argparse, fnmatch, hashlib, json, mimetypes, os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
ROOT=Path(__file__).resolve().parents[1]; SCHEMA0=ROOT/'06_SCHEMA/030_phase05_brain_archaeology.sql'; SCHEMA=ROOT/'06_SCHEMA/048_phase05_allowlisted_ingest.sql'; SEC=ROOT/'05_OUTPUTS/security'; OUT=ROOT/'05_OUTPUTS/phase05'
TEXT_SUFFIXES={'.md','.txt','.json','.jsonl','.yaml','.yml','.toml','.ini','.conf','.sql','.py','.rs','.sh'}
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def db(a): return a.database_url or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def latest_manifest():
 xs=sorted(SEC.glob('security_quarantine_manifest_*.json'),key=lambda p:p.stat().st_mtime,reverse=True); return xs[0] if xs else None
def rel(p):
 try: return str(p.resolve().relative_to(ROOT))
 except Exception: return str(p)
def write(name,d): OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'phase05_allowlisted_ingest_{name}_{stamp()}.json'; d['report_path']=rel(p); p.write_text(json.dumps(d,indent=2,sort_keys=False,default=str)); print(f'REPORT_PATH={rel(p)}'); return p
def is_excluded_by_manifest(path,patterns):
 r=rel(path); s=str(path)
 return any(fnmatch.fnmatch(r,p) or fnmatch.fnmatch(s,p) for p in patterns)
def init(a):
 if a.execute:
  with psycopg.connect(db(a)) as c:
   with c.cursor() as cur: cur.execute(SCHEMA0.read_text()); cur.execute(SCHEMA.read_text())
   c.commit()
 write('init_schema_execute' if a.execute else 'init_schema_dry_run',{'execute_performed':bool(a.execute)}); return 0
def iter_allowed(manifest,limit):
 # Manifest v2 has included clean count but not explicit clean paths; rescan roots while
 # enforcing the manifest's exclusion patterns plus any finding samples.
 data=json.loads(manifest.read_text());
 if not data.get('clean_manifest'): raise SystemExit('security_manifest_not_clean')
 blocked={f['path'] for f in data.get('findings',[])} | {f['path'] for f in data.get('excluded_findings_sample',[])}
 excludes=list(data.get('exclusion_patterns') or [])
 skip_dirs={'.git','node_modules','target','__pycache__','.venv','venv','.cache'}
 count=0
 for root in data.get('roots_scanned',[]):
  rp=Path(root)
  if not rp.exists() or not str(rp).startswith(str(ROOT)): continue
  for dirpath, dirnames, filenames in os.walk(rp, followlinks=False):
   dirnames[:]=[d for d in dirnames if d not in skip_dirs]
   for name in filenames:
    path=Path(dirpath)/name
    r=rel(path)
    if r in blocked or is_excluded_by_manifest(path,excludes): continue
    yield path
    count+=1
    if limit and count>=limit: return
def ingest(a):
 manifest=Path(a.manifest) if a.manifest else latest_manifest()
 if not manifest: write('execute',{'blockers':['security_manifest_missing'],'execute_performed':False}); return 2
 rows=[]
 for path in iter_allowed(manifest,a.limit):
  try:
   raw=path.read_bytes(); sh=sha_bytes(raw); status='custody_recorded' if path.suffix.lower() in TEXT_SUFFIXES else 'skipped_non_text'; preview=sha_bytes(raw[:4096]) if raw else sha_bytes(b'')
   rows.append({'path':rel(path),'sha':sh,'size':len(raw),'mime':mimetypes.guess_type(str(path))[0] or '', 'status':status, 'preview':preview})
  except Exception as e:
   rows.append({'path':rel(path),'sha':'0'*64,'size':0,'mime':'','status':'failed','preview':None,'error':str(e)})
 if a.execute:
  with psycopg.connect(db(a)) as c:
   with c.cursor() as cur:
    for r in rows:
     cur.execute('''INSERT INTO lucidota_archaeology.allowlisted_ingest_artifact(manifest_path,source_path,source_sha256,size_bytes,mime_guess,ingest_status,content_in_db,content_preview_sha256,detail)
       VALUES (%s,%s,%s,%s,%s,%s,false,%s,%s::jsonb) ON CONFLICT(manifest_path,source_path,source_sha256) DO UPDATE SET ingest_status=EXCLUDED.ingest_status, detail=EXCLUDED.detail''',(rel(manifest),r['path'],r['sha'],r['size'],r['mime'],r['status'],r['preview'],json.dumps({'script':'phase05_allowlisted_ingest.py'})))
   c.commit()
 report={'execute_performed':bool(a.execute),'manifest_path':rel(manifest),'clean_manifest':True,'rows_considered':len(rows),'custody_recorded':sum(1 for r in rows if r['status']=='custody_recorded'),'skipped_non_text':sum(1 for r in rows if r['status']=='skipped_non_text'),'failed':sum(1 for r in rows if r['status']=='failed'),'graph_writes_performed':False}
 write('execute' if a.execute else 'dry_run',report); return 0
def main():
 p=argparse.ArgumentParser(); p.add_argument('--database-url'); sub=p.add_subparsers(dest='cmd',required=True)
 sp=sub.add_parser('init-schema'); sp.add_argument('--execute',action='store_true'); sp.set_defaults(func=init)
 sp=sub.add_parser('ingest'); sp.add_argument('--execute',action='store_true'); sp.add_argument('--manifest'); sp.add_argument('--limit',type=int,default=50); sp.set_defaults(func=ingest)
 a=p.parse_args(); return a.func(a)
if __name__=='__main__': raise SystemExit(main())
