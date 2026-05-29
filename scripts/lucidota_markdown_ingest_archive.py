#!/usr/bin/env python3
"""Ingest legacy Markdown breadcrumbs into Postgres graph and remove from active tree.

Default is dry-run. Use --execute to move non-README Markdown to 03_VAULT/ingested_markdown.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, uuid
from datetime import datetime, timezone
from pathlib import Path
import psycopg

ROOT=Path(__file__).resolve().parents[1]
DSN=os.environ.get('LUCIDOTA_GO_STORAGE_DSN','postgresql:///lucidota_storage')
OPERATOR_UUID=os.environ.get('LUCIDOTA_OPERATOR_UUID','00000000-0000-4000-8000-000000000414')
SCHEMAS=[ROOT/'06_SCHEMA'/'016_go_graph_core.sql', ROOT/'06_SCHEMA'/'017_indy_reads_library.sql']
KEEP_NAMES={'README.md'}
SKIP_PARTS={'.git','.venv','01_REPOS','03_VAULT'}
GRAPH_APPROVAL_MODE='approved' if os.environ.get('LUCIDOTA_GRAPH_APPROVAL_MODE','').strip().lower()=='approved' and os.environ.get('LUCIDOTA_ALLOW_DIRECT_GRAPH_APPROVAL','').strip().lower() in {'1','true','yes','on'} else 'staged'

def sha(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def rel(p:Path)->str: return str(p.relative_to(ROOT))
def title_of(text:str,path:Path)->str:
    for line in text.splitlines()[:20]:
        m=re.match(r'^#\s+(.+)', line.strip())
        if m: return m.group(1)[:500]
    return path.stem.replace('_',' ')[:500]
def candidates()->list[Path]:
    out=[]
    for p in ROOT.rglob('*.md'):
        if any(part in SKIP_PARTS for part in p.relative_to(ROOT).parts): continue
        if p.name in KEEP_NAMES: continue
        out.append(p)
    return sorted(out)
def graph_direct_approved()->bool: return GRAPH_APPROVAL_MODE=='approved'
def graph_item(cur, original:str, h:str, title:str, excerpt:str)->str:
    u=str(uuid.uuid5(uuid.NAMESPACE_URL, f'lucidota-md:{original}:{h}'))
    payload={'sha256':h,'source':'legacy_markdown_ingest','original_path':original,'excerpt':excerpt[:2000],'graph_write_mode':GRAPH_APPROVAL_MODE,'approval_required':not graph_direct_approved()}
    if graph_direct_approved():
        cur.execute("""
        INSERT INTO lucidota_go.graph_item(uuid,term,label,status,time_on_graph,location_at_on_graph,location_real_at_added,time_approved,location_real_at_approved,approval_scope,operator_uuid,payload)
        VALUES (%s,'SOURCE',%s,'approved',now(),%s,%s::jsonb,now(),%s::jsonb,'historical',%s,%s::jsonb)
        ON CONFLICT(uuid) DO UPDATE SET label=EXCLUDED.label,payload=EXCLUDED.payload,updated_at=now()
        RETURNING uuid::text
        """,(u,title,original,json.dumps({'path':original}),json.dumps({'path':original}),OPERATOR_UUID,json.dumps(payload,sort_keys=True)))
    else:
        cur.execute("""
        INSERT INTO lucidota_go.graph_item(uuid,term,label,status,time_on_graph,location_at_on_graph,location_real_at_added,operator_uuid,payload)
        VALUES (%s,'SOURCE',%s,'staged',now(),%s,%s::jsonb,%s,%s::jsonb)
        ON CONFLICT(uuid) DO UPDATE SET label=EXCLUDED.label,payload=EXCLUDED.payload,updated_at=now()
        RETURNING uuid::text
        """,(u,title,original,json.dumps({'path':original}),OPERATOR_UUID,json.dumps(payload,sort_keys=True)))
    return cur.fetchone()[0]

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--json',action='store_true')
    args=ap.parse_args()
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive=ROOT/'03_VAULT'/'ingested_markdown'/run_id
    manifest=[]
    files=candidates()
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            for s in SCHEMAS: cur.execute(s.read_text())
            for p in files:
                data=p.read_bytes(); h=sha(data); text=data.decode('utf-8',errors='ignore')
                original=rel(p); title=title_of(text,p); excerpt='\n'.join(text.splitlines()[:40])
                gid=graph_item(cur,original,h,title,excerpt)
                archived=''
                if args.execute:
                    dest=archive/original
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.move(str(p), str(dest))
                    archived=rel(dest)
                cur.execute("""
                INSERT INTO lucidota_indy.markdown_artifact(graph_item_uuid,original_path,archived_path,sha256,size_bytes,line_count,status,title,excerpt)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(original_path) DO UPDATE SET graph_item_uuid=EXCLUDED.graph_item_uuid, archived_path=EXCLUDED.archived_path, sha256=EXCLUDED.sha256,
                  size_bytes=EXCLUDED.size_bytes,line_count=EXCLUDED.line_count,status=EXCLUDED.status,title=EXCLUDED.title,excerpt=EXCLUDED.excerpt,updated_at=now()
                """,(gid,original,archived,h,len(data),text.count('\n')+1,'archived' if args.execute else 'ingested',title,excerpt[:4000]))
                manifest.append({'original_path':original,'archived_path':archived,'sha256':h,'graph_item_uuid':gid})
        conn.commit()
    if args.execute:
        (archive/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    out={'ok':True,'executed':args.execute,'count':len(manifest),'archive':rel(archive) if args.execute else None,'graph_approval_mode':GRAPH_APPROVAL_MODE,'items':manifest[:10]}
    print(json.dumps(out,indent=2,sort_keys=True) if args.json else out)
    return 0
if __name__=='__main__': raise SystemExit(main())
