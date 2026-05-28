#!/usr/bin/env python3
"""Production document parse ingestion path.

Local-first parser: text/markdown/json/csv become Markdown+JSON provenance records.
Binary/unsupported documents are deferred unless a real parser backend is installed.
Parser output is NOT truth; it is evidence-candidate material for later GLiNER/claim packets.
"""
from __future__ import annotations
import argparse, hashlib, json, mimetypes, os, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'06_SCHEMA/045_document_ingestion_pipeline.sql'
OUT=ROOT/'05_OUTPUTS/document_ingest'
TEXT_SUFFIXES={'.txt','.md','.markdown','.json','.jsonl','.csv','.tsv','.yaml','.yml','.toml','.ini','.conf'}
SPAN_NAMESPACE=uuid.UUID('00000000-0000-4000-8000-000000000045')

def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def sha_bytes(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def rel(p:Path)->str:
    try: return str(p.relative_to(ROOT))
    except Exception: return str(p)
def write_report(name,d): OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'document_parse_{name}_{stamp()}.json'; d.setdefault('generated_at',now()); d['report_path']=rel(p); p.write_text(json.dumps(d,indent=2,sort_keys=False,default=str)); print(f'REPORT_PATH={rel(p)}'); return p

def spans_for_text(text:str)->list[dict[str,Any]]:
    spans=[]; offset=0
    for idx,line in enumerate(text.splitlines(True)):
        end=offset+len(line)
        if line.strip():
            spans.append({'page_number':1,'block_index':idx,'start_char':offset,'end_char':end,'span_sha256':hashlib.sha256(line.encode()).hexdigest(),'text':line.rstrip('\n'),'label':'text_span','provenance':{'line_index':idx}})
        offset=end
    if not spans and text:
        spans.append({'page_number':1,'block_index':0,'start_char':0,'end_char':len(text),'span_sha256':hashlib.sha256(text.encode()).hexdigest(),'text':text,'label':'text_span','provenance':{'line_index':0}})
    return spans

def stable_span_uuid(run_uuid:str, span:dict[str,Any])->str:
    key=f"{run_uuid}:{span['page_number']}:{span['block_index']}:{span['start_char']}:{span['end_char']}:{span['span_sha256']}"
    return str(uuid.uuid5(SPAN_NAMESPACE,key))

def init_schema(a):
    if not a.execute: write_report('init_schema_dry_run',{'action':'init_schema','execute_performed':False,'schema':rel(SCHEMA)}); return 0
    with psycopg.connect(db(a)) as c:
      with c.cursor() as cur: cur.execute(SCHEMA.read_text())
      c.commit()
    write_report('init_schema_execute',{'action':'init_schema','execute_performed':True,'schema':rel(SCHEMA)}); return 0

def ingest(a):
    src=Path(a.input).expanduser().resolve(); raw=src.read_bytes(); source_sha=sha_bytes(raw); mime=mimetypes.guess_type(str(src))[0] or ''
    out_dir=OUT/'parsed'/source_sha[:12]; out_dir.mkdir(parents=True,exist_ok=True)
    status='deferred'; text=''; spans=[]; blockers=[]
    if src.suffix.lower() in TEXT_SUFFIXES:
        text=raw.decode('utf-8',errors='replace'); spans=spans_for_text(text); status='parsed'
    else:
        blockers.append('parser_backend_missing_for_non_text_document')
    json_path=out_dir/'parse.json'; md_path=out_dir/'parse.md'
    parsed={'schema':'lucidota.document_parse.local_text_v1','source_path':rel(src),'source_sha256':source_sha,'mime_guess':mime,'status':status,'parser_output_truth_status':'not_truth_evidence_candidate','spans':[{k:v for k,v in s.items() if k!='text'} for s in spans]}
    md='\n'.join(s.get('text','') for s in spans)
    report={'action':'ingest','execute_performed':False,'source_path':rel(src),'source_sha256':source_sha,'status':status,'span_count':len(spans),'output_json_path':rel(json_path),'output_markdown_path':rel(md_path),'db_writes_performed':False,'graph_writes_performed':False,'blockers':blockers}
    if not a.execute:
        write_report('ingest_dry_run',report); return 0 if status=='parsed' else 2
    json_path.write_text(json.dumps(parsed,indent=2,sort_keys=False),encoding='utf-8'); md_path.write_text(md,encoding='utf-8')
    with psycopg.connect(db(a)) as c:
      with c.cursor() as cur:
        cur.execute("""INSERT INTO lucidota_korpus.document_parse_run(parser_name,parser_version,source_path,source_sha256,mime_guess,status,output_json_path,output_markdown_path,detail)
          VALUES ('local_text','local_text_v1',%s,%s,%s,%s,%s,%s,%s::jsonb)
          ON CONFLICT(source_sha256,parser_name,parser_version) DO UPDATE SET status=EXCLUDED.status, output_json_path=EXCLUDED.output_json_path, output_markdown_path=EXCLUDED.output_markdown_path, detail=EXCLUDED.detail
          RETURNING run_uuid::text""",(rel(src),source_sha,mime,status,rel(json_path),rel(md_path),json.dumps({'script':'scripts/document_parse_ingest.py'})))
        run=cur.fetchone()[0]
        for s in spans:
          span_uuid=stable_span_uuid(run,s)
          provenance=dict(s['provenance'])
          provenance.update({'preservation_mode':'stable_uuid_upsert_no_delete','source_script':'scripts/document_parse_ingest.py'})
          cur.execute("""INSERT INTO lucidota_korpus.document_parse_span(span_uuid,run_uuid,page_number,block_index,start_char,end_char,span_sha256,label,provenance)
            VALUES (%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
            ON CONFLICT(span_uuid) DO UPDATE SET
              page_number=EXCLUDED.page_number,
              block_index=EXCLUDED.block_index,
              start_char=EXCLUDED.start_char,
              end_char=EXCLUDED.end_char,
              span_sha256=EXCLUDED.span_sha256,
              label=EXCLUDED.label,
              provenance=lucidota_korpus.document_parse_span.provenance || EXCLUDED.provenance""",(span_uuid,run,s['page_number'],s['block_index'],s['start_char'],s['end_char'],s['span_sha256'],s['label'],json.dumps(provenance)))
      c.commit()
    report.update({'execute_performed':True,'db_writes_performed':True,'run_uuid':run,'span_preservation_mode':'stable_uuid_upsert_no_delete','claim_packet_cascade_avoided':True})
    write_report('ingest_execute',report); print(f'RUN_UUID={run}'); return 0 if not blockers else 2

def main():
    p=argparse.ArgumentParser(); p.add_argument('--database-url'); sub=p.add_subparsers(dest='cmd',required=True)
    sp=sub.add_parser('init-schema'); sp.add_argument('--execute',action='store_true'); sp.set_defaults(func=init_schema)
    sp=sub.add_parser('ingest'); sp.add_argument('--execute',action='store_true'); sp.add_argument('--input',required=True); sp.set_defaults(func=ingest)
    a=p.parse_args(); return a.func(a)
if __name__=='__main__': raise SystemExit(main())
