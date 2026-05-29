#!/usr/bin/env python3
"""Deterministic telemetry finding worker from Sticker feature vectors."""
from __future__ import annotations

import argparse, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'06_SCHEMA/086_telemetry_finding_worker.sql'
OUT=ROOT/'05_OUTPUTS/phase05'

METRICS=[
 ('forensic_shield_ratio',0.05,'Forensic Shield signal observed'),
 ('manic_velocity',0.25,'Rapidthink / Manic Velocity signal observed'),
 ('resource_exhaustion_metric',0.25,'Resource Exhaustion Metric signal observed'),
 ('directive_ratio',0.20,'Directive density signal observed'),
]

def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def sig(obj:Any)->str: return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()

def write_report(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'telemetry_finding_worker_{name}_{stamp()}.json'
 payload.setdefault('generated_at',now()); payload['report_path']=rel(p)
 p.write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
 print(f'REPORT_PATH={rel(p)}'); return p

def init_schema(a):
 if a.execute:
  with psycopg.connect(db(a)) as conn:
   with conn.cursor() as cur: cur.execute(SCHEMA.read_text())
   conn.commit()
 write_report('init_schema_execute' if a.execute else 'init_schema_dry_run', {'action':'init_schema','execute_performed':bool(a.execute),'schema':rel(SCHEMA)})
 return 0

def candidates(row):
 raw=row.get('raw_features') or {}
 base={'vector_uuid':str(row['vector_uuid']),'artifact_uuid':str(row['artifact_uuid']),'component_uuid':str(row['component_uuid']) if row.get('component_uuid') else None,'chrono_uuid':str(row['chrono_uuid']) if row.get('chrono_uuid') else None}
 findings=[]
 # Always emit a deterministic observed-vector finding so every vector gets a machine-checkable telemetry receipt.
 findings.append({**base,'finding_kind':'other','label_key':None,'finding_text':'Sticker feature vector v1 observed and converted to deterministic telemetry evidence.','method':'mixed','authority_class':'deterministic_metric','confidence_bps':7000,'parameters':{'feature_version':row.get('feature_version'),'raw_feature_keys':sorted(raw.keys()) if isinstance(raw,dict) else []}})
 for metric,threshold,text in METRICS:
  val=row.get(metric)
  if val is not None and float(val) >= threshold:
   findings.append({**base,'finding_kind':'operator_diagnostic','label_key':None,'finding_text':f'{text}: {metric}={float(val):.6f} >= {threshold:.6f}.','method':'mixed','authority_class':'deterministic_metric','confidence_bps':min(10000,max(5000,int(float(val)*10000))),'parameters':{'metric':metric,'value':float(val),'threshold':threshold}})
 return findings

def run(a):
 inserted=0; seen=0; rows=[]
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute('''SELECT v.* FROM lucidota_archaeology.sticker_feature_vector v
                  LEFT JOIN lucidota_archaeology.telemetry_finding_worker_receipt r ON r.vector_uuid=v.vector_uuid
                  WHERE (%s OR r.receipt_uuid IS NULL)
                  ORDER BY v.created_at DESC LIMIT %s''',(a.reprocess,a.limit))
   vectors=[dict(r) for r in cur.fetchall()]
   for vec in vectors:
    for f in candidates(vec):
     seen+=1; signature=sig({'vector_uuid':f['vector_uuid'],'text':f['finding_text'],'params':f['parameters']})
     if a.execute:
      cur.execute('''INSERT INTO lucidota_archaeology.telemetry_finding(artifact_uuid,component_uuid,chrono_uuid,vector_uuid,finding_kind,label_key,finding_text,method,authority_class,confidence_bps,evidence,parameters)
                     SELECT %s::uuid,%s::uuid,%s::uuid,%s::uuid,%s,%s,%s,%s,%s::lucidota_archaeology.authority_class,%s,%s::jsonb,%s::jsonb
                     WHERE NOT EXISTS (SELECT 1 FROM lucidota_archaeology.telemetry_finding_worker_receipt WHERE finding_signature=%s)
                     RETURNING finding_uuid::text''',
                  (f['artifact_uuid'],f['component_uuid'],f['chrono_uuid'],f['vector_uuid'],f['finding_kind'],f['label_key'],f['finding_text'],f['method'],f['authority_class'],f['confidence_bps'],json.dumps([{'vector_uuid':f['vector_uuid'],'source':'sticker_feature_vector'}]),json.dumps(f['parameters']),signature))
      fr=cur.fetchone(); fid=fr['finding_uuid'] if fr else None
      if fid:
       inserted+=1
       cur.execute('INSERT INTO lucidota_archaeology.telemetry_finding_worker_receipt(vector_uuid,finding_signature,finding_uuid,detail) VALUES (%s::uuid,%s,%s::uuid,%s::jsonb) ON CONFLICT DO NOTHING',(f['vector_uuid'],signature,fid,json.dumps({'worker':'telemetry_finding_worker_v1'})))
     rows.append({'signature':signature,'finding_text':f['finding_text'],'would_insert':not a.execute})
  conn.commit()
 report={'action':'extract','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'vectors_seen':len(vectors),'findings_seen':seen,'findings_inserted':inserted,'sample':rows[:10]}
 write_report('extract_execute' if a.execute else 'extract_dry_run', report)
 print(f'TELEMETRY_FINDINGS_INSERTED={inserted}')
 return 0

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); sub=ap.add_subparsers(dest='cmd',required=True)
 s=sub.add_parser('init-schema'); s.add_argument('--execute',action='store_true')
 r=sub.add_parser('extract'); r.add_argument('--execute',action='store_true'); r.add_argument('--limit',type=int,default=100); r.add_argument('--reprocess',action='store_true')
 a=ap.parse_args(); return init_schema(a) if a.cmd=='init-schema' else run(a)
if __name__=='__main__': raise SystemExit(main())
