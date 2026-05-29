#!/usr/bin/env python3
"""CLI reviewer for graph promotion packets: list/defer/reject/operator_confirmed."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/graph'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'graph_promotion_packet_reviewer_{name}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def run(cmd):
 proc=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=180); return {'command':' '.join(cmd),'returncode':proc.returncode,'stdout_tail':proc.stdout[-3000:],'stderr_tail':proc.stderr[-3000:]}
def list_packets(a):
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute('''SELECT packet_uuid::text, source_system, candidate_kind, promotion_status::text, authority_class, created_at::text, jsonb_array_length(evidence_refs) AS evidence_count
                  FROM lucidota_go.graph_promotion_packet WHERE promotion_status=%s ORDER BY created_at DESC LIMIT %s''',(a.status,a.limit))
   rows=[dict(r) for r in cur.fetchall()]
 report={'action':'list','execute_performed':False,'packets':rows,'count':len(rows)}; write('list',report); print(f'PACKETS={len(rows)}'); return 0
def review(a):
 cmd=[sys.executable,'scripts/graph_promotion_approval_worker.py','decide','--packet-uuid',a.packet_uuid,'--decision',a.decision,'--rationale',a.rationale,'--evidence-ref',','.join(a.evidence_ref)]
 if a.command_envelope_uuid: cmd+=['--command-envelope-uuid',a.command_envelope_uuid]
 if a.execute: cmd.append('--execute')
 res=run(cmd); packet_status=None
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute('SELECT promotion_status::text FROM lucidota_go.graph_promotion_packet WHERE packet_uuid=%s::uuid',(a.packet_uuid,)); r=cur.fetchone(); packet_status=r['promotion_status'] if r else None
 report={'action':'review','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute and res['returncode']==0),'graph_writes_performed':False,'packet_uuid':a.packet_uuid,'decision':a.decision,'packet_status_after':packet_status,'approval_worker':res,'status':'PASS' if res['returncode']==0 else 'FAIL','blockers':[] if res['returncode']==0 else ['approval_worker_failed']}
 write('review_execute' if a.execute else 'review_dry_run',report); print('PACKET_REVIEW='+report['status']); return 0 if res['returncode']==0 else 4
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); sub=ap.add_subparsers(dest='cmd',required=True)
 l=sub.add_parser('list'); l.add_argument('--status',default='candidate'); l.add_argument('--limit',type=int,default=20)
 r=sub.add_parser('review'); r.add_argument('--packet-uuid',required=True); r.add_argument('--decision',choices=['defer','reject','operator_confirmed'],required=True); r.add_argument('--rationale',default='packet reviewer decision'); r.add_argument('--evidence-ref',action='append',required=True); r.add_argument('--command-envelope-uuid'); r.add_argument('--execute',action='store_true')
 a=ap.parse_args(); return list_packets(a) if a.cmd=='list' else review(a)
if __name__=='__main__': raise SystemExit(main())
