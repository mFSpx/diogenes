#!/usr/bin/env python3
"""Query and validate graph promotion materialization receipts."""
from __future__ import annotations
import argparse,json,os
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
def write(payload):
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'graph_materialization_receipt_query_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print('REPORT_PATH='+rel(p))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--materialization-uuid'); ap.add_argument('--packet-uuid'); ap.add_argument('--command-envelope-uuid'); ap.add_argument('--limit',type=int,default=20); a=ap.parse_args(); blockers=[]; params=[]; where=[]
 if a.materialization_uuid: where.append('m.materialization_uuid=%s::uuid'); params.append(a.materialization_uuid)
 if a.packet_uuid: where.append('m.packet_uuid=%s::uuid'); params.append(a.packet_uuid)
 if a.command_envelope_uuid: where.append('m.command_envelope_uuid=%s::uuid'); params.append(a.command_envelope_uuid)
 sql='''SELECT m.materialization_uuid::text,m.packet_uuid::text,m.decision_uuid::text,m.command_envelope_uuid::text,m.graph_item_uuid::text,m.graph_edge_uuid::text,m.journal_uuid::text,m.materialization_kind,m.evidence_refs,m.created_at::text,d.decision,j.action,j.reason
        FROM lucidota_go.graph_promotion_materialization m
        LEFT JOIN lucidota_go.graph_promotion_decision d ON d.decision_uuid=m.decision_uuid
        LEFT JOIN lucidota_go.graph_journal j ON j.journal_uuid=m.journal_uuid'''
 if where: sql += ' WHERE ' + ' AND '.join(where)
 sql += ' ORDER BY m.created_at DESC LIMIT %s'; params.append(a.limit)
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute(sql,tuple(params)); rows=[dict(r) for r in cur.fetchall()]
   for r in rows:
    if not r.get('packet_uuid'): blockers.append(f"{r['materialization_uuid']}:missing_packet")
    if not r.get('decision_uuid'): blockers.append(f"{r['materialization_uuid']}:missing_decision")
    if not r.get('journal_uuid'): blockers.append(f"{r['materialization_uuid']}:missing_journal")
    if not r.get('evidence_refs'): blockers.append(f"{r['materialization_uuid']}:missing_evidence_refs")
 payload={'action':'materialization_receipt_query','rows_returned':len(rows),'rows':rows,'blockers':blockers,'status':'PASS' if rows and not blockers else ('FAIL' if blockers else 'NO_ROWS')}
 write(payload); print('GRAPH_MATERIALIZATION_RECEIPT_QUERY='+payload['status']); return 0 if rows and not blockers else 4
if __name__=='__main__': raise SystemExit(main())
