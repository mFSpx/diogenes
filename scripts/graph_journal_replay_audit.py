#!/usr/bin/env python3
"""Replay/audit graph promotion journal materialization receipts."""
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
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'graph_journal_replay_audit_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print('REPORT_PATH='+rel(p))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); a=ap.parse_args(); blockers=[]; rows=[]
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute('''SELECT m.materialization_uuid::text,m.packet_uuid::text,m.decision_uuid::text,m.graph_item_uuid::text,m.graph_edge_uuid::text,m.journal_uuid::text,m.materialization_kind,m.evidence_refs,j.after_state
                  FROM lucidota_go.graph_promotion_materialization m LEFT JOIN lucidota_go.graph_journal j ON j.journal_uuid=m.journal_uuid ORDER BY m.created_at DESC LIMIT 5000''')
   for r in cur.fetchall():
    ok=True; problems=[]
    if not r['journal_uuid']: ok=False; problems.append('journal_missing')
    if not r['packet_uuid']: ok=False; problems.append('packet_missing')
    if not r['decision_uuid']: ok=False; problems.append('decision_missing')
    if r['materialization_kind']=='item' and r['graph_item_uuid']:
     cur.execute('SELECT 1 FROM lucidota_go.graph_item WHERE uuid=%s::uuid',(r['graph_item_uuid'],));
     if not cur.fetchone(): ok=False; problems.append('graph_item_missing')
    if r['materialization_kind']=='edge' and r['graph_edge_uuid']:
     cur.execute('SELECT 1 FROM lucidota_go.graph_edge WHERE edge_uuid=%s::uuid',(r['graph_edge_uuid'],));
     if not cur.fetchone(): ok=False; problems.append('graph_edge_missing')
    after=dict(r.get('after_state') or {})
    for key in ('packet_uuid','decision_uuid'):
     if after and after.get(key) and str(after.get(key)) != str(r[key]): ok=False; problems.append(f'after_state_{key}_mismatch')
    if problems: blockers.append(f"{r['materialization_uuid']}:{','.join(problems)}")
    d=dict(r); d['ok']=ok; d['problems']=problems; rows.append(d)
 payload={'action':'journal_replay_audit','materializations_checked':len(rows),'failures':len(blockers),'sample':rows[:20],'blockers':blockers[:100],'status':'PASS' if not blockers else 'FAIL'}
 write(payload); print('GRAPH_JOURNAL_REPLAY_AUDIT='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
