#!/usr/bin/env python3
"""Detect orphan graph promotion packets/decisions/materializations."""
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
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'graph_promotion_orphan_detector_{payload["status"].lower()}_{stamp()}.json'; payload.setdefault('generated_at',now()); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8'); print(f'REPORT_PATH={rel(p)}'); return p
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); ap.add_argument('--max-candidate-age-minutes',type=int,default=1440); a=ap.parse_args(); blockers=[]
 with psycopg.connect(db(a), row_factory=dict_row) as conn:
  with conn.cursor() as cur:
   cur.execute('''SELECT p.packet_uuid::text,p.source_system,p.promotion_status::text,p.created_at::text FROM lucidota_go.graph_promotion_packet p
                  LEFT JOIN lucidota_go.graph_promotion_decision d ON d.packet_uuid=p.packet_uuid
                  WHERE d.decision_uuid IS NULL AND p.promotion_status <> 'candidate'
                  ORDER BY p.created_at DESC LIMIT 100'''); noncandidate_no_decision=[dict(r) for r in cur.fetchall()]
   cur.execute('''SELECT d.decision_uuid::text,d.packet_uuid::text,d.decision::text FROM lucidota_go.graph_promotion_decision d LEFT JOIN lucidota_go.graph_promotion_packet p ON p.packet_uuid=d.packet_uuid WHERE p.packet_uuid IS NULL LIMIT 100'''); decisions_no_packet=[dict(r) for r in cur.fetchall()]
   cur.execute('''SELECT m.materialization_uuid::text,m.packet_uuid::text,m.decision_uuid::text FROM lucidota_go.graph_promotion_materialization m LEFT JOIN lucidota_go.graph_promotion_packet p ON p.packet_uuid=m.packet_uuid LEFT JOIN lucidota_go.graph_promotion_decision d ON d.decision_uuid=m.decision_uuid WHERE p.packet_uuid IS NULL OR d.decision_uuid IS NULL LIMIT 100'''); mats_orphan=[dict(r) for r in cur.fetchall()]
   cur.execute('''SELECT count(*) AS n FROM lucidota_go.graph_promotion_packet WHERE promotion_status='candidate' AND created_at < now() - (%s || ' minutes')::interval''',(a.max_candidate_age_minutes,)); aged_candidates=int(cur.fetchone()['n'])
 if noncandidate_no_decision: blockers.append('noncandidate_packets_without_decision')
 if decisions_no_packet: blockers.append('decisions_without_packet')
 if mats_orphan: blockers.append('materializations_without_packet_or_decision')
 report={'action':'detect_orphans','db_writes_performed':False,'graph_writes_performed':False,'noncandidate_without_decision':noncandidate_no_decision,'decisions_without_packet':decisions_no_packet,'materializations_without_packet_or_decision':mats_orphan,'aged_candidate_count':aged_candidates,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 write(report); print('GRAPH_PROMOTION_ORPHANS='+report['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
