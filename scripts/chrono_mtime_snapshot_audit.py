#!/usr/bin/env python3
"""Audit mtime_snapshot_v1 temporal claims remain queryable and linked."""
from __future__ import annotations
import argparse, json, os
from datetime import datetime, timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_OUTPUTS/chrono_ledger'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def db(a): return a.database_url or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def write(n,d):
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'chrono_mtime_snapshot_audit_{n}_{stamp()}.json'; d.setdefault('generated_at',now()); d['report_path']=rel(p); p.write_text(json.dumps(d,indent=2,sort_keys=False,default=str)); print(f'REPORT_PATH={rel(p)}'); return p
def audit(a):
    with psycopg.connect(db(a), row_factory=dict_row) as c:
      cur=c.cursor()
      cur.execute("""
        SELECT count(*) AS total,
               count(DISTINCT file_uuid) AS files,
               count(*) FILTER (WHERE claim_uuid IS NULL) AS missing_claim_uuid,
               count(*) FILTER (WHERE file_uuid IS NULL) AS missing_file_uuid,
               count(*) FILTER (WHERE extractor NOT IN ('chrono_freeze_mtime','chrono_freeze_mtime.py')) AS unexpected_extractor,
               count(*) FILTER (WHERE trust_weight <> 0.10) AS unexpected_weight
        FROM lucidota_korpus.temporal_claim
        WHERE evidence_source='mtime_snapshot_v1'
      """)
      metrics=dict(cur.fetchone())
      cur.execute("""
        SELECT count(*) AS orphaned
        FROM lucidota_korpus.temporal_claim tc
        LEFT JOIN lucidota_korpus.file_object f ON f.file_uuid=tc.file_uuid
        WHERE tc.evidence_source='mtime_snapshot_v1' AND f.file_uuid IS NULL
      """)
      metrics['orphaned_file_links']=int(cur.fetchone()['orphaned'])
      cur.execute("""
        SELECT count(*) AS selected
        FROM lucidota_korpus.resolved_chrono_timeline_with_claim r
        JOIN lucidota_korpus.temporal_claim tc ON tc.claim_uuid=r.selected_claim_uuid
        WHERE tc.evidence_source='mtime_snapshot_v1'
      """)
      metrics['selected_as_best_count']=int(cur.fetchone()['selected'])
      cur.execute("SELECT evidence_source,count(*) AS n FROM lucidota_korpus.temporal_claim GROUP BY evidence_source ORDER BY n DESC")
      source_counts=[dict(r) for r in cur.fetchall()]
    total=int(metrics['total'] or 0)
    blockers=[]
    if total < a.expected_count: blockers.append(f'mtime_snapshot_count_below_expected:{total}<{a.expected_count}')
    if int(metrics['missing_claim_uuid'] or 0): blockers.append('missing_claim_uuid')
    if int(metrics['missing_file_uuid'] or 0): blockers.append('missing_file_uuid')
    if int(metrics['orphaned_file_links'] or 0): blockers.append('orphaned_file_links')
    if int(metrics['unexpected_extractor'] or 0): blockers.append('unexpected_extractor')
    if int(metrics['unexpected_weight'] or 0): blockers.append('unexpected_weight')
    report={'action':'audit','status':'PASS' if not blockers else 'FAIL','execute_performed':False,'db_writes_performed':False,'graph_writes_performed':False,'expected_count':a.expected_count,'metrics':metrics,'source_counts':source_counts,'blockers':blockers}
    write('pass' if not blockers else 'fail', report)
    print('MTIME_SNAPSHOT_AUDIT='+report['status']); print(f"MTIME_SNAPSHOT_CLAIMS={total}"); return 0 if not blockers else 4
def main():
    p=argparse.ArgumentParser(); p.add_argument('--database-url'); p.add_argument('--expected-count',type=int,default=18625); p.set_defaults(func=audit); a=p.parse_args(); return a.func(a)
if __name__=='__main__': raise SystemExit(main())
