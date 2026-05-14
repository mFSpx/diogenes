#!/usr/bin/env python3
"""Authorized extractor registry: adapters first, browser fallback last."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import psycopg

ROOT=Path(__file__).resolve().parents[1]
DB=os.environ.get('LUCIDOTA_GRAPH_DATABASE_URL','postgresql://mfspx@/lucidota_graph')
SCHEMA=ROOT/'06_SCHEMA'/'012_authorized_extractors.sql'

def main()->int:
    ap=argparse.ArgumentParser(prog='lucidota-extractor-registry')
    ap.add_argument('--db-url', default=DB)
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()
    with psycopg.connect(args.db_url) as conn:
        conn.execute(SCHEMA.read_text())
        rows=conn.execute("""
          SELECT adapter_id, adapter_kind, stability, default_priority, browser_required
          FROM lucidota_extract.adapter
          ORDER BY default_priority ASC, adapter_id
        """).fetchall()
        conn.commit()
    adapters=[dict(zip(['adapter_id','adapter_kind','stability','default_priority','browser_required'], r)) for r in rows]
    report={'ok':True,'adapters':adapters,'browser_default':False,'policy':'adapters_first'}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0
if __name__=='__main__': raise SystemExit(main())
