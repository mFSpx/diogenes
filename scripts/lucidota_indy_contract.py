#!/usr/bin/env python3
"""Render Indy_Reads runtime contract from graph/DB first, Markdown if still present."""
from pathlib import Path
import json, argparse, os, sys
import psycopg
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'00_PROJECT_BRAIN'/'INDY_READS_RUNTIME_CONTRACT.md'
DSN=os.environ.get('LUCIDOTA_GO_STORAGE_DSN','postgresql:///lucidota_storage')

def graph_contract():
    try:
        with psycopg.connect(DSN, connect_timeout=3) as conn:
            row=conn.execute("""
              SELECT excerpt FROM lucidota_indy.markdown_artifact
              WHERE original_path='00_PROJECT_BRAIN/INDY_READS_RUNTIME_CONTRACT.md'
              ORDER BY updated_at DESC LIMIT 1
            """).fetchone()
            if row and row[0]:
                return row[0].splitlines()[:24]
    except Exception as exc:
        print(f"warning: unable to read Indy contract from Postgres: {exc}", file=sys.stderr)
    return []

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--json', action='store_true'); args=ap.parse_args()
    head=CONTRACT.read_text().splitlines()[:24] if CONTRACT.exists() else graph_contract()
    data={'ok': bool(head), 'contract_path': str(CONTRACT.relative_to(ROOT)), 'source': 'markdown' if CONTRACT.exists() else 'postgres:lucidota_indy.markdown_artifact', 'contract_head': head}
    print(json.dumps(data, sort_keys=True) if args.json else '\n'.join(head))
    return 0 if data['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
