#!/usr/bin/env python3
"""Seed operator-supervised source policy rows for DIOGENES/LUCIDOTA."""
from __future__ import annotations
import argparse, json, os
import psycopg

DB=os.environ.get('DBOS_SYSTEM_DATABASE_URL','postgresql://mfspx@/lucidota_state')
ROWS=[
    ('public_web','web',['head','get_metadata','get_small_body'],30,False,'Public-web source intake; local-address intake requires explicit CLI option.'),
    ('wayback','api',['lookup_metadata'],10,True,'External archive enrichment is explicit only because it discloses the queried URL.'),
    ('local_files','filesystem',['read_operator_selected_file'],None,True,'Operator-selected local files only; never expose through unauthenticated surfaces.'),
    ('google_drive','drive',['list_requested','fetch_requested_metadata'],20,True,'Drive is not ambient context; use only after explicit operator request.'),
]

def main()->int:
    ap=argparse.ArgumentParser(prog='lucidota-source-policy-seed')
    ap.add_argument('--db-url', default=DB)
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()
    with psycopg.connect(args.db_url) as conn:
        for r in ROWS:
            conn.execute("""
              INSERT INTO lucidota_control.source_policy
                (source_id, source_kind, allowed_actions, rate_limit_per_minute, requires_user_approval, notes, updated_at)
              VALUES (%s,%s,%s,%s,%s,%s,now())
              ON CONFLICT (source_id) DO UPDATE SET
                source_kind=EXCLUDED.source_kind,
                allowed_actions=EXCLUDED.allowed_actions,
                rate_limit_per_minute=EXCLUDED.rate_limit_per_minute,
                requires_user_approval=EXCLUDED.requires_user_approval,
                notes=EXCLUDED.notes,
                updated_at=now()
            """, r)
        conn.commit()
    report={'ok':True,'policies':len(ROWS)}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0
if __name__=='__main__':
    raise SystemExit(main())
