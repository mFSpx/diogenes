#!/usr/bin/env python3
"""Local-first Wake Bus using a Postgres outbox.

Truth stays in workflow_event and related tables. This worker handles only small wake-up refs and marks outbox rows delivered. It is intentionally boring: durable DB rows, idempotent readers, no blobs in refs.
"""
from __future__ import annotations

import argparse, json, os
import psycopg

DB=os.environ.get('DBOS_SYSTEM_DATABASE_URL','postgresql://mfspx@/lucidota_state')
SCHEMAS=['06_SCHEMA/001_lucidota_control.sql','06_SCHEMA/010_wake_bus.sql']


def ensure_schema(conn):
    import pathlib
    root=pathlib.Path(__file__).resolve().parents[1]
    for rel in SCHEMAS:
        conn.execute((root/rel).read_text())
    conn.commit()


def seed_from_events(conn, limit:int)->int:
    rows=conn.execute("""
      INSERT INTO lucidota_control.event_outbox (event_id, topic, ref_body)
      SELECT e.event_id, 'workflow_event', jsonb_build_object(
        'event_id', e.event_id::text,
        'workflow_id', e.workflow_id,
        'run_id', e.run_id,
        'phase', e.phase,
        'status', e.status,
        'source', e.source
      )
      FROM lucidota_control.workflow_event e
      WHERE NOT EXISTS (
        SELECT 1 FROM lucidota_control.event_outbox o WHERE o.event_id = e.event_id
      )
      ORDER BY e.created_at ASC
      LIMIT %s
      RETURNING outbox_id
    """, (limit,)).fetchall()
    return len(rows)


def deliver_pending(conn, limit:int)->tuple[int,int,list[dict]]:
    rows=conn.execute("""
      WITH target_events AS (
        SELECT outbox_id
        FROM lucidota_control.event_outbox
        WHERE status='pending'
        ORDER BY created_at ASC
        FOR UPDATE SKIP LOCKED
        LIMIT %s
      ),
      delivered AS (
        UPDATE lucidota_control.event_outbox o
        SET status='delivered',
            attempts=attempts+1,
            delivered_at=now(),
            last_error=''
        FROM target_events t
        WHERE o.outbox_id=t.outbox_id
        RETURNING o.outbox_id, o.topic, o.ref_body
      ),
      notified AS (
        SELECT pg_notify(
          'lucidota_wake_bus',
          jsonb_build_object(
            'outbox_id', outbox_id::text,
            'topic', topic,
            'ref_body', ref_body
          )::text
        )
        FROM delivered
      )
      SELECT outbox_id, topic, ref_body FROM delivered
    """, (limit,)).fetchall()
    refs=[{'outbox_id':str(outbox_id),'topic':topic,'ref_body':ref_body} for outbox_id, topic, ref_body in rows]
    return len(rows), 0, refs


def main()->int:
    ap=argparse.ArgumentParser(prog='lucidota-wake-bus')
    ap.add_argument('--db-url', default=DB)
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--seed', action='store_true', help='Create pending outbox rows for workflow events missing one.')
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()
    with psycopg.connect(args.db_url) as conn:
        ensure_schema(conn)
        seeded=seed_from_events(conn,args.limit) if args.seed else 0
        delivered, failed, refs=deliver_pending(conn,args.limit)
        conn.execute("""
          INSERT INTO lucidota_bus.wake_run (status, scanned, delivered, failed, detail)
          VALUES (%s,%s,%s,%s,%s::jsonb)
        """, ('succeeded' if failed==0 else 'failed', seeded+delivered+failed, delivered, failed, json.dumps({'seeded':seeded,'sample':refs[:10]})))
        conn.commit()
    report={'ok':failed==0,'seeded':seeded,'delivered':delivered,'failed':failed,'mode':'postgres_outbox_signal'}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0 if failed==0 else 1
if __name__=='__main__': raise SystemExit(main())
