#!/usr/bin/env python3
"""Mini Bytewax live-stream proof over committed workflow events."""
from __future__ import annotations

import argparse, json, os
from pathlib import Path
import psycopg
from bytewax.dataflow import Dataflow
import bytewax.operators as op
from bytewax.testing import TestingSource, TestingSink, run_main

ROOT=Path(__file__).resolve().parents[1]
DB=os.environ.get('DBOS_SYSTEM_DATABASE_URL','postgresql://mfspx@/lucidota_state')
SCHEMA=ROOT/'06_SCHEMA'/'007_bytewax_stream.sql'

def event_to_hint(e: dict) -> dict:
    ok=e.get('status')=='succeeded'
    decision=(e.get('detail') or {}).get('decision') or (e.get('detail') or {}).get('survey_decision') or ''
    return {
        'source': e.get('source',''), 'phase': e.get('phase',''), 'status': e.get('status',''),
        'hint': f"{e.get('source','')}:{e.get('phase','')}:{decision or e.get('status','')}",
        'score': 80 if ok else 20, 'detail': {'workflow_id': e.get('workflow_id'), 'decision': decision}
    }

def load_events(limit:int, since_last: bool = False)->tuple[list[dict], dict]:
    meta={'cursor_lock': True, 'cursor_name': 'bytewax_live'}
    with psycopg.connect(DB) as conn:
        conn.execute(SCHEMA.read_text())
        if since_last:
            locked=conn.execute("SELECT pg_try_advisory_xact_lock(hashtext('lucidota_bytewax_live_cursor'))").fetchone()[0]
            meta['cursor_lock']=bool(locked)
            if not locked:
                conn.commit()
                return [], meta
            conn.execute("""
              INSERT INTO lucidota_learning.river_event_cursor (cursor_name)
              VALUES ('bytewax_live')
              ON CONFLICT (cursor_name) DO NOTHING
            """)
            rows=conn.execute("""
              SELECT event_id, workflow_id, phase, status, source, detail, created_at
              FROM lucidota_control.workflow_event
              WHERE (created_at, event_id) > (
                SELECT last_event_at, COALESCE(last_event_id, '00000000-0000-0000-0000-000000000000'::uuid)
                FROM lucidota_learning.river_event_cursor
                WHERE cursor_name='bytewax_live'
              )
              ORDER BY created_at ASC, event_id ASC
              LIMIT %s
            """, (limit,)).fetchall()
        else:
            rows=conn.execute("""
            SELECT event_id, workflow_id, phase, status, source, detail, created_at
            FROM lucidota_control.workflow_event
            ORDER BY created_at DESC LIMIT %s
            """, (limit,)).fetchall()
        conn.commit()
    events=[dict(zip(['event_id','workflow_id','phase','status','source','detail','created_at'], r)) for r in rows]
    meta['events_loaded']=len(events)
    return events, meta

def run_flow(events:list[dict])->list[dict]:
    flow=Dataflow('lucidota-bytewax-mini')
    inp=op.input('events', flow, TestingSource(events))
    hints=op.map('event-to-hint', inp, event_to_hint)
    out=[]
    op.output('out', hints, TestingSink(out))
    run_main(flow)
    return out

def persist(events:list[dict], hints:list[dict], mode: str, meta: dict):
    with psycopg.connect(DB) as conn:
        conn.execute(SCHEMA.read_text())
        for h in hints:
            conn.execute("""
                INSERT INTO lucidota_learning.bytewax_hint (source, phase, status, hint, score, detail)
                VALUES (%s,%s,%s,%s,%s,%s::jsonb)
            """, (h['source'], h['phase'], h['status'], h['hint'], h['score'], json.dumps(h['detail'])))
        conn.execute("""
            INSERT INTO lucidota_learning.bytewax_stream_run (status, events_in, hints_out, detail)
            VALUES ('succeeded', %s, %s, %s::jsonb)
        """, (len(events), len(hints), json.dumps({'mode':mode, **meta})))
        if events and mode == 'live_cursor':
            last=events[-1]
            conn.execute("""
              INSERT INTO lucidota_learning.river_event_cursor (cursor_name, last_event_at, last_event_id, updated_at)
              VALUES ('bytewax_live', %s, %s, now())
              ON CONFLICT (cursor_name) DO UPDATE SET
                last_event_at=EXCLUDED.last_event_at,
                last_event_id=EXCLUDED.last_event_id,
                updated_at=now()
            """, (last['created_at'], last['event_id']))
        conn.commit()

def main()->int:
    ap=argparse.ArgumentParser(prog='lucidota-bytewax-mini')
    ap.add_argument('--limit', type=int, default=25)
    ap.add_argument('--live-cursor', action='store_true', help='Process new workflow_event rows since the bytewax cursor.')
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()
    events, meta=load_events(args.limit, args.live_cursor)
    hints=run_flow(events)
    mode='live_cursor' if args.live_cursor else 'latest_window'
    persist(events,hints,mode,meta)
    report={'ok': True, 'events_in': len(events), 'hints_out': len(hints), 'mode': mode, **meta}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0
if __name__=='__main__':
    raise SystemExit(main())
