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
    decision=(e.get('detail') or {}).get('decision') or (e.get('detail') or {}).get('scout_decision') or ''
    return {
        'source': e.get('source',''), 'phase': e.get('phase',''), 'status': e.get('status',''),
        'hint': f"{e.get('source','')}:{e.get('phase','')}:{decision or e.get('status','')}",
        'score': 80 if ok else 20, 'detail': {'workflow_id': e.get('workflow_id'), 'decision': decision}
    }

def load_events(limit:int)->list[dict]:
    with psycopg.connect(DB) as conn:
        conn.execute(SCHEMA.read_text())
        rows=conn.execute("""
            SELECT workflow_id, phase, status, source, detail
            FROM lucidota_control.workflow_event
            ORDER BY created_at DESC LIMIT %s
        """, (limit,)).fetchall()
        conn.commit()
    return [dict(zip(['workflow_id','phase','status','source','detail'], r)) for r in rows]

def run_flow(events:list[dict])->list[dict]:
    flow=Dataflow('lucidota-bytewax-mini')
    inp=op.input('events', flow, TestingSource(events))
    hints=op.map('event-to-hint', inp, event_to_hint)
    out=[]
    op.output('out', hints, TestingSink(out))
    run_main(flow)
    return out

def persist(events:list[dict], hints:list[dict]):
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
        """, (len(events), len(hints), json.dumps({'mode':'TestingSource mini live graph'})))
        conn.commit()

def main()->int:
    ap=argparse.ArgumentParser(prog='lucidota-bytewax-mini')
    ap.add_argument('--limit', type=int, default=25)
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()
    events=load_events(args.limit)
    hints=run_flow(events)
    persist(events,hints)
    report={'ok': True, 'events_in': len(events), 'hints_out': len(hints)}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0
if __name__=='__main__':
    raise SystemExit(main())
