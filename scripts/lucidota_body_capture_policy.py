#!/usr/bin/env python3
"""Body Capture watcher policy evaluator.

Evaluates latest capture pairs against watcher profiles. Capture is evidence;
alerts are policy-dependent.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import psycopg

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=ROOT/'06_SCHEMA'/'011_body_capture.sql'
DB=os.environ.get('LUCIDOTA_GRAPH_DATABASE_URL','postgresql://mfspx@/lucidota_graph')
ORDER={'ignore':0,'record_only':1,'alert':2,'escalate':3}


def max_outcome(*vals):
    return max(vals, key=lambda v: ORDER[v])


def ensure_assignment(conn, source:str, profile:str):
    conn.execute("""
      INSERT INTO lucidota_body_capture.watcher_assignment (source, profile_id, enabled, notes)
      VALUES (%s,%s,true,'auto v0 assignment')
      ON CONFLICT (source) DO UPDATE SET profile_id=EXCLUDED.profile_id, enabled=true, updated_at=now()
    """, (source, profile))


def evaluate(conn, source:str):
    row=conn.execute("""
      SELECT wa.profile_id, wp.alert_on_content, wp.alert_on_structure, wp.alert_on_visual, wp.visual_mode, wp.min_severity
      FROM lucidota_body_capture.watcher_assignment wa
      JOIN lucidota_body_capture.watcher_profile wp USING(profile_id)
      WHERE wa.source=%s AND wa.enabled
    """, (source,)).fetchone()
    if not row:
        return None
    profile_id, alert_content, alert_structure, alert_visual, visual_mode, min_severity = row
    caps=conn.execute("""
      SELECT capture_id, content_hash, structure_hash, visual_hash, sha256
      FROM lucidota_body_capture.capture
      WHERE source=%s AND status='succeeded'
      ORDER BY created_at DESC LIMIT 2
    """, (source,)).fetchall()
    if not caps:
        return None
    new=caps[0]; old=caps[1] if len(caps)>1 else None
    content_changed=bool(old and old[1] and new[1] and old[1] != new[1])
    structure_changed=bool(old and old[2] and new[2] and old[2] != new[2])
    visual_changed=bool(old and old[3] and new[3] and old[3] != new[3])
    outcome='record_only' if not old else 'ignore'
    reasons=[]
    if content_changed:
        reasons.append('content_changed')
        outcome=max_outcome(outcome, 'alert' if alert_content else 'record_only')
    if structure_changed:
        reasons.append('structure_changed')
        outcome=max_outcome(outcome, 'alert' if alert_structure else 'record_only')
    if visual_changed:
        reasons.append('visual_changed')
        outcome=max_outcome(outcome, visual_mode if alert_visual else 'record_only')
    if not reasons:
        reasons.append('no_profile_relevant_delta' if old else 'first_capture')
    if ORDER[outcome] < ORDER[min_severity] and outcome != 'ignore' and reasons != ['first_capture']:
        outcome=min_severity
    dec=conn.execute("""
      INSERT INTO lucidota_body_capture.watcher_decision
        (source,profile_id,old_capture_id,new_capture_id,content_changed,structure_changed,visual_changed,outcome,rationale,detail)
      VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
      RETURNING decision_id
    """, (source,profile_id,old[0] if old else None,new[0],content_changed,structure_changed,visual_changed,outcome,','.join(reasons),json.dumps({'old_sha256':old[4] if old else '', 'new_sha256':new[4]}))).fetchone()[0]
    return {'decision_id':str(dec),'source':source,'profile_id':profile_id,'outcome':outcome,'content_changed':content_changed,'structure_changed':structure_changed,'visual_changed':visual_changed,'rationale':reasons}


def main()->int:
    ap=argparse.ArgumentParser(prog='lucidota-body_capture-policy')
    ap.add_argument('source')
    ap.add_argument('--profile', default='content_truth')
    ap.add_argument('--db-url', default=DB)
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()
    with psycopg.connect(args.db_url) as conn:
        conn.execute(SCHEMA.read_text())
        ensure_assignment(conn,args.source,args.profile)
        result=evaluate(conn,args.source)
        conn.commit()
    report={'ok': result is not None, 'result': result}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0 if result else 1
if __name__=='__main__': raise SystemExit(main())
