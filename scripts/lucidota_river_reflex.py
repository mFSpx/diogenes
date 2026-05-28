#!/usr/bin/env python3
"""River incremental online-learning tick over new workflow events only."""
from __future__ import annotations
import argparse, json, os
from collections import defaultdict
from pathlib import Path
import psycopg
from river import compose, linear_model, preprocessing
ROOT=Path(__file__).resolve().parents[1]
DB=os.environ.get("DBOS_SYSTEM_DATABASE_URL") or os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"
SCHEMA=ROOT/"06_SCHEMA"/"004_learning_reflex.sql"
CURSOR="river_reflex_live"

def ensure(conn): conn.execute(SCHEMA.read_text(encoding="utf-8")); conn.commit()
def load_new(conn,limit:int):
    conn.execute("INSERT INTO lucidota_learning.river_event_cursor(cursor_name) VALUES (%s) ON CONFLICT (cursor_name) DO NOTHING",(CURSOR,))
    rows=conn.execute("""
      SELECT event_id::text, source, phase, status,
             COALESCE(detail->>'decision', detail->>'survey_decision', '') AS decision, created_at
      FROM lucidota_control.workflow_event
      WHERE status IN ('succeeded','failed') AND (created_at,event_id) > (
        SELECT last_event_at, COALESCE(last_event_id,'00000000-0000-0000-0000-000000000000'::uuid)
        FROM lucidota_learning.river_event_cursor WHERE cursor_name=%s)
      ORDER BY created_at ASC,event_id ASC LIMIT %s
    """,(CURSOR,limit)).fetchall()
    keys=["event_id","source","phase","status","decision","created_at"]
    return [dict(zip(keys,r)) for r in rows]
def feats(e): return {"source":e["source"],"phase":e["phase"],"decision":e["decision"] or "none"}
def update_cursor(conn,events):
    if not events: return
    last=events[-1]
    conn.execute("""UPDATE lucidota_learning.river_event_cursor SET last_event_at=%s,last_event_id=%s,updated_at=now() WHERE cursor_name=%s""",(last["created_at"],last["event_id"],CURSOR))

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--db-url",default=DB); ap.add_argument("--limit",type=int,default=5000); ap.add_argument("--json",action="store_true"); args=ap.parse_args()
    model=compose.Pipeline(preprocessing.OneHotEncoder(),linear_model.LogisticRegression()); grouped=defaultdict(lambda:{"n":0,"ok":0,"bad":0,"pred":None})
    with psycopg.connect(args.db_url) as conn:
        ensure(conn); events=load_new(conn,args.limit)
        for e in events:
            y=e["status"]=="succeeded"; pred=model.predict_proba_one(feats(e)).get(True,0.5); model.learn_one(feats(e),y); key=(e["source"],e["phase"],e["decision"] or ""); b=grouped[key]; b["n"]+=1; b["ok"]+=int(y); b["bad"]+=int(not y); b["pred"]=float(pred)
        for (source,phase,decision),b in grouped.items():
            conn.execute("""INSERT INTO lucidota_learning.river_score(source,phase,decision,examples,successes,failures,success_rate,river_prediction,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,now()) ON CONFLICT (source,phase,decision) DO UPDATE SET examples=lucidota_learning.river_score.examples+EXCLUDED.examples,successes=lucidota_learning.river_score.successes+EXCLUDED.successes,failures=lucidota_learning.river_score.failures+EXCLUDED.failures,success_rate=(lucidota_learning.river_score.successes+EXCLUDED.successes)::float/GREATEST(lucidota_learning.river_score.examples+EXCLUDED.examples,1),river_prediction=EXCLUDED.river_prediction,updated_at=now()""",(source,phase,decision,b["n"],b["ok"],b["bad"],b["ok"]/b["n"] if b["n"] else 0,b["pred"]))
        update_cursor(conn,events)
        conn.execute("INSERT INTO lucidota_learning.river_run(status,events_seen,examples_trained,detail) VALUES ('succeeded',%s,%s,%s::jsonb)",(len(events),len(events),json.dumps({"cursor":CURSOR,"groups_updated":len(grouped),"semantics":"new_events_since_cursor_not_replay_window"})))
        conn.commit()
    report={"ok":True,"cursor":CURSOR,"new_events_seen":len(events),"new_examples_trained":len(events),"groups_updated":len(grouped),"semantics":"new_events_since_cursor_not_replay_window"}; print(json.dumps(report,sort_keys=True) if args.json else report); return 0
if __name__=="__main__": raise SystemExit(main())
