#!/usr/bin/env python3
"""Bytewax live cursor over workflow_event rows; hard-fails on errors."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import psycopg
import numpy as np
from treelite import gtil, model_builder as mb
from bytewax.dataflow import Dataflow
import bytewax.operators as op
from bytewax.testing import TestingSink, TestingSource, run_main
ROOT=Path(__file__).resolve().parents[1]
DB=os.environ.get("DBOS_SYSTEM_DATABASE_URL") or os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"
SCHEMAS=[ROOT/"06_SCHEMA"/"004_learning_reflex.sql", ROOT/"06_SCHEMA"/"007_bytewax_stream.sql"]
ROUTER_COUNT=26
def router(i:int):
    meta=mb.Metadata(num_feature=5,task_type="kRegressor",average_tree_output=False,num_target=1,num_class=[1],leaf_vector_shape=(1,1)); ann=mb.TreeAnnotation(num_tree=1,target_id=[0],class_id=[0])
    b=mb.ModelBuilder(threshold_type="float32",leaf_output_type="float32",metadata=meta,tree_annotation=ann,postprocessor=mb.PostProcessorFunc(name="identity"),base_scores=[0.0])
    b.start_tree(); b.start_node(0); b.numerical_test(i%5,0.15+0.08*(i%4),default_left=True,opname="<",left_child_key=1,right_child_key=2); b.end_node(); b.start_node(1); b.leaf(0.0); b.end_node(); b.start_node(2); b.leaf(1.0+i/100); b.end_node(); b.end_tree(); return b.commit()
ROUTERS=[router(i) for i in range(ROUTER_COUNT)]
def treelite_scores(text:str):
    low=text.lower(); xs=np.array([[min(len(text)/180,1),sum(t in low for t in ["evidence","source","receipt","timeline","verify"])/5,sum(t in low for t in ["first step","run","wire","test","ship"])/5,sum(t in low for t in ["risk","blocker","failure","weak","trap","fraud"])/6,sum(t in low for t in ["because","therefore","must","exact","json"])/5]],dtype=np.float32)
    return [float(gtil.predict(m,xs).reshape(-1)[0]) for m in ROUTERS]

def hint(e):
    detail=e.get("detail") or {}; decision=detail.get("decision") or detail.get("survey_decision") or ""; ok=e.get("status")=="succeeded"
    text=" ".join([e.get("source",""),e.get("phase",""),e.get("status",""),decision,json.dumps(detail,sort_keys=True,default=str)])
    scores=treelite_scores(text); avg=sum(scores)/len(scores); score=max(0,min(100,round(avg*80)))
    return {"source":e.get("source",""),"phase":e.get("phase",""),"status":e.get("status",""),"hint":f"{e.get('source','')}:{e.get('phase','')}:{decision or e.get('status','')}","score":score if ok else min(score,40),"detail":{"workflow_id":e.get("workflow_id"),"decision":decision,"treelite_router_count":ROUTER_COUNT,"treelite_router_scores":scores,"treelite_matrix_score":avg}}
def flow(events):
    f=Dataflow("lucidota-bytewax-mini"); inp=op.input("events",f,TestingSource(events)); out=[]; op.output("out",op.map("hint",inp,hint),TestingSink(out)); run_main(f); return out
def rows(conn, limit:int, live:bool):
    if live:
        locked=conn.execute("SELECT pg_try_advisory_xact_lock(hashtext('lucidota_bytewax_live_cursor'))").fetchone()[0]
        if not locked: return [], {"cursor_lock":False}
        conn.execute("INSERT INTO lucidota_learning.river_event_cursor(cursor_name) VALUES ('bytewax_live') ON CONFLICT (cursor_name) DO NOTHING")
        sql="""SELECT event_id,workflow_id,phase,status,source,detail,created_at FROM lucidota_control.workflow_event WHERE (created_at,event_id) > (SELECT last_event_at,COALESCE(last_event_id,'00000000-0000-0000-0000-000000000000'::uuid) FROM lucidota_learning.river_event_cursor WHERE cursor_name='bytewax_live') ORDER BY created_at ASC,event_id ASC LIMIT %s"""
    else: sql="SELECT event_id,workflow_id,phase,status,source,detail,created_at FROM lucidota_control.workflow_event ORDER BY created_at DESC LIMIT %s"
    got=conn.execute(sql,(limit,)).fetchall(); return [dict(zip(["event_id","workflow_id","phase","status","source","detail","created_at"],r)) for r in got], {"cursor_lock":True}
def persist(conn, events, hints, mode, meta):
    for h in hints: conn.execute("INSERT INTO lucidota_learning.bytewax_hint(source,phase,status,hint,score,detail) VALUES (%s,%s,%s,%s,%s,%s::jsonb)",(h["source"],h["phase"],h["status"],h["hint"],h["score"],json.dumps(h["detail"])))
    conn.execute("INSERT INTO lucidota_learning.bytewax_stream_run(status,events_in,hints_out,detail) VALUES ('succeeded',%s,%s,%s::jsonb)",(len(events),len(hints),json.dumps({"mode":mode,**meta})))
    if events and mode=="live_cursor":
        last=events[-1]; conn.execute("""INSERT INTO lucidota_learning.river_event_cursor(cursor_name,last_event_at,last_event_id,updated_at) VALUES ('bytewax_live',%s,%s,now()) ON CONFLICT (cursor_name) DO UPDATE SET last_event_at=EXCLUDED.last_event_at,last_event_id=EXCLUDED.last_event_id,updated_at=now()""",(last["created_at"],last["event_id"]))
def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--db-url",default=DB); ap.add_argument("--limit",type=int,default=25); ap.add_argument("--live-cursor",action="store_true"); ap.add_argument("--json",action="store_true"); args=ap.parse_args()
    with psycopg.connect(args.db_url) as conn:
        for s in SCHEMAS: conn.execute(s.read_text(encoding="utf-8"))
        events,meta=rows(conn,args.limit,args.live_cursor); hints=flow(events); mode="live_cursor" if args.live_cursor else "latest_window"; persist(conn,events,hints,mode,meta); conn.commit()
    report={"ok":True,"events_in":len(events),"hints_out":len(hints),"mode":mode,**meta}; print(json.dumps(report,sort_keys=True) if args.json else report); return 0
if __name__=="__main__": raise SystemExit(main())
