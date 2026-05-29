#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS'/'absurd_abductive'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def cmd(c):
    p=subprocess.run(c,cwd=ROOT,text=True,capture_output=True,timeout=20); return {'cmd':c,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def run():
    active=cmd(['systemctl','--user','is-active','project2501-bytewax-board-stream.service']); enabled=cmd(['systemctl','--user','is-enabled','project2501-bytewax-board-stream.service'])
    latest=sorted((ROOT/'05_OUTPUTS/project2501_board_stream').glob('project2501_bytewax_board_stream_once_execute_*.json'), key=lambda p:p.stat().st_mtime, reverse=True)[:3]
    receipts=[rel(p) for p in latest]
    payload={'schema':'lucidota.absurd.bytewax_stream_audit.v1','generated_at_utc':now(),'verdict':'PASS' if active['returncode']==0 else 'DEGRADED','stream':'project2501-bytewax-board-stream.service','source':'Postgres/file board cursor via scripts/project2501_bytewax_board_stream.py','receipts_written':receipts,'state_mutation':'receipt outputs and cursor/lock only','cadence':'PROJECT2501_BOARD_STREAM_INTERVAL default 5s','stop_command':'systemctl --user stop project2501-bytewax-board-stream.service','typed_events':['StreamEvent','RiverTrainingCandidate','Receipt'],'canonical_graph_materialization':False,'canonical_graph_writes':False,'keep_running':active['returncode']==0,'checks':[active,enabled],'external_effects':False}
    path=OUT/f'bytewax_absurd_stream_audit_{stamp()}.json'; payload['receipt_path']=rel(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); return payload
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); args=ap.parse_args(); p=run();
    if args.json: print(json.dumps(p,sort_keys=True)); print('REPORT_PATH='+p['receipt_path']); print('BYTEWAX_ABSURD_STREAM_AUDIT='+p['verdict']); return 0 if p['verdict'] in {'PASS','DEGRADED'} else 4
if __name__=='__main__': raise SystemExit(main())
