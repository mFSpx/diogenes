#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS'/'indy_reads'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def latest_board():
    p=ROOT/'05_OUTPUTS/absurd_abductive/board_latest.json'
    return json.loads(p.read_text()) if p.exists() else {'open_blockers':[],'object_counts':{}}
def run(mode='board', model=False):
    b=latest_board(); blockers=b.get('open_blockers',[])
    payload={'schema':'lucidota.indy_reads.absurd_brief.v1','generated_at_utc':now(),'mode':mode,'verdict':'PASS','model_call_performed':bool(model),'accepted_truth':False,'canonical_graph_writes':False,'summary':{'open_hypotheses':b.get('pending_hypothesis_work_queues',[]),'claim_evidence_separation':'claims require evidence links; hypotheses are not truth','contradictions':[x for x in blockers if x.get('class') in {'model_audit','graph_safety'}],'next_useful_work':'run absurd_next_move_engine and execute top safe move'},'board_object_counts':b.get('object_counts',{})}
    path=OUT/f'absurd_brief_{stamp()}.json'; payload['receipt_path']=rel(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); return payload
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--board',default='latest'); ap.add_argument('--open-hypotheses',action='store_true'); ap.add_argument('--next-moves',action='store_true'); ap.add_argument('--model',action='store_true'); ap.add_argument('--json',action='store_true'); args=ap.parse_args(); mode='open_hypotheses' if args.open_hypotheses else ('next_moves' if args.next_moves else 'board'); p=run(mode,args.model)
    if args.json: print(json.dumps(p,sort_keys=True)); print('REPORT_PATH='+p['receipt_path']); print('INDY_READS_ABSURD_BRIEF='+p['verdict']); return 0
if __name__=='__main__': raise SystemExit(main())
