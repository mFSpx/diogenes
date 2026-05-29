#!/usr/bin/env python3
"""Tiny GOALS next-goal packet queue."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def loadq(root:Path)->dict:
 p=root/'GOALS/NEXT_GOAL_QUEUE.json'
 return json.loads(p.read_text()) if p.exists() else {'schema':'lucidota.goals.next_goal_queue.v1','queue':[]}
def seed(root:Path=ROOT,title:str='Next Goal',objective:str='',minutes:int=0)->dict:
 root=Path(root); q=loadq(root)
 pkt={'schema':'lucidota.goals.next_goal_packet.v1','created_at':now(),'title':title,'objective':objective,'away_minutes':minutes,'status':'queued','proof_rule':'no claim without receipt/test/check','resume':'cat GOALS/NEXT_GOAL.md && cat GOALS/CURRENT_HANDOFF.md'}
 q['queue'].append(pkt); (root/'GOALS').mkdir(exist_ok=True)
 (root/'GOALS/NEXT_GOAL_QUEUE.json').write_text(json.dumps(q,indent=2,sort_keys=True)+'\n')
 (root/'GOALS/NEXT_GOAL.md').write_text(f"# {title}\n\nCreated: `{pkt['created_at']}`\nAway minutes: `{minutes}`\n\n{objective}\n\nProof: every feature needs evidence before complete.\n")
 return pkt
def check(root:Path=ROOT)->dict:
 q=loadq(Path(root)); bad=[i for i,p in enumerate(q.get('queue',[])) if not p.get('objective') or not p.get('proof_rule')]
 return {'schema':'lucidota.goals.next_goal_check.v1','generated_at':now(),'queued':len(q.get('queue',[])),'bad_packets':bad,'passed':not bad}
def main()->int:
 ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True); s=sub.add_parser('seed'); s.add_argument('--title',required=True); s.add_argument('--objective',required=True); s.add_argument('--away-minutes',type=int,default=0); sub.add_parser('check'); a=ap.parse_args()
 r=seed(title=a.title,objective=a.objective,minutes=a.away_minutes) if a.cmd=='seed' else check()
 print(json.dumps(r,indent=2,sort_keys=True)); return 0 if r.get('passed',True) else 1
if __name__=='__main__': raise SystemExit(main())
