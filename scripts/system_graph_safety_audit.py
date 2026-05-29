#!/usr/bin/env python3
"""System-wide graph safety audit: orphan/direct-write/materialization/journal checks."""
from __future__ import annotations
import argparse,json,subprocess,sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/graph'
RUN_STEPS=[
 [sys.executable,'scripts/graph_promotion_orphan_detector.py'],
 [sys.executable,'scripts/graph_write_blocker_probe.py'],
 [sys.executable,'scripts/graph_edge_write_blocker_probe.py','--execute'],
 [sys.executable,'scripts/graph_journal_replay_audit.py'],
 [sys.executable,'scripts/graph_canonical_mutation_detector.py'],
]
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def text(value):
 if value is None: return ''
 if isinstance(value,bytes): return value.decode('utf-8','replace')
 return str(value)
def run(cmd, *, timeout_sec:int):
 try:
  p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=timeout_sec)
  rc=p.returncode; stdout=p.stdout; stderr=p.stderr
 except subprocess.TimeoutExpired as exc:
  rc=124; stdout=text(exc.stdout); stderr=text(exc.stderr)+f"\nTIMEOUT after {timeout_sec}s"
 rp=None
 for line in stdout.splitlines():
  if line.startswith('REPORT_PATH='): rp=line.split('=',1)[1]
 return {'command':' '.join(cmd),'rc':rc,'report_path':rp,'stdout_tail':stdout[-1500:],'stderr_tail':stderr[-1500:]}
def build_parser():
 p=argparse.ArgumentParser(description='System-wide graph safety audit: orphan/direct-write/materialization/journal checks.')
 p.add_argument('--timeout-seconds',type=int,default=30,help='Per-child-command timeout; timed-out children fail the audit instead of hanging the gate.')
 p.add_argument('--list-steps',action='store_true',help='Print the audit plan without executing child checks.')
 return p
def main(argv=None):
 args=build_parser().parse_args(argv)
 if args.list_steps:
  print(json.dumps({'schema':'lucidota.system_graph_safety_audit.plan.v1','steps':[' '.join(s) for s in RUN_STEPS]},indent=2))
  return 0
 steps=[run(s,timeout_sec=args.timeout_seconds) for s in RUN_STEPS]
 blockers=[f"failed:{s['command']}" for s in steps if s['rc']!=0]
 payload={'action':'system_graph_safety_audit','timeout_seconds':args.timeout_seconds,'steps':steps,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'system_graph_safety_audit_{stamp()}.json'; payload['generated_at']=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+rel(p)); print('SYSTEM_GRAPH_SAFETY='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
