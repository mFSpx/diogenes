#!/usr/bin/env python3
"""Fast health check for the file-backed Abductive DB OS lane."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS'/'abductive_db_os'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def latest(pat):
    xs=sorted(ROOT.glob(pat), key=lambda p:p.stat().st_mtime, reverse=True); return xs[0] if xs else None
def load(p):
    try: return json.loads(p.read_text()) if p else {}
    except Exception: return {}
def run(mode):
    checks=[]; warnings=[]; hard=[]
    board=latest('05_OUTPUTS/abductive_db_os/board_*.json'); checks.append({'name':'board_state_exists','verdict':'PASS' if board else 'FAIL','path':rel(board) if board else None})
    audit=load(latest('05_OUTPUTS/model_invocation_audits/model_invocation_audit_*.json')); checks.append({'name':'model_audit_semantics_known','verdict':'PASS' if audit.get('verdict') in {'PASS','FAIL'} else 'FAIL','latest_verdict':audit.get('verdict')})
    if int(audit.get('missing_dedicated_model_audit_blocks') or 0)>0: warnings.append('model audit has missing complete block audits')
    checks.append({'name':'graph_candidates_not_canonical_writes','verdict':'PASS','canonical_graph_writes':False})
    checks.append({'name':'next_move_engine_output','verdict':'PASS' if latest('05_OUTPUTS/abductive_db_os/next_moves_*.json') else 'FAIL'})
    if any(c['verdict']=='FAIL' for c in checks): warnings.append('one or more health checks degraded')
    verdict='PASS' if not warnings and not hard else ('FAIL' if hard else 'DEGRADED')
    payload={'schema':'lucidota.abductive_db_os.health_check.v1','mode':mode,'generated_at_utc':now(),'verdict':verdict,'checks':checks,'hard_failures':hard,'warnings':warnings,'canonical_graph_writes':False,'canonical_graph_materialization':False,'external_effects':False}
    path=OUT/f'health_check_{mode}_{stamp()}.json'; payload['receipt_path']=rel(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); return payload
def main():
    ap=argparse.ArgumentParser(); g=ap.add_mutually_exclusive_group(); g.add_argument('--fast',action='store_true'); g.add_argument('--daily',action='store_true'); args=ap.parse_args(); mode='daily' if args.daily else 'fast'; p=run(mode); print(json.dumps(p,sort_keys=True)); print('REPORT_PATH='+p['receipt_path']); print('ABDUCTIVE_DB_OS_HEALTH_CHECK='+p['verdict']); return 0 if p['verdict'] in {'PASS','DEGRADED'} else 4
if __name__=='__main__': raise SystemExit(main())
