#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, os
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/deploy'
def ts(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def run(cmd):
    proc=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    return {'command':' '.join(cmd),'returncode':proc.returncode,'stdout_tail':proc.stdout[-4000:],'stderr_tail':proc.stderr[-4000:]}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--execute',action='store_true',help='still non-mutating; executes validation probes only'); args=ap.parse_args()
    checks=[]
    for path in ['scripts/install_chrono_ledger_service.sh','scripts/check_chrono_ledger_service.sh','scripts/lucidota_ci_gate.py','scripts/lucidota_mega_gate.py','scripts/run_dev_order_methodology_checks.py']:
        p=ROOT/path; checks.append({'path':path,'exists':p.exists(),'executable':os.access(p,os.X_OK) if p.exists() else False})
    probes=[]
    if args.execute:
        probes.append(run(['python3','scripts/lucidota_status_ledger.py','--check']))
        probes.append(run(['python3','scripts/tickletrunk_scan.py','--check']))
        probes.append(run(['python3','scripts/run_dev_order_methodology_checks.py']))
    blockers=[c['path'] for c in checks if not c['exists']]
    payload={'schema':'lucidota.deploy_dry_run.v1','generated_at':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'execute_performed':args.execute,'canonical_mutation_performed':False,'canonical_graph_materialization':False,'canonical_graph_writes':False,'daemon_restart_performed':False,'checks':checks,'probes':probes,'blockers':blockers,'status':'PASS' if not blockers and all(p.get('returncode',0)==0 for p in probes) else 'FAIL'}
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'lucidota_deploy_dry_run_{ts()}.json'; payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('REPORT_PATH='+rel(p)); print('DEPLOY_DRY_RUN='+payload['status']); return 0 if payload['status']=='PASS' else 4
if __name__=='__main__': raise SystemExit(main())
