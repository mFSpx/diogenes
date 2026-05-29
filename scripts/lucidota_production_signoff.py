#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, os
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/production'
def ts(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def latest(pattern):
    xs=glob.glob(str(ROOT/pattern))
    if 'lucidota_mega_gate_*.json' in pattern:
        xs=[x for x in xs if 'validate_report' not in Path(x).name]
    return Path(max(xs,key=os.path.getmtime)) if xs else None
def read_status(path):
    if not path: return None
    try:
        data=json.loads(path.read_text())
        if data.get('status'):
            return data.get('status')
        if data.get('readiness_status'):
            return data.get('readiness_status')
        if data.get('verdict'):
            return data.get('verdict')
        if data.get('validation_result'):
            return data.get('validation_result')
        if data.get('release_ready') is True and not data.get('blockers'):
            return 'PASS'
        return None
    except Exception: return None
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--sign-off',action='store_true'); ap.add_argument('--operator-id',default='operator'); args=ap.parse_args()
    evidence={
      'ci':latest('05_OUTPUTS/ci/lucidota_ci_gate_*.json'),
      'mega_gate':latest('05_OUTPUTS/mega_gate/lucidota_mega_gate_*.json'),
      'readiness':latest('05_OUTPUTS/production/production_readiness_eval_*.json'),
      'services':latest('05_OUTPUTS/services/check_all_lucidota_services_*.json'),
      'release_manifest':latest('05_OUTPUTS/release/lucidota_release_manifest_*.json'),
      'deploy_dry_run':latest('05_OUTPUTS/deploy/lucidota_deploy_dry_run_*.json'),
      'dev_order_methodology':latest('05_OUTPUTS/dev_order_matrix/dev_order_methodology_*.json'),
    }
    blockers=[]
    for k,p in evidence.items():
        if not p: blockers.append(k+'_missing')
        elif read_status(p) not in {'PASS','pass','signed_off'}: blockers.append(k+'_not_pass')
    status='SIGNED_OFF' if args.sign_off and not blockers else 'READY_FOR_SIGNOFF' if not blockers else 'BLOCKED'
    payload={'schema':'lucidota.production_signoff.v1','generated_at':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'sign_off_requested':args.sign_off,'operator_id':args.operator_id,'status':status,'evidence':{k:rel(p) if p else None for k,p in evidence.items()},'blockers':blockers,'canonical_mutation_performed':False,'canonical_graph_materialization':False,'canonical_graph_writes':False}
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'lucidota_production_signoff_{ts()}.json'; payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('REPORT_PATH='+rel(p)); print('PRODUCTION_SIGNOFF='+status); return 0 if not blockers else 6
if __name__=='__main__': raise SystemExit(main())
