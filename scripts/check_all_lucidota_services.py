#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/services'
def ts(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def run(cmd):
    proc=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    return {'command':' '.join(cmd),'returncode':proc.returncode,'stdout':proc.stdout[-8000:],'stderr':proc.stderr[-8000:]}
def chrono_status():
    script=ROOT/'scripts/check_chrono_ledger_service.sh'
    if not script.exists(): return {'name':'chrono-ledger','status':'script_missing','healthy':False,'evidence':None}
    res=run([str(script)])
    return {'name':'chrono-ledger','status':'healthy' if res['returncode']==0 else 'unhealthy','healthy':res['returncode']==0,'evidence':res}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--execute',action='store_true',help='run service probes'); args=ap.parse_args()
    services=[chrono_status()] if args.execute else [{'name':'chrono-ledger','status':'not_checked','healthy':None,'evidence':'use --execute'}]
    blockers=[s['name']+':'+s['status'] for s in services if s['healthy'] is False]
    payload={'schema':'lucidota.service_check_all.v1','generated_at':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'execute_performed':args.execute,'services':services,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'check_all_lucidota_services_{ts()}.json'; payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('REPORT_PATH='+rel(p)); print('SERVICE_CHECK_ALL='+payload['status']); return 0 if payload['status']=='PASS' else 5
if __name__=='__main__': raise SystemExit(main())
