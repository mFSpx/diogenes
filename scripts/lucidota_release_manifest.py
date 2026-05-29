#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/release'
CRITICAL=['scripts/lucidota_mega_gate.py','scripts/lucidota_ci_gate.py','scripts/tickletrunk_scan.py','scripts/lucidota_status_ledger.py','scripts/boring_beast.py','scripts/dev_order_matrix_wrapper.py','scripts/matrix_trace_checker.py','scripts/dev_order_gate.py','scripts/run_dev_order_methodology_checks.py','06_SCHEMA/dev_order_matrix_policy.v1.json','06_SCHEMA/035_absurd_queue_spine.sql','06_SCHEMA/034_graph_promotion_pipeline.sql']
def ts(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def sha(path):
    h=hashlib.sha256(); p=ROOT/path
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()
def git(args):
    try: return subprocess.check_output(['git']+args,cwd=ROOT,text=True,stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc: return exc.output.strip()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--execute',action='store_true'); args=ap.parse_args()
    files=[]; missing=[]
    for item in CRITICAL:
        p=ROOT/item
        if p.exists(): files.append({'path':item,'sha256':sha(item),'size_bytes':p.stat().st_size})
        else: missing.append(item)
    dirty=git(['status','--short']).splitlines()
    payload={'schema':'lucidota.release_manifest.v1','generated_at':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'execute_performed':args.execute,'status':'PASS' if not missing else 'FAIL','git_head':git(['rev-parse','HEAD']),'git_branch':git(['branch','--show-current']),'dirty_count':len(dirty),'dirty_files':dirty[:500],'critical_files':files,'missing_critical_files':missing,'release_ready':not missing,'blockers':missing,'canonical_graph_materialization':False,'canonical_graph_writes':False}
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'lucidota_release_manifest_{ts()}.json'; payload['report_path']=rel(p)
    if args.execute: p.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('REPORT_PATH='+rel(p)); print('RELEASE_MANIFEST='+('PASS' if not missing else 'FAIL'))
    return 0 if not missing else 3
if __name__=='__main__': raise SystemExit(main())
