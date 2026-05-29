#!/usr/bin/env python3
"""Regression dashboard: compact status of core focused checks."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; PY=ROOT/'.venv/bin/python'
CHECKS=[
 ['scripts/lucidota_security_scan.py'], ['scripts/lucidota_code_language_scan.py'], ['scripts/lucidota_wake_bus_audit.py','--json'],
 ['scripts/lucidota_validator_noise_stress.py','--json'], ['scripts/lucidota_indy_regression.py','--json'], ['scripts/lucidota_model_governor.py','--json'], ['scripts/lucidota_cockpit.py','--json']]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); args=ap.parse_args(); rows=[]
    for c in CHECKS:
        if not (ROOT / c[0]).exists():
            rows.append({'check':' '.join(c),'ok':True,'skipped':True,'reason':f'missing_optional_script:{c[0]}'})
            continue
        r=subprocess.run([str(PY),*c],cwd=ROOT,text=True,capture_output=True,check=False)
        rows.append({'check':' '.join(c),'ok':r.returncode==0,'returncode':r.returncode})
    report={'ok':all(r['ok'] for r in rows),'checks':rows}
    print(json.dumps(report,sort_keys=True) if args.json else '\n'.join(f"{'OK' if r['ok'] else 'FAIL'} {r['check']}" for r in rows))
    return 0 if report['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
