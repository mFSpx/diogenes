#!/usr/bin/env python3
"""End-to-end operator demo script: cockpit, Survey file, model governor."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; PY=ROOT/'.venv/bin/python'

def run(args):
    r=subprocess.run([str(PY),*args],cwd=ROOT,text=True,capture_output=True,check=False)
    return {'ok':r.returncode==0,'returncode':r.returncode,'stdout_tail':r.stdout[-800:],'stderr_tail':r.stderr[-400:]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); args=ap.parse_args()
    steps={
      'cockpit': run(['scripts/lucidota_cockpit.py','--json']),
      'survey_file': run(['scripts/lucidota_survey.py','BRAIN.md','--fetch','--keyword','LUCIDOTA']),
      'model_governor': run(['scripts/lucidota_model_governor.py','--json']),
      'indy_brief': run(['scripts/lucidota_indy_brief.py','--json']),
    }
    report={'ok':all(s['ok'] for s in steps.values()),'steps':steps}
    print(json.dumps(report,sort_keys=True) if args.json else report)
    return 0 if report['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
