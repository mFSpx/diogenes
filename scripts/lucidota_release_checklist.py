#!/usr/bin/env python3
"""Release checklist gate: local verification signals only."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def cmd(args):
    r=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,check=False)
    return {'ok':r.returncode==0,'returncode':r.returncode,'stdout_tail':r.stdout[-500:],'stderr_tail':r.stderr[-500:]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); args=ap.parse_args()
    checks={
      'git_clean': cmd(['git','diff','--quiet']),
      'security_scan': cmd([str(ROOT/'.venv/bin/python'),'scripts/lucidota_security_scan.py']),
      'language_scan': cmd([str(ROOT/'.venv/bin/python'),'scripts/lucidota_code_language_scan.py']),
      'cockpit_json': cmd([str(ROOT/'.venv/bin/python'),'scripts/lucidota_cockpit.py','--json']),
    }
    # git_clean may be false during active dev; release gate reports, not fails active harness.
    ok=all(v['ok'] for k,v in checks.items() if k!='git_clean')
    report={'ok':ok,'checks':checks,'release_ready':ok and checks['git_clean']['ok']}
    print(json.dumps(report,sort_keys=True) if args.json else report)
    return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())
