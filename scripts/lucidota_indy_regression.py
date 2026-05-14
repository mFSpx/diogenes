#!/usr/bin/env python3
"""Indy_Reads regression smoke: brief shape + correction loop."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
import psycopg
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DB=os.environ.get('DBOS_SYSTEM_DATABASE_URL','postgresql://mfspx@/lucidota_state')
PY=ROOT/'.venv'/'bin'/'python'
if not PY.exists(): PY=Path(sys.executable)

def run(args):
    return subprocess.run([str(PY), str(ROOT/'scripts'/'lucidota_indy_brief.py'), *args], text=True, capture_output=True, check=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--json', action='store_true'); args=ap.parse_args()
    brief=run(['--json'])
    ok=brief.returncode==0
    data=json.loads(brief.stdout) if ok else {}
    checks={
      'identity': data.get('identity')=='Indy_Reads',
      'citations': len(data.get('citations',[])) >= 4,
      'counters': 'workflow_events' in data.get('counters',{}),
      'queue_present': 'queue' in data,
      'auth_present': 'auth' in data,
    }
    corr=run(['correct','Regression correction loop smoke','--body','operator correction captured without changing command flow','--source','regression','--json'])
    correction_id=json.loads(corr.stdout).get('memory_id') if corr.returncode==0 else None
    checks['correction_insert']=bool(correction_id)
    if correction_id:
        with psycopg.connect(DB) as conn:
            conn.execute("UPDATE lucidota_indy.task_memory SET status='archived', updated_at=now() WHERE memory_id=%s::uuid", (correction_id,))
            conn.commit()
    checks['correction_archive']=bool(correction_id)
    report={'ok': all(checks.values()), 'checks': checks}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0 if report['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
