#!/usr/bin/env python3
"""One-screen LUCIDOTA cockpit: bars + Indy_Reads + model governor."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY=ROOT/'.venv'/'bin'/'python'
if not PY.exists(): PY=Path(sys.executable)

def j(script,*args):
    r=subprocess.run([str(PY), str(ROOT/'scripts'/script), *args, '--json'], text=True, capture_output=True, check=False)
    try: data=json.loads(r.stdout)
    except Exception: data={'ok':False,'error':r.stderr[-200:] or r.stdout[-200:]}
    data['_returncode']=r.returncode
    return data

def main():
    ap=argparse.ArgumentParser(prog='lucidota-cockpit'); ap.add_argument('--json', action='store_true'); args=ap.parse_args()
    report={'ok': True, 'big_board': j('lucidota_big_board.py'), 'indy': j('lucidota_indy_brief.py'), 'model_governor': j('lucidota_model_governor.py')}
    report['ok']=all(x.get('ok') for x in report.values() if isinstance(x,dict) and 'ok' in x)
    if args.json: print(json.dumps(report, sort_keys=True)); return 0 if report['ok'] else 1
    print('LUCIDOTA COCKPIT')
    print('================')
    print('Overall:', report['big_board'].get('bars',{}).get('overall','unknown'))
    print('Indy:', report['indy'].get('indy_phase',{}).get('bar','unknown'), 'queue=', report['indy'].get('counters',{}).get('quiet_queue'))
    mg=report['model_governor']; print('Model governor:', mg.get('decision'), mg.get('rationale'))
    print('GPU:', mg.get('gpu',{}))
    return 0 if report['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
