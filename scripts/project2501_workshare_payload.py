#!/usr/bin/env python3
"""Emit Project 2501 GO-25 workshare JzLOADS and receipt."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ALGOS.workshare_allocator import allocate_workshare, summarize_savings
OUT=ROOT/'05_OUTPUTS/model_admin_prompt'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def main()->int:
 ap=argparse.ArgumentParser(description='Emit strict GO-25 Project 2501 workshare payload.')
 ap.add_argument('--total-units',type=float,default=100.0)
 ap.add_argument('--deterministic-target-pct',type=float,default=90.0)
 ap.add_argument('--json',action='store_true')
 a=ap.parse_args()
 plan=allocate_workshare(total_units=a.total_units,deterministic_target_pct=a.deterministic_target_pct)
 receipt={'schema':'lucidota.project2501.workshare_receipt.v1','generated_at':now(),'allocation':plan,'savings':summarize_savings(total_units=a.total_units,deterministic_target_pct=a.deterministic_target_pct),'canonical_graph_writes_performed':False,'jzloads_kinds':[x['kind'] for x in plan['jzloads']]}
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'project2501_workshare_{stamp()}.json'; receipt['report_path']=rel(p); p.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 if a.json: print(json.dumps(receipt,sort_keys=True))
 print('REPORT_PATH='+rel(p)); print('PROJECT2501_WORKSHARE=PASS')
 return 0
if __name__=='__main__': raise SystemExit(main())
