#!/usr/bin/env python3
"""Tiny system/function index for GOALS dev mode."""
from __future__ import annotations
import argparse,ast,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/goals'
SKIP={'.git','.venv','__pycache__','node_modules','target'}
DEFAULT=['GOALS','scripts','ALGOS','core','services','06_SCHEMA']
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def rel(p:Path,root:Path)->str:
 try: return str(p.resolve().relative_to(root.resolve()))
 except Exception: return str(p)
def py_funcs(p:Path,root:Path)->list[str]:
 try: tree=ast.parse(p.read_text(encoding='utf-8',errors='ignore'))
 except Exception: return []
 return [f"{rel(p,root)}:{n.name}" for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
def iter_files(root:Path,dirs:list[str]):
 for d in dirs:
  base=root/d
  if not base.exists(): continue
  for p in ([base] if base.is_file() else base.rglob('*')):
   if p.is_file() and not any(part in SKIP for part in p.parts): yield p
def build_index(root:Path=ROOT,dirs:list[str]|None=None)->dict:
 root=Path(root); dirs=dirs or DEFAULT; systems=[]; funcs=[]; files=0
 for d in dirs:
  base=root/d; paths=list(iter_files(root,[d])); files+=len(paths); systems.append({'path':d,'exists':base.exists(),'file_count':len(paths)})
  for p in paths:
   if p.suffix=='.py': funcs+=py_funcs(p,root)
 return {'schema':'lucidota.goals.system_function_index.v1','generated_at':now(),'systems':systems,'file_count':files,'function_count':len(funcs),'functions':sorted(funcs),'dirs':dirs}
def write(r:dict)->Path:
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/'system_function_index_latest.json'; p.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n'); return p
def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('--dir',action='append'); ap.add_argument('--json',action='store_true'); a=ap.parse_args(); r=build_index(dirs=a.dir); p=write(r)
 print('REPORT_PATH='+str(p.relative_to(ROOT))); print('SYSTEM_FUNCTION_INDEX=PASS'); print(f"SYSTEMS={len(r['systems'])} FILES={r['file_count']} FUNCTIONS={r['function_count']}")
 if a.json: print(json.dumps(r,sort_keys=True))
 return 0
if __name__=='__main__': raise SystemExit(main())
