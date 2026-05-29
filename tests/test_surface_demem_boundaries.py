#!/usr/bin/env python3
"""Surface instruction acceptance tests against DeMem boundaries."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_OUTPUTS/tests'; OUT.mkdir(parents=True, exist_ok=True)
def run(cmd): return subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
def report_path(stdout):
 for line in stdout.splitlines():
  if line.startswith('REPORT_PATH='): return line.split('=',1)[1]
 return None
def main():
 results=[]
 init=run([sys.executable,'scripts/demem_runtime_guard.py','init-schema','--execute']); results.append({'name':'demem_init','rc':init.returncode,'stdout_tail':init.stdout[-1000:],'stderr_tail':init.stderr[-1000:]})
 compile_cmd=[sys.executable,'scripts/surface_instruction_compile_dry_run.py','--surface-id','demem_boundary_test','--operator-action','selected','--target-ref','lucidota_go.graph_item:test','--evidence-refs','tests/test_surface_demem_boundaries.py','--artifact-refs','07_SURFACES/generated/marrow_loop_status.html','--current-object-state','{"generated":"surface"}','--allowed-effect','direct canonical graph state mutation requested by generated surface','--dry-run']
 comp=run(compile_cmd); comp_path=report_path(comp.stdout); results.append({'name':'compile','rc':comp.returncode,'report_path':comp_path})
 data=json.loads((ROOT/comp_path).read_text()) if comp_path else {}; instr=data.get('plain_language_instruction','')
 guard=run([sys.executable,'scripts/demem_runtime_guard.py','check','--instruction',instr,'--source-ref','surface_demem_boundary_test','--execute']); guard_path=report_path(guard.stdout); results.append({'name':'guard','rc':guard.returncode,'report_path':guard_path,'stdout_tail':guard.stdout[-1000:]})
 guard_data=json.loads((ROOT/guard_path).read_text()) if guard_path else {}; ok=(comp.returncode==0 and guard.returncode in (0,2) and guard_data.get('decision') in ('warn','rewrite','block') and data.get('canonical_mutation_allowed') is False)
 payload={'status':'PASS' if ok else 'FAIL','compile_report':comp_path,'demem_report':guard_path,'guard_decision':guard_data.get('decision'),'results':results}
 out=OUT/'surface_demem_boundaries_result.json'; out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
 print('REPORT_PATH='+str(out.relative_to(ROOT))); print('SURFACE_DEMEM_BOUNDARIES='+payload['status']); return 0 if ok else 4
if __name__=='__main__': raise SystemExit(main())
