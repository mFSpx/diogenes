#!/usr/bin/env python3
"""Integrated Surfaces/Marrow/CEP command loop proof."""
from __future__ import annotations
import argparse,json,subprocess,sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/marrow_loop'
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def run(cmd):
 p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True); d={'cmd':' '.join(cmd),'rc':p.returncode,'stdout_tail':p.stdout[-1500:],'stderr_tail':p.stderr[-1500:]}
 for line in p.stdout.splitlines():
  if '=' in line: d[line.split('=',1)[0]]=line.split('=',1)[1]
 return d
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--execute',action='store_true'); a=ap.parse_args(); blockers=[]; steps=[]
 if not a.execute: blockers.append('execute_required')
 else:
  receipt=run([sys.executable,'scripts/diogenes_stub_kernel.py','--raw-command','Round 95 integrated surfaces marrow cep proof','--normalized-intent','system.integrated_loop_proof','--authority-class','operator_authored_assertion','--source','operator_cli']); steps.append(receipt)
  rp=receipt.get('RECEIPT_PATH')
  if not rp: blockers.append('receipt_missing')
  if rp:
   apply=run([sys.executable,'scripts/marrow_loop_apply.py','--receipt',rp,'--execute']); steps.append(apply)
  render=run([sys.executable,'scripts/marrow_loop_render_surface.py','--state','05_OUTPUTS/marrow_loop/marrow_state.md']); steps.append(render)
  comp=run([sys.executable,'scripts/surface_instruction_compile_dry_run.py','--surface-id','marrow_loop_status','--operator-action','refined','--target-ref','marrow_loop:latest','--evidence-refs',rp or 'missing_receipt','--artifact-refs','07_SURFACES/generated/marrow_loop_status.html','--current-object-state','{"round":95}','--allowed-effect','queue ABSURD work order only','--execute','--queue-absurd']); steps.append(comp)
  if comp.get('COMMAND_UUID') is None: blockers.append('conversation_command_missing')
  if comp.get('ABSURD_JOB_UUID') is None: blockers.append('absurd_job_missing')
  verify=run([sys.executable,'scripts/marrow_state_append_only_verify.py','--record']); steps.append(verify)
  for s in steps:
   if s['rc']!=0: blockers.append(f"step_failed:{s['cmd']}:rc={s['rc']}")
 payload={'action':'surfaces_marrow_cep_loop_proof','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'steps':steps,'receipt_path':steps[0].get('RECEIPT_PATH') if steps else None,'conversation_command_uuid':steps[-2].get('COMMAND_UUID') if len(steps)>2 else None,'absurd_job_uuid':steps[-2].get('ABSURD_JOB_UUID') if len(steps)>2 else None,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'surfaces_marrow_cep_loop_proof_{stamp()}.json'; payload['generated_at']=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+rel(p)); print('SURFACES_MARROW_CEP_LOOP='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
