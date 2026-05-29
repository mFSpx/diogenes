#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/ci'
def ts():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try:return str(Path(p).resolve().relative_to(ROOT))
 except Exception:return str(p)
def run(cmd,timeout_sec:int):
 try:
  p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=None if timeout_sec<=0 else timeout_sec)
  stdout,stderr,rc=p.stdout,p.stderr,p.returncode
 except subprocess.TimeoutExpired as exc:
  stdout=(exc.stdout or '') if isinstance(exc.stdout,str) else (exc.stdout or b'').decode(errors='replace')
  stderr=(exc.stderr or '') if isinstance(exc.stderr,str) else (exc.stderr or b'').decode(errors='replace')
  stderr += f'\nTIMEOUT_EXPIRED_AFTER_SECONDS={timeout_sec}'
  rc=124
 rp=None
 for l in stdout.splitlines():
  if l.startswith('REPORT_PATH=') or l.startswith('RECEIPT_PATH='): rp=l.split('=',1)[1]
 return {'cmd':' '.join(cmd),'rc':rc,'report_path':rp,'stdout_tail':stdout[-1500:],'stderr_tail':stderr[-1500:]}
def step_commands():
 return [[sys.executable,'-m','py_compile','scripts/lucidota_mega_gate.py','scripts/cep_full_e2e.py','scripts/mega_gate_metrics_validator.py','scripts/tickletrunk_scan.py','scripts/lucidota_status_ledger.py','scripts/dev_order_matrix_wrapper.py','scripts/matrix_trace_checker.py','scripts/dev_order_gate.py','scripts/run_dev_order_methodology_checks.py','scripts/lucidota_acceptance.py','scripts/durable_workflow_decision_check.py','scripts/script_survival_coverage.py','scripts/absurd_worker_contracts.py','scripts/chrono_queue_event_bridge.py','scripts/graph_promotion_materialize.py','scripts/lucidota_strict_model_stack_admission.py','scripts/quality_work_order_compiler.py'],[sys.executable,'scripts/cep_full_e2e.py','--dry-run','--operator-instruction','CI golden path dry-run guard'],[sys.executable,'scripts/tickletrunk_scan.py','--execute'],[sys.executable,'scripts/tickletrunk_scan.py','--check'],[sys.executable,'scripts/lucidota_status_ledger.py','--check'],[sys.executable,'scripts/durable_workflow_decision_check.py'],[sys.executable,'scripts/script_survival_coverage.py'],[sys.executable,'scripts/run_dev_order_methodology_checks.py'],[sys.executable,'scripts/lucidota_acceptance.py','--self-fixture','--base-dir','05_OUTPUTS/ci_acceptance_cases','--case-id','ci-acceptance','--json'],[sys.executable,'scripts/lucidota_strict_model_stack_admission.py','--run-diogenes-gate','--json'],[sys.executable,'scripts/quality_work_order_compiler.py','--limit','10','--json'],[sys.executable,'-m','pytest','tests/test_graph_materialization_command_policy.py','tests/test_graph_promotion_gate_safety.py::test_boring_beast_direct_graph_materialize_default_refuses','-q'],[sys.executable,'scripts/boring_beast_full_e2e.py'],[sys.executable,'scripts/lucidota_mega_gate.py']]
def main():
 ap=argparse.ArgumentParser(description='Run the local LUCIDOTA CI gate and write an auditable receipt.')
 ap.add_argument('--json',action='store_true',help='Print the full receipt JSON in addition to gate summary lines.')
 ap.add_argument('--list-steps',action='store_true',help='Print planned child commands without executing them.')
 ap.add_argument('--timeout-sec',type=int,default=int(os.environ.get('LUCIDOTA_CI_GATE_TIMEOUT_SEC','900')),help='Per-child command timeout; use 0 to disable.')
 args=ap.parse_args()
 commands=step_commands()
 if args.list_steps:
  print(json.dumps({'schema':'lucidota.ci_gate.plan.v1','steps':[' '.join(c) for c in commands]},indent=2))
  return 0
 steps=[run(cmd,args.timeout_sec) for cmd in commands]
 blockers=[s['cmd'] for s in steps if s['rc']!=0]
 payload={'schema':'lucidota.ci_gate.v3','action':'lucidota_ci_gate','gate_scope':'local_product_gate_plus_architecture_hygiene_not_full_live_db_materialization','scope_notes':['validates local fixture product flow','validates TICKLETRUNK/status ledger/durable workflow decision hygiene','reports script survival coverage non-strict','does not prove full canonical graph materialization','does not prove full model runtime residency'],'timeout_sec':args.timeout_sec,'steps':steps,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}; OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'lucidota_ci_gate_{ts()}.json'; payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2),encoding='utf-8')
 if args.json: print(json.dumps(payload,sort_keys=True))
 print('REPORT_PATH='+rel(p)); print('LUCIDOTA_CI_GATE='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
