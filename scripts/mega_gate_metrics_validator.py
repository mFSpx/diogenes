#!/usr/bin/env python3
"""Validate Mega-Gate v2 report metrics and repair coverage."""
from __future__ import annotations
import argparse,json
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/mega_gate'
REPAIRS={"compile_critical_gate_scripts_first","run_tickletrunk_execute_before_check","parse_child_json_reports_and_validate_status","enforce_cross_system_invariants","require_report_paths_for_report_steps","emit_component_metrics_summary"}
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--report',required=True); a=ap.parse_args(); p=ROOT/a.report if not Path(a.report).is_absolute() else Path(a.report); blockers=[]
 try: d=json.loads(p.read_text())
 except Exception as e: d={}; blockers.append('REPORT_UNREADABLE:'+str(e))
 metrics=d.get('metrics') or {}; repairs=set(d.get('repairs_applied') or [])
 if d.get('schema')!='lucidota.mega_gate.v2': blockers.append('SCHEMA_NOT_V2')
 if d.get('status')!='PASS': blockers.append('STATUS_NOT_PASS')
 if d.get('blockers'): blockers.append('REPORT_BLOCKERS_PRESENT')
 if REPAIRS-repairs: blockers.append('REPAIRS_MISSING:'+','.join(sorted(REPAIRS-repairs)))
 if metrics.get('steps_total')!=metrics.get('steps_passed'): blockers.append('STEPS_NOT_ALL_PASSED')
 golden=metrics.get('golden_path') or {}
 if golden.get('status') not in {'PASS_DRY_RUN','PASS'}: blockers.append('GOLDEN_PATH_NOT_PASSED')
 if golden.get('canonical_graph_writes_performed') is not False: blockers.append('GOLDEN_PATH_CANONICAL_GRAPH_WRITE')
 if golden.get('graph_writes_performed') is not False: blockers.append('GOLDEN_PATH_GRAPH_WRITE')
 if golden.get('db_writes_performed') is not False: blockers.append('GOLDEN_PATH_DB_WRITE')
 if int(golden.get('receipt_refs') or 0)<3: blockers.append('GOLDEN_PATH_RECEIPTS_INSUFFICIENT')
 if metrics.get('chrono_ranking_violations')!=0: blockers.append('CHRONO_RANKING_NONZERO')
 soak=metrics.get('absurd_soak') or {}
 if int(soak.get('inserted_new') or 0)<1 or int(soak.get('duplicates_reused') or 0)<1 or int(soak.get('succeeded') or 0)<1: blockers.append('ABSURD_SOAK_METRICS_INSUFFICIENT')
 payload={'action':'mega_gate_metrics_validate','validated_report':rel(p),'metrics':metrics,'repairs_present':sorted(repairs),'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 OUT.mkdir(parents=True,exist_ok=True); out=OUT/f'mega_gate_regression_summary_{stamp()}.json'; payload['generated_at']=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'); payload['report_path']=rel(out); out.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+rel(out)); print('MEGA_GATE_METRICS='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
