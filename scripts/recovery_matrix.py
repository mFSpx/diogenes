#!/usr/bin/env python3
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/recovery'
def ts():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try:return str(Path(p).resolve().relative_to(ROOT))
 except Exception:return str(p)
def main():
 actions=[
 {'key':'absurd_stale_lock_recovery','risk':'medium','execute_allowed':True,'script':'scripts/spine_queue_soak_test.py'},
 {'key':'chrono_projection_refresh','risk':'low','execute_allowed':True,'script':'scripts/chrono_timeline_projection_refresh.py'},
 {'key':'graph_orphan_defer','risk':'high','execute_allowed':False,'script':'scripts/graph_promotion_orphan_detector.py'},
 {'key':'tickletrunk_regenerate','risk':'low','execute_allowed':True,'script':'scripts/tickletrunk_scan.py'},
 {'key':'goal_handoff_check','risk':'low','execute_allowed':True,'script':'scripts/goal_handoff.py --root GOALS check'},
 {'key':'goal_dev_control_check','risk':'low','execute_allowed':True,'script':'scripts/goal_dev_control.py --away-minutes 0 --text recovery'},
 {'key':'goal_agent_packet','risk':'low','execute_allowed':True,'script':'scripts/goal_agent_packet.py --target generic --task recovery --complexity simple --json'},
 {'key':'goal_swarm_dispatch','risk':'medium','execute_allowed':True,'script':'scripts/goal_swarm_dispatch.py --target generic --task recovery --jobs 1 --json'},
 {'key':'goal_model_fabric_status','risk':'low','execute_allowed':True,'script':'scripts/goal_model_fabric_control.py status'},
 {'key':'goal_model_fabric_start_needles','risk':'low','execute_allowed':True,'script':'scripts/goal_model_fabric_control.py start --target needles --json'},
 {'key':'goal_model_fabric_stop_heavy','risk':'medium','execute_allowed':True,'script':'scripts/goal_model_fabric_control.py stop --target heavy'},
 {'key':'goal_chain_check','risk':'low','execute_allowed':True,'script':'scripts/goal_chain.py check'},
 {'key':'goal_system_index','risk':'low','execute_allowed':True,'script':'scripts/goal_system_index.py'},
 {'key':'goal_telemetry_snapshot','risk':'low','execute_allowed':True,'script':'scripts/goal_telemetry.py snapshot'},
 {'key':'no_delete_guard','risk':'medium','execute_allowed':True,'script':'scripts/no_delete_guard.py check'},
 {'key':'language_router_smoke','risk':'low','execute_allowed':True,'script':'scripts/language_router.py --text recovery --json'},
 {'key':'operator_language_smoke','risk':'low','execute_allowed':True,'script':'scripts/lucidota_cli.py language --text recovery --json'},
 {'key':'runtime_smoke','risk':'low','execute_allowed':True,'script':'.venv/bin/python scripts/lucidota_runtime_smoke.py'}]
 payload={'action':'recovery_matrix','actions':actions,'blockers':[],'status':'PASS'}; OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'recovery_matrix_{ts()}.json'; payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+rel(p)); print('RECOVERY_MATRIX=PASS'); return 0
if __name__=='__main__': raise SystemExit(main())
