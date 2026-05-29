#!/usr/bin/env python3
"""Export one coding-only GOALS packet for Codex/Claude/OpenCode/Aider/Continue/llxprt-code/generic agents."""
from __future__ import annotations
import argparse,json,re
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/goals'
TIERS={'simple':'tiny_or_mini','repeat':'tiny_or_mini','standard':'standard_coding','integration':'standard_coding','architecture':'strong_or_frontier'}
ADAPTER_DEFAULTS={
 'groq':{'provider_type':'cloud','env_key_names':['GROQ_API_KEY'],'endpoint':'https://api.groq.com/openai/v1/chat/completions','dry_run_cmd':'python3 scripts/model_runner_cli.py groq-chat --prompt ping --max-tokens 8 --json','execute_cmd':'python3 scripts/model_runner_cli.py groq-chat --prompt @PROMPT --max-tokens 256 --json --execute','expected_receipt_glob':'05_OUTPUTS/model_invocations/groq_chat_*.json','stop_or_rollback_cmd':'none','safety_limits':['dry_run_default','secret_values_forbidden']},
 'cohere':{'provider_type':'cloud','env_key_names':['COHERE_API_KEY','CO_API_KEY'],'endpoint':'https://api.cohere.com/v2/chat','dry_run_cmd':'python3 scripts/model_runner_cli.py cohere-chat --prompt ping --max-tokens 8 --json','execute_cmd':'python3 scripts/model_runner_cli.py cohere-chat --prompt @PROMPT --max-tokens 256 --json --execute','expected_receipt_glob':'05_OUTPUTS/model_invocations/cohere_chat_*.json','stop_or_rollback_cmd':'none','safety_limits':['dry_run_default','secret_values_forbidden']},
 'needle_swarm_6x':{'provider_type':'local','env_key_names':[],'endpoint':'http://127.0.0.1:8090-8095/health','dry_run_cmd':'python3 scripts/goal_model_fabric_control.py status --json','execute_cmd':'scripts/lucidota_start_needle_swarm.sh','expected_receipt_glob':'05_OUTPUTS/goals/goal_model_fabric_control_*.json','stop_or_rollback_cmd':'python3 scripts/goal_model_fabric_control.py stop --target needles','safety_limits':['cpu_only','explicit_start']},
 'llama_cpp_heavy':{'provider_type':'local','env_key_names':[],'endpoint':'OpenAI-compatible llama.cpp servers on 8080-8083 when explicitly launched','dry_run_cmd':'python3 scripts/goal_model_fabric_control.py status --json','execute_cmd':'use existing lucidota_start_*_llama.sh launchers one lane at a time','expected_receipt_glob':'05_OUTPUTS/goals/goal_model_fabric_control_*.json','stop_or_rollback_cmd':'python3 scripts/goal_model_fabric_control.py stop --target heavy','safety_limits':['explicit_start','vram_guard','no_auto_heavy_daemon']}}
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def rel(p):
 try:return str(Path(p).resolve().relative_to(ROOT))
 except Exception:return str(p)
def read(p:Path)->str: return p.read_text(encoding='utf-8',errors='replace') if p.exists() else ''
def manifest(root:Path)->dict[str,Any]:
 try:return json.loads(read(root/'GOALS/plugin_build_mode_bootstrap.json'))
 except Exception:return {}
def lanes(m:dict, key:str)->list[str]: return [str(x.get('name')) for x in m.get(key,[]) if x.get('name')]
def adapter_def(m:dict,name:str)->dict[str,Any]: return dict(m.get('adapter_registry',{}).get(name) or ADAPTER_DEFAULTS.get(name) or ADAPTER_DEFAULTS['llama_cpp_heavy'])
def pick_adapter(task:str, complexity:str, m:dict)->dict[str,Any]:
 low=task.lower(); cloud=lanes(m,'cloud_lanes'); local=lanes(m,'local_lanes')
 if local:
  if ('needle' in low or complexity in {'simple','repeat'}) and 'needle_swarm_6x' in local: name='needle_swarm_6x'
  elif re.search(r'local|offline|vram|llama|mamba|bonsai|deepseek',low): name='llama_cpp_heavy'
  else: name='llama_cpp_heavy' if any(x in local for x in ['mamba7b_ram','bonsai4b_ram','deepseek_r1_qwen_1p5b_gpu']) else (local[0] if local else 'llama_cpp_heavy')
 else:
  if 'needle' in low: name='needle_swarm_6x'
  elif re.search(r'local|offline|vram|llama|mamba|bonsai|deepseek',low): name='llama_cpp_heavy'
  elif complexity in {'simple','repeat'}: name='groq'
  else: name='groq' if 'groq' in cloud else ('cohere' if 'cohere' in cloud else 'llama_cpp_heavy')
 return {'selected':name, **adapter_def(m,name)}
def build_prompt(task:str, files:list[str], checks:list[str])->dict[str,Any]:
 return {'role':'coding_worker','task':task,'file_ownership':files or ['repo-scoped-read-only-first'],'inputs':['GOALS/CURRENT_HANDOFF.md','GOALS/AGENT_ORCHESTRATION_POLICY.md','GOALS/plugin_build_mode_bootstrap.json'],'constraints':'Coding only. Do not revert unrelated edits. Do not change the main-window model. Do not print secrets. Keep edits surgical. Return proof, not vibes.','acceptance_checks':checks or ['run focused tests','name receipt paths'],'output_contract':{'schema':'lucidota.worker_order.v1','envelope_style':'single_exact_top_level_json_object','required_output':['status','result','next_action','receipt_path','evidence_refs','decision_pairs'],'forbidden_output':['nested_result_object','prose','meta-summary','hidden-reasoning'],'min_decision_pairs':2,'evidence_required':True,'recommended_max_tokens_floor':256,'recommended_max_tokens_ceiling':512,'parse_rule':'If any field is missing, emit the key with null or [] rather than nesting it under result.'},'return_contract':'Return changed files, commands run, pass/fail output, receipt paths, blockers, and next smallest step.'}
def build_packet(root:Path=ROOT, target:str='generic', task:str='', files:list[str]|None=None, complexity:str='standard', checks:list[str]|None=None)->dict[str,Any]:
 root=Path(root); m=manifest(root); handoff=read(root/'GOALS/CURRENT_HANDOFF.md'); task=task or re.sub(r'\s+',' ',handoff[-500:]).strip() or 'read CURRENT_HANDOFF and propose next coding slice'
 adapter=pick_adapter(task,complexity,m); tier=TIERS.get(complexity,'standard_coding'); blockers=[]
 if re.search(r'change|switch|set',task,re.I) and re.search(r'main[- ]window model|main model',task,re.I): blockers.append('main_window_model_change_forbidden')
 return {'schema':'lucidota.goals.agent_packet.v1','generated_at':now(),'target':target,'task':task,'handoff_ref':'GOALS/CURRENT_HANDOFF.md','model_policy':{'selected_tier':tier,'cheapest_capable':True,'reasoning_split':'frontier/high only for architecture; mid for contained coding; tiny/local/deterministic for repeat production checks','main_window_model_change':'forbidden_without_explicit_operator_tool','blocked_operations':['change main-window model','print secrets','revert unrelated edits'],'real_model_names_require_current_verification':True,'output_contract':{'schema':'lucidota.worker_order.v1','envelope_style':'single_exact_top_level_json_object','required_output':['status','result','next_action','receipt_path','evidence_refs','decision_pairs'],'forbidden_output':['nested_result_object','prose','meta-summary','hidden-reasoning'],'decision_pairs_min':2,'evidence_required':True,'recommended_max_tokens_floor':256,'recommended_max_tokens_ceiling':512,'parse_rule':'If any field is missing, emit the key with null or [] rather than nesting it under result.'}},'adapter':adapter,'adapters':{'local_lanes':lanes(m,'local_lanes'),'cloud_lanes':lanes(m,'cloud_lanes'),'registry_source':'GOALS/plugin_build_mode_bootstrap.json'},'coding_prompt':build_prompt(task,files or [],checks or []),'proof_policy':'No completion claim without changed files + commands + receipts. Operator slop allowed; orchestrator slop rejected.','blockers':blockers,'model_calls_performed':False,'canonical_graph_writes_performed':False}
def write(pkt:dict)->dict:
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'goal_agent_packet_{stamp()}.json'; pkt['report_path']=rel(p); p.write_text(json.dumps(pkt,indent=2,sort_keys=True)+'\n'); return pkt
def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('--target',default='generic'); ap.add_argument('--task',default=''); ap.add_argument('--file',action='append',default=[]); ap.add_argument('--complexity',choices=sorted(TIERS),default='standard'); ap.add_argument('--check',action='append',default=[]); ap.add_argument('--json',action='store_true'); a=ap.parse_args()
 pkt=write(build_packet(target=a.target,task=a.task,files=a.file,complexity=a.complexity,checks=a.check)); print('REPORT_PATH='+pkt['report_path']); print('GOAL_AGENT_PACKET=PASS'); print(json.dumps(pkt,sort_keys=True) if a.json else 'TARGET='+pkt['target']); return 0
if __name__=='__main__': raise SystemExit(main())
