#!/usr/bin/env python3
"""Bounded proof harness for case, campaign, code, and hypertimeline use-cases."""
from __future__ import annotations
import argparse,json,subprocess,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'scripts'):
 if str(p) not in sys.path: sys.path.insert(0,str(p))
from language_router import route_text  # noqa:E402
from percyphon_kernel_bridge import build_bridge as build_percyphon_bridge  # noqa:E402
import lucidota_anthropic_llama_proxy as anthropic_proxy  # noqa:E402
from spine_common import rel, sha256_json, write_json  # noqa:E402

def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def out_path(root:Path,*parts:str)->Path:
 p=root.joinpath(*parts); p.parent.mkdir(parents=True,exist_ok=True); return p

def marker(stdout:str)->dict[str,str]:
 m={}
 for line in stdout.splitlines():
  if line.startswith(('REPORT_PATH=','RECEIPT_PATH=','LEDGER_PATH=')):
   k,v=line.split('=',1); m[k.lower()]=v
 return m

def run_cmd(cmd:list[str], timeout:int)->dict[str,Any]:
 try:
  p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=timeout)
  return {'cmd':' '.join(cmd),'rc':p.returncode,'stdout_tail':p.stdout[-2500:],'stderr_tail':p.stderr[-2500:],'timeout_sec':timeout}|marker(p.stdout)
 except subprocess.TimeoutExpired as e:
  so=(e.stdout or '') if isinstance(e.stdout,str) else (e.stdout or b'').decode(errors='replace')
  se=(e.stderr or '') if isinstance(e.stderr,str) else (e.stderr or b'').decode(errors='replace')
  return {'cmd':' '.join(cmd),'rc':124,'stdout_tail':so[-2500:],'stderr_tail':(se+f'\nTIMEOUT {timeout}s')[-2500:],'timeout_sec':timeout}

def email_campaign(case_id:str, root:Path)->dict[str,Any]:
 template='Subject: {{ intent }} proof update\n\nDRAFT ONLY — {{ rendered_brief }}\nCase={{ text }}'
 steps=['preserve evidence and ask for confirmation','follow up with timeline facts','close with next-action request']
 routes=[route_text(f'draft_only strategic email campaign {s} for {case_id} with evidence',channel='email',template=template,verbosity='brief') for s in steps]
 text='\n\n---\n\n'.join(r['rendered'] for r in routes); path=out_path(root,'strategic_email_campaign.md'); path.write_text(text,encoding='utf-8')
 return {'schema':'lucidota.usecase.email_campaign.v1','path':str(path),'route':routes[0]['membrane'],'draft_count':len(routes),'output_sha256':sha256_json({'text':text}),'model_calls_performed':False,'external_sends_performed':False}

def write_hypertimeline_fixture(root:Path)->list[str]:
 base=datetime(2026,5,26,tzinfo=timezone.utc); paths=[]
 for source,offset in [('codex',0),('claude',7)]:
  p=out_path(root,f'{source}_events.jsonl')
  with p.open('w',encoding='utf-8') as f:
   for i in range(32):
    ts=(base+timedelta(minutes=30*i+offset)).isoformat().replace('+00:00','Z')
    f.write(json.dumps({'source':source,'timestamp':ts,'sender':source,'action':'proof_event','content':f'{source} event {i}'},sort_keys=True)+'\n')
  paths.append(str(p))
 return paths

def proxy_mask_identity_proof(case_id:str, root:Path)->dict[str,Any]:
 bridge=build_percyphon_bridge(
  raw_command=f'route procedural mask scaffold for {case_id}',
  normalized_intent='procedural scaffold only; no truth promotion',
  authority_class='operator_authored_assertion',
  source='lucidota_usecase_proof',
  villagers=['operator','scribe','witness'],
  fluid_slots=5,
  queue_name='proof_proxy_mask',
  lane='identity_scaffold',
  ledger_path=root/'ledger.jsonl',
  event_log=root/'events.jsonl',
  receipt_dir=root/'proxy_mask'
 )
 routed=anthropic_proxy.anthropic_to_openai({
  'system':'local shim proof',
  'messages':[{'role':'user','content':[{'type':'text','text':f'{case_id} / procedural mask proof'}]}],
  'max_tokens':96,
  'temperature':0.1,
  'stream':False,
 })
 path=out_path(root,'proxy_mask_identity.json')
 report={'schema':'lucidota.usecase.proxy_mask_identity.v1','case_id':case_id,'path':str(path),'bridge':bridge,'anthropic_route':{'model':routed['model'],'message_count':len(routed['messages']),'stream':routed['stream']},'model_calls_performed':False,'external_sends_performed':False}
 path.write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
 return report

def learning_network_proof(case_id:str, timeout:int, root:Path)->dict[str,Any]:
 return run_cmd([sys.executable,'scripts/lucidota_bytewax_mini.py','--json','--limit','5'],timeout)

def prove(case_id:str, root:Path, timeout:int=120)->dict[str,Any]:
 root=Path(root); root.mkdir(parents=True,exist_ok=True); campaign=email_campaign(case_id,root/'campaign'); fixture=write_hypertimeline_fixture(root/'hypertimeline_fixture')
 proxy_mask=proxy_mask_identity_proof(case_id,root/'proxy_mask')
 learning=learning_network_proof(case_id,timeout,root/'learning_network')
 code=run_cmd([sys.executable,'scripts/lucidota_ouroboros_loop.py','--loops','1','--target','scripts/lucidota_usecase_proof.py','--receipt-root',str(root/'code_job'),'--timeout-sec',str(min(timeout,30)),'--json'],timeout)
 acc=run_cmd([sys.executable,'scripts/lucidota_acceptance.py','--self-fixture','--base-dir',str(root/'acceptance'),'--case-id',case_id,'--json'],timeout)
 hyp=[sys.executable,'scripts/hypertimeline_engine.py','--skip-gc','--case-key',case_id,'--case-title','LUCIDOTA usecase proof','--batch-size','64']
 for f in fixture: hyp += ['--input',f]
 hyper=run_cmd(hyp,timeout)
 checks={'campaign_draft':campaign['draft_count']>=3 and not campaign['model_calls_performed'],'proxy_mask':proxy_mask['bridge']['status']=='ROUTED' and proxy_mask['anthropic_route']['model']==anthropic_proxy.MODEL,'learning_network':learning['rc']==0,'code_job':code['rc']==0,'case_acceptance':acc['rc']==0,'hypertimeline':hyper['rc']==0}
 report={'schema':'lucidota.usecase_proof.v1','generated_at':now(),'case_id':case_id,'status':'PASSED' if all(checks.values()) else 'FAILED','checks':checks,'campaign':campaign,'proxy_mask':proxy_mask,'hypertimeline_fixture':fixture,'learning_network':learning,'commands':{'code_job':code,'case_acceptance':acc,'hypertimeline':hyper},'model_calls_performed':False,'external_sends_performed':False,'canonical_graph_writes_requested':False}
 path=root/f'usecase_proof_{stamp()}.json'; report['report_path']=rel(path); write_json(path,report); return report

def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('--case-id',default='CASE-20260526-PROOF'); ap.add_argument('--out-dir',default='05_OUTPUTS/usecase_proofs'); ap.add_argument('--timeout-sec',type=int,default=120); ap.add_argument('--json',action='store_true'); a=ap.parse_args()
 r=prove(a.case_id,ROOT/a.out_dir/a.case_id,a.timeout_sec); print('REPORT_PATH='+r['report_path']); print('LUCIDOTA_USECASE_PROOF='+r['status']); print(json.dumps(r,sort_keys=True) if a.json else ''); return 0 if r['status']=='PASSED' else 5
if __name__=='__main__': raise SystemExit(main())
