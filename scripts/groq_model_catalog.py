#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,urllib.error,urllib.request
from datetime import datetime,timezone
from pathlib import Path
from groq_env import load_groq_env
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/model_invocations'; API='https://api.groq.com/openai/v1/models'; DOCS='https://console.groq.com/docs/models'
PRICE={
 'llama-3.1-8b-instant':(.05,.08,560,'cheap_default'),'openai/gpt-oss-20b':(.075,.30,1000,'cheap_reasoning'),'meta-llama/llama-4-scout-17b-16e-instruct':(.11,.34,750,'vision_or_larger_fast'),'openai/gpt-oss-120b':(.15,.60,500,'strong_reasoning'),'qwen/qwen3-32b':(.29,.59,400,'code_math_preview'),'llama-3.3-70b-versatile':(.59,.79,280,'large_quality'),'meta-llama/llama-prompt-guard-2-22m':(.03,.03,0,'guard_only'),'meta-llama/llama-prompt-guard-2-86m':(.04,.04,0,'guard_only'),'openai/gpt-oss-safeguard-20b':(.075,.30,1000,'safety_only')}

def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try:return str(Path(p).resolve().relative_to(ROOT))
 except Exception:return str(p)
def fetch(key:str)->list[dict]:
 req=urllib.request.Request(API,headers={'Authorization':'Bearer '+key,'Accept':'application/json','User-Agent':'lucidota-groq-model-catalog/1.0'})
 with urllib.request.urlopen(req,timeout=20) as r: data=json.loads(r.read().decode())
 return sorted([{k:m.get(k) for k in ('id','object','created','owned_by')} for m in data.get('data',[])],key=lambda x:x.get('id') or '')
def rank(models:list[dict])->dict:
 ids={m.get('id') for m in models}; ranked=[]
 for mid,(i,o,s,role) in PRICE.items():
  if mid in ids: ranked.append({'id':mid,'role':role,'input_per_1m_usd':i,'output_per_1m_usd':o,'speed_tps':s,'efficiency_score':round((s or 1)/max((i+o)/2,0.001),2)})
 return {'default_model':'llama-3.1-8b-instant' if 'llama-3.1-8b-instant' in ids else (ranked[0]['id'] if ranked else None),'ranked_known':sorted(ranked,key=lambda x:(x['role']=='guard_only',-x['efficiency_score'])),'unknown_pricing_models':sorted(ids-set(PRICE))}
def write(r:dict)->Path:
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'groq_model_catalog_{stamp()}.json'; r['report_path']=rel(p); p.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n',encoding='utf-8'); return p

def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('--execute',action='store_true'); ap.add_argument('--json',action='store_true'); a=ap.parse_args()
 key=os.environ.get('GROQ_API_KEY'); blockers=[]; models=[]
 if a.execute: load_groq_env(); key=os.environ.get('GROQ_API_KEY')
 if a.execute and not key: blockers.append('missing_api_key_env:GROQ_API_KEY')
 if a.execute and key and not blockers:
  try: models=fetch(key)
  except urllib.error.HTTPError as e: blockers.append(f'groq_http_error:{e.code}')
  except Exception as e: blockers.append(f'groq_models_failed:{type(e).__name__}:{e}')
 r={'schema':'lucidota.groq.model_catalog.v1','generated_at':now(),'provider':'groq','endpoint':API,'docs_source':DOCS,'api_key_env_used':'GROQ_API_KEY' if key else None,'api_key_redacted':bool(key),'execute_performed':bool(a.execute and key and not blockers),'model_count':len(models),'models':models,'recommendation':rank(models),'policy':'Default GOALS cloud worker: llama-3.1-8b-instant; escalate only for bounded reasoned review. Core LUCIDOTA never requires cloud.','blockers':blockers}
 p=write(r); print('REPORT_PATH='+rel(p)); print('GROQ_MODEL_CATALOG='+('PASS' if not blockers else 'BLOCKED'))
 if a.json: print(json.dumps(r,sort_keys=True));
 return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
