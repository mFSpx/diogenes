#!/usr/bin/env python3
"""Deterministic language subsystem: messy text -> GO/JSON lanes/work orders."""
from __future__ import annotations
import argparse,json,re,sys
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/language_router'
for p in (ROOT,ROOT/'scripts'):
 if str(p) not in sys.path: sys.path.insert(0,str(p))
from ALGOS.decision_hygiene import counts, score_features  # noqa:E402
from ALGOS.percyphon import procedural_entity_generator  # noqa:E402
from ALGOS.rete_bandit_gate import GO25  # noqa:E402
from core.language_membrane import route_inbound_text, weave_output  # noqa:E402
from scripts.cep_builder import build_cep  # noqa:E402
from scripts.fast_slow_lane_gate import route_packet  # noqa:E402
from scripts.kernel_control_packet import absurd_enqueue_packet, verify_control_packet  # noqa:E402
from scripts.spine_common import sha256_json  # noqa:E402
from scripts.template_contract import render_template  # noqa:E402
INTENTS={'coding':r'\b(code|bug|test|pytest|build|refactor|script|loc|cli|agent)\b','investigation':r'\b(osint|investigat|research|evidence|claim|source|news|journal|case|krampus|korpus|hypertimeline)\b','correspondence':r'\b(email|letter|reply|correspond|send|draft)\b','ops':r'\b(status|telemetry|cpu|ram|vram|db|service|recover|queue|absurd)\b','ontology':r'\b(ontology|term|json|schema|hypothesis|entity)\b'}
TERM_HINTS={'coding':['TOOL','ALGORITHM','ACTION','MODE'],'investigation':['EVIDENCE','CLAIM','HYPOTHESIS','SIGNAL','PATTERN'],'correspondence':['COMMENT','OPERATOR','EVIDENCE','CLAIM'],'ops':['TIME','EVENT','TOOL','MODE'],'ontology':['ENTITY','RELATIONSHIP','TERM','CLAIM']}
TEMPLATES={'operator':'INTENT={{ intent }} TERMS={{ ontology_terms }} LANE={{ lane.lane }} TASK={{ work_order.task }}','email':'Subject: {{ intent }} update\n\n{{ rendered_brief }}','coding':'{{ intent }} work-order {{ work_order.work_order_id }}: {{ work_order.task }}','research':'{{ intent }} / {{ ontology_terms }} / evidence-first draft'}
VERBOSITY={'terse':35,'brief':90,'normal':180,'full':420}
GO25_SET=set(GO25)
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def file_terms()->list[str]:
 out=[]
 for p in (ROOT/'OFFICIAL_ONTOLOGY.json',ROOT/'BOOKS/GO_ACTIVE_TERMS.json'):
  try:
   data=json.loads(p.read_text())
   vals=data.get('active_terms',[])+data.get('extension_terms',[])+[x.get('term') for x in data.get('terms',[]) if isinstance(x,dict)]
   out += [str(v) for v in vals if v]
  except Exception:
   pass
 return list(dict.fromkeys(out))
def go25_terms(terms:list[str])->list[str]:
 out=[]; seen=set()
 for t in terms:
  if t in GO25_SET and t not in seen: out.append(t); seen.add(t)
 return out
def db_terms(database_url:str|None=None)->tuple[list[str],dict[str,Any]]:
 try:
  import psycopg  # type: ignore
  url=database_url or 'postgresql:///lucidota_storage'
  with psycopg.connect(url, connect_timeout=2) as conn, conn.cursor() as cur:
   cur.execute("SELECT term FROM lucidota_go.term_registry WHERE status='active' ORDER BY term_number NULLS LAST, term")
   return [r[0] for r in cur.fetchall()], {'source':'db','database_url':'redacted'}
 except Exception as e: return [], {'source':'db_unavailable','error':type(e).__name__}
def ontology_terms(refresh:bool=False,database_url:str|None=None)->tuple[list[str],dict[str,Any]]:
 if refresh:
  terms,meta=db_terms(database_url)
  if terms: return go25_terms(terms),meta
 terms=go25_terms(file_terms()); return terms, {'source':'json','term_count':len(terms)}
def pick_intent(text:str, channel:str='operator')->str:
 low=(text+' '+channel).lower()
 if channel in {'email','letter'}: return 'correspondence'
 scores={k:len(re.findall(rx,low,re.I)) for k,rx in INTENTS.items()}
 return max(scores,key=scores.get) if max(scores.values() or [0]) else 'ops'
def pick_terms(text:str,intent:str,terms:list[str])->list[str]:
 hits=[t for t in terms if re.search(rf'\b{re.escape(t)}\b',text,re.I)]; use=[]
 for t in TERM_HINTS.get(intent,[]):
  if t in terms and (intent!='investigation' or t in hits or t in {'EVIDENCE','CLAIM'}): use.append(t)
 for t in hits:
  if t not in use: use.append(t)
 return use[:12]
def workflow_release(text:str, channel:str)->bool:
 low=(text+' '+channel).lower()
 return bool(re.search(r'\b(?:end of workflow|workflow end|final(?: print)?|handoff|done|complete|release|print at end)\b', low))
def model_hook(text:str,intent:str,lane:dict)->dict[str,Any]:
 low=text.lower(); blocked=bool(re.search(r'\b(no|without)\s+(llm|model)|unless necessary',low)); needs=not blocked and (lane.get('lane')=='SLOWLANE') and bool(re.search(r'\b(deep|synth|draft|summarize|ambiguous|model|llm)\b',low))
 tier='none' if not needs else ('cheap_local_or_mini' if intent in {'coding','ops'} else 'standard_review')
 return {'policy':'deterministic_first','needed':needs,'selected_tier':tier,'vram_budget_mb':0 if not needs else 1650,'model_calls_performed':False}
def route_text(text:str, channel:str='operator', template:str|None=None, *, verbosity:str='normal', refresh_ontology:bool=False, database_url:str|None=None, source_surfaces:list[str]|None=None)->dict:
 terms,refresh=ontology_terms(refresh_ontology,database_url); intent=pick_intent(text,channel); c=counts(text); hygiene,label=score_features(c); use=pick_terms(text,intent,terms)
 release=workflow_release(text,channel)
 lane=route_packet({'text':text,'metadata':{'proof_required':'prove' in text.lower() or 'evidence' in text.lower(),'deep_analysis':intent in {'coding','investigation'}}},slow_char_threshold=180)
 cep=build_cep(raw_command=text, normalized_intent=intent, source=f'{channel}_language_subsystem', target_refs=['language_membrane','absurd_queue','hypertimeline'], evidence_refs=['scripts/language_router.py'], allowed_effect='draft_work_order_only')
 wid='wo:'+sha256_json({'cep':cep['command_id'],'text':text})[:24]; work={'work_order_id':wid,'cep':cep,'task':f'{intent}: '+re.sub(r'\s+',' ',text).strip()[:220],'allowed_files':['scripts/language_router.py','core/language_membrane.py','scripts/template_contract.py','scripts/fast_slow_lane_gate.py'],'status':'CREATED'}
 auth=absurd_enqueue_packet(queue_name='language_subsystem', lane=lane['lane'].lower(), source_path='scripts/language_router.py', idempotency_key=wid, authorized_by='language_router'); ok,err=verify_control_packet(auth)
 model=model_hook(text,intent,lane); membrane=route_inbound_text(text,k=3); brief=f"{intent} via {lane['lane']} using {','.join(use) or 'SIGNAL'}"; ctx={'text':text,'channel':channel,'intent':intent,'ontology_terms':use,'lane':lane,'hygiene':{'score':hygiene,'label':label},'work_order':work,'rendered_brief':brief}
 rendered=render_template(template or TEMPLATES.get(channel) or TEMPLATES.get(intent) or TEMPLATES['operator'],ctx)
 woven=weave_output(deterministic_template=rendered,rag_quotes=[],deepseek_synthesis='',fairyfuse_context={'intent':intent,'terms':use,'model':model})
 return {'schema':'lucidota.language_subsystem.route.v2','generated_at':now(),'source_surfaces':source_surfaces or ['cli','operator'],'text_chars':len(text),'intent':intent,'ontology_mode':'GO25_STRICT','ontology_terms':use,'ontology_release':release,'ontology_refresh':refresh,'lane':lane,'membrane':membrane,'outbound':{'audience':channel,'state':'final_print' if release else 'draft_only','max_words':VERBOSITY.get(verbosity,180)},'output_contract':{'verbosity':verbosity,'template':template or 'default'},'rendered':rendered,'woven':woven,'decision_hygiene':ctx['hygiene'],'work_order':work,'absurd':{'required_for_workflow':True,'enqueue_packet_authorized':ok,'authorization_error':err,'execute_performed':False},'hypertimeline_hook':{'script':'scripts/hypertimeline_engine.py','mode':'timeline_receipt_after_queue_event'},'model_route':model,'percyphon':procedural_entity_generator(use or [intent],fluid_slots=4),'model_calls_performed':False,'canonical_graph_writes_performed':False}
def write(r:dict)->dict:
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'language_router_{stamp()}.json'; r['report_path']=str(p.relative_to(ROOT)); p.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n'); return r
def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('--text',required=True); ap.add_argument('--channel',default='operator'); ap.add_argument('--template'); ap.add_argument('--verbosity',default='normal',choices=sorted(VERBOSITY)); ap.add_argument('--refresh-ontology',action='store_true'); ap.add_argument('--database-url'); ap.add_argument('--json',action='store_true'); a=ap.parse_args()
 r=write(route_text(a.text,a.channel,a.template,verbosity=a.verbosity,refresh_ontology=a.refresh_ontology,database_url=a.database_url)); print('REPORT_PATH='+r['report_path']); print('LANGUAGE_ROUTER=PASS'); print(json.dumps(r,sort_keys=True) if a.json else 'INTENT='+r['intent']); return 0
if __name__=='__main__': raise SystemExit(main())
