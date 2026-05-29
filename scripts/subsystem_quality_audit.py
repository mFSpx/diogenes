#!/usr/bin/env python3
"""Legal-authority-system quality sweep: verdict every subsystem/script."""
from __future__ import annotations
import hashlib,json,re,shutil
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/subsystem_quality_audit'
TT=ROOT/'00_PROJECT_BRAIN/TICKLETRUNK.json'; REF=Path('/tmp/legal_authority_system_ref')
VERDICTS=('PROMOTE','REPAIR','QUARANTINE','MERGE','KRAMPUSCHEW')
OWN=[('absurd','ABSURD'),('chrono','Chrono'),('graph','Graph'),('krampus','KRAMPUSCHEWING'),('korpus','KORPUS'),('case','CaseOps'),('security','Security'),('secret','Security'),('indy','Indy_READs'),('model','ModelRuntime'),('surface','Surfaces'),('ahoy','AHOY_PAUSED'),('legacy','LegacyArchive'),('tickletrunk','TICKLETRUNK'),('status_ledger','Governance')]
def now(): return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def read(p,n=200000):
 try: return Path(p).read_text(errors='ignore')[:n]
 except Exception: return ''
def sha(p):
 h=hashlib.sha256()
 try:
  with Path(p).open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
  return h.hexdigest()
 except Exception: return ''
def jl(path,rows):
 path.write_text(''.join(json.dumps(r,sort_keys=True,default=str)+'\n' for r in rows),encoding='utf-8')
def owner(path):
 low=path.lower()
 for k,v in OWN:
  if k in low: return v
 return 'Toolbox'
def is_proof_hoard_backlog(row):
 rp=(row.get('path') or '').lower(); role=(row.get('proof_hoard_role') or '').lower()
 if rp in {'scripts','01_repos','algos'} or rp.endswith(('.jsonl','.pyc')): return True
 if rp.startswith('01_repos/') or rp.startswith('scripts/legacy/') or '/legacy/' in rp: return True
 if role in {'external_repo','reference_material','sandbox_experiment','sovereign_original','corpse_preserved_for_hashes','paused_lab'}: return True
 return False
def production_blocking(row):
 return row.get('verdict')!='PROMOTE' and not is_proof_hoard_backlog(row)
def tests_for(names):
 tests={}
 for p in (ROOT/'tests').rglob('test_*.py'):
  txt=read(p)
  for n in names:
   if n in txt: tests.setdefault(n,[]).append(rel(p))
 return tests
def facts(row,tests,hash_dupes,name_dupes):
 p=Path(row.get('path') or ROOT/row['relative_path']); rp=rel(p); txt=read(p); low=txt.lower()
 loc=sum(1 for l in txt.splitlines() if l.strip()); h=sha(p)
 purpose=(row.get('what_it_does') or row.get('one_line_description') or '').strip()
 f={'path':rp,'name':row.get('name') or p.name,'owner_subsystem':owner(rp),'proof_hoard_role':row.get('proof_hoard_role',''),'sha256':h,'loc':loc,'size_bytes':p.stat().st_size if p.exists() else 0,
 'purpose_ok':bool(purpose and 'UNKNOWN' not in purpose and 'filename suggests' not in purpose),'purpose':purpose[:240],
 'io_contract':bool(row.get('inputs') or row.get('outputs') or 'argparse' in low or 'def main' in low),
 'bounded':any(x in low for x in ('max_','limit','timeout','batch','chunk','stream','reserve','size_bytes','read(1024','iter(')),
 'receipts':bool('report_path=' in low or 'receipt' in low or '05_outputs' in low or 'write_report' in low),
 'fails_closed':bool('raise systemexit' in low or 'return 1' in low or 'return 2' in low or 'blocker' in low or 'check=false' in low),
 'idempotent':bool('idempot' in low or 'on conflict' in low or 'duplicate' in low or 'sha256' in low),
 'graph_safe':not bool(re.search(r'(insert|update|delete).*lucidota_go\\.(graph_item|graph_edge)',low,re.S)) or 'graph_promotion' in low or 'materialization_helper' in low,
 'tests':tests.get(p.name,[]),'duplicate_names':name_dupes.get(p.name,[]),'duplicate_hashes':hash_dupes.get(h,[]) if h else []}
 f['score']=sum(bool(f[k]) for k in ['purpose_ok','io_contract','bounded','receipts','fails_closed','idempotent','graph_safe'])+(2 if f['tests'] else 0)
 return f
def verdict(f):
 rp=f['path'].lower(); dangerous=not f['graph_safe'] or 'secret' in rp and not f['tests']
 if 'legacy/active_root_pruned' in rp or 'legacy/dbos' in rp or 'lucidota_dbos' in rp: return 'KRAMPUSCHEW','retired/legacy executable below launch bar'
 if dangerous: return 'QUARANTINE','unsafe boundary or secret/graph risk'
 if f['duplicate_hashes'] or ('legacy/' in rp and f['owner_subsystem']=='LegacyArchive'): return 'MERGE','duplicate/legacy copy; collapse into stronger implementation'
 if f['score']>=8 and f['tests'] and f['receipts'] and f['bounded'] and f['graph_safe']: return 'PROMOTE','meets reference floor for current role'
 if f['score']>=4: return 'REPAIR','useful but below legal_authority_system floor'
 return 'KRAMPUSCHEW','unbounded/untested/unclear executable'
def main():
 OUT.mkdir(parents=True,exist_ok=True); run=stamp(); data=json.loads(TT.read_text()); scripts=[r for r in data.get('toolboxes',{}).get('SCRIPTS',[]) if r.get('relative_path')]
 names=[r.get('name') or Path(r['relative_path']).name for r in scripts]; t=tests_for(names)
 byh=defaultdict(list); byn=defaultdict(list)
 for r in scripts:
  p=Path(r.get('path') or ROOT/r['relative_path']); h=sha(p); n=r.get('name') or p.name
  if h: byh[h].append(rel(p)); byn[n].append(rel(p))
 hash_dupes={h:v for h,v in byh.items() if len(v)>1}; name_dupes={n:v for n,v in byn.items() if len(v)>1}
 verdicts=[]; corpses=[]; repair=[]; promo=[]; subs=defaultdict(Counter); kr=ROOT/'KRAMPUSCHEWING/Quality_Audit'/run; kr.mkdir(parents=True,exist_ok=True)
 for r in scripts:
  f=facts(r,t,hash_dupes,name_dupes); v,reason=verdict(f); f.update({'verdict':v,'reason':reason,'quality_bar':'legal_authority_system@'+(shutil.which('git') and __import__('subprocess').check_output(['git','-C',str(REF),'rev-parse','--short','HEAD'],text=True).strip() if (REF/'.git').exists() else 'README_ONLY')})
  p=ROOT/f['path']; route=''
  if v!='PROMOTE' and p.exists() and p.is_file():
   dest=kr/(f['path'].replace('/','__')+'.'+f['sha256']+'.artifact'); shutil.copy2(p,dest); route=rel(dest)
  f['krampuschewing_ingest_route']=route; verdicts.append(f); subs[f['owner_subsystem']][v]+=1
  q={'path':f['path'],'hash':f['sha256'],'reason':reason,'replacement_target':'stronger subsystem implementation or Rust port','dependencies':r.get('depends_on',[]),'last_known_purpose':f['purpose'],'krampuschewing_ingest_route':route}
  f['proof_hoard_backlog']=is_proof_hoard_backlog(f); f['production_blocking']=production_blocking(f)
  if v=='KRAMPUSCHEW': corpses.append(q)
  if v in {'REPAIR','QUARANTINE','MERGE'}: repair.append(q|{'verdict':v})
  if v=='PROMOTE': promo.append(f)
 inv=[{'subsystem':k,'counts':dict(c),'total':sum(c.values())} for k,c in sorted(subs.items())]
 active_blockers=[x for x in verdicts if x.get('production_blocking')]
 backlog=[x for x in verdicts if x.get('proof_hoard_backlog') and x.get('verdict')!='PROMOTE']
 for name,rows in [('subsystem_inventory_latest.jsonl',inv),('script_verdicts_latest.jsonl',verdicts),('corpse_manifest_latest.jsonl',corpses),('repair_queue_latest.jsonl',repair),('promotion_manifest_latest.jsonl',promo),('active_runtime_blockers_latest.jsonl',active_blockers),('proof_hoard_backlog_latest.jsonl',backlog)]: jl(OUT/name,rows); jl(OUT/f'{run}_{name}',rows)
 vc=Counter(x['verdict'] for x in verdicts); sec=sorted((ROOT/'05_OUTPUTS/security').glob('security_scan_*.json')); sc=0
 if sec:
  try: sc=json.loads(sec[-1].read_text()).get('count',0)
  except Exception: sc=0
 md=['# Subsystem Quality Gate Report',f'Generated: `{now()}`','',f'Reference: `legal_authority_system` commit `{verdicts[0]["quality_bar"].split("@")[-1] if verdicts else "unknown"}`; observed repo is README-only, so the encoded bar is narrow scope + explicit authority + provenance + no theater.','',f'Verdicts: `{dict(vc)}`','',f'What survives: {vc["PROMOTE"]} promoted scripts/components.','',f'What dies as executable code: {vc["KRAMPUSCHEW"]} KRAMPUSCHEW candidates, copied under `{rel(kr)}`.','',f'What merges / duplicates: {vc["MERGE"]}.','',f'What is dangerous: {vc["QUARANTINE"]} quarantined scripts plus {sc} latest masked security-scan findings.','',f'High-value unfinished: {vc["REPAIR"]}.','',f'Active runtime blockers: {len(active_blockers)} non-promoted active/supporting scripts; see `active_runtime_blockers_latest.jsonl`.','',f'Proof-hoard backlog, non-blocking until promoted: {len(backlog)} external/legacy/reference artifacts; see `proof_hoard_backlog_latest.jsonl`.','',f'Production blockers: active runtime blockers, secret quarantine findings, remaining direct/unsafe graph boundaries, and active files over 100 LOC until justified or pruned.','',f'What should become canonical: see `promotion_manifest_latest.jsonl`.']
 (OUT/'quality_gate_report_latest.md').write_text('\n'.join(md)+'\n',encoding='utf-8'); (OUT/f'{run}_quality_gate_report.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
 print('REPORT_PATH='+rel(OUT/'quality_gate_report_latest.md')); print('SCRIPT_VERDICTS='+str(len(verdicts))); print('VERDICT_COUNTS='+json.dumps(dict(vc),sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
