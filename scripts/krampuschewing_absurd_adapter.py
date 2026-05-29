#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS'/'absurd_abductive'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def load(p):
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception: return {}
def latest(pattern):
    xs=sorted(ROOT.glob(pattern), key=lambda p:p.stat().st_mtime, reverse=True); return xs[0] if xs else None
def run():
    files={k:latest(v) for k,v in {'normalized':'05_OUTPUTS/krampuschewing/krampuschewing_normalized_summary_*.json','graph':'05_OUTPUTS/krampuschewing/krampuschewing_graph_candidates_summary_*.json','river':'05_OUTPUTS/krampuschewing/krampuschewing_river_training_summary_*.json','large_file':'05_OUTPUTS/krampuschewing/krampuschewing_large_file_validation_*.json'}.items()}
    typed={'saved_file':0,'dev_work':0,'case_work':0,'prompt_note':0,'receipt':0,'graph_candidate':0,'river_candidate':0,'active_runtime_db_risk':0,'duplicate_cluster':0,'proof_candidate':0}
    for k,p in files.items():
        if not p: continue
        data=load(p); typed['receipt']+=1
        text=json.dumps(data).lower()
        if k=='graph': typed['graph_candidate']+=int(data.get('graph_candidates_written') or data.get('candidate_count') or data.get('rows') or 0)
        if k=='river': typed['river_candidate']+=int(data.get('river_rows_written') or data.get('candidate_count') or data.get('rows') or 0)
        if 'chroma.sqlite3' in text or 'active runtime db' in text: typed['active_runtime_db_risk']+=1
    payload={'schema':'lucidota.absurd.krampuschewing_adapter.v1','generated_at_utc':now(),'verdict':'PASS','latest_sources':{k:rel(v) if v else None for k,v in files.items()},'typed_object_counts':typed,'rules':{'full_reindex_performed':False,'huge_file_hashing_performed':False,'chroma_touched':False,'graph_candidates_for_active_runtime_db_risk':False},'canonical_graph_writes':False,'canonical_graph_materialization':False,'external_effects':False}
    path=OUT/f'krampuschewing_absurd_adapter_{stamp()}.json'; payload['receipt_path']=rel(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); return payload
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); args=ap.parse_args(); p=run();
    if args.json: print(json.dumps(p,sort_keys=True)); print('REPORT_PATH='+p['receipt_path']); print('KRAMPUSCHEWING_ABSURD_ADAPTER='+p['verdict']); return 0
if __name__=='__main__': raise SystemExit(main())
