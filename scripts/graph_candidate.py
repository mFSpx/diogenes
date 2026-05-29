#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
from spine_common import ROOT, rel, receipt, sha256_json

def candidates_from_staging(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    out=[]
    for r in rows:
        cand={'candidate_id':'graphcand:'+sha256_json({'staging_id':r['staging_id'],'claim_text':r['claim_text']})[:24], 'candidate_type':'claim_node', 'staging_ref':r['staging_id'], 'label':r['claim_text'][:120], 'entity_candidates':r.get('entity_candidates',[]), 'evidence_refs':[r['source_ref']], 'decision':'PENDING', 'canonical_mutation_allowed':False, 'promotion_blocker':'operator_decision_required'}
        out.append(cand)
    return out

def candidates_from_file(staging_path: str|Path, output_path: str|Path|None=None) -> dict[str,Any]:
    rows=[json.loads(l) for l in Path(staging_path).read_text().splitlines() if l.strip()]
    cands=candidates_from_staging(rows)
    out=Path(output_path) if output_path else ROOT/'05_OUTPUTS/product_graph/graph_candidates.jsonl'; out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for c in cands: f.write(json.dumps(c,sort_keys=True)+'\n')
    rec={'status':'PASSED','graph_candidates_path':rel(out),'candidate_count':len(cands),'candidates':cands}
    rp=receipt('graph_candidate', rec, root='05_OUTPUTS/product_graph'); rec['receipt_path']=rel(rp); return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('staging_path'); ap.add_argument('--output-path'); a=ap.parse_args(); candidates_from_file(a.staging_path,a.output_path); return 0
if __name__=='__main__': raise SystemExit(main())
