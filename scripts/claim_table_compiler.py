#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from spine_common import ROOT, rel, receipt, sha256_json
from source_quote_extractor import attach_quote_context
from claim_support_score import score_claims
from claim_clusterer import cluster_claims

def compile_claim_table(staging_rows: list[dict[str,Any]], *, output_path: str|Path|None=None, chunk_texts: dict[str,str]|None=None) -> dict[str,Any]:
    claims=[]
    for r in staging_rows:
        span=r.get('source_span') or {}; ref=r.get('source_ref') or {}
        if span.get('start') is None or span.get('end') is None or not ref.get('custody_id') or not ref.get('chunk_id'):
            raise ValueError('claim row missing source span/custody/chunk refs')
        row={'claim_id':'claim:'+sha256_json({'staging_id':r['staging_id'],'text':r['claim_text']})[:24], 'claim_text':r['claim_text'], 'modality':r['modality'], 'confidence_bps':r['confidence_bps'], 'source_ref':ref, 'source_span':span, 'staging_ref':r['staging_id'], 'status':'CANDIDATE'}
        if chunk_texts and ref.get('chunk_id') in chunk_texts:
            row=attach_quote_context(row, chunk_texts[ref['chunk_id']])
        claims.append(row)
    claims=cluster_claims(score_claims(claims))
    out=Path(output_path) if output_path else ROOT/'05_OUTPUTS/product_claims/claim_table.jsonl'; out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for c in claims: f.write(json.dumps(c,sort_keys=True)+'\n')
    rec={'status':'PASSED','claim_table_path':rel(out),'claim_count':len(claims),'claims':claims}
    rp=receipt('claim_table',rec,root='05_OUTPUTS/product_claims'); rec['receipt_path']=rel(rp); return rec

def claim_table_from_file(staging_path: str|Path, output_path: str|Path|None=None):
    rows=[json.loads(l) for l in Path(staging_path).read_text().splitlines() if l.strip()]
    return compile_claim_table(rows, output_path=output_path)
