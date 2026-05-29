#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any
from spine_common import ROOT, rel, receipt
from ontology_staging import stage_claim_from_sentence
SENTENCE_RE=re.compile(r'[^.!?]+[.!?]?')

def chunks_to_staging(chunks: list[dict[str,Any]]) -> list[dict[str,Any]]:
    rows=[]
    for chunk in chunks:
        for m in SENTENCE_RE.finditer(chunk.get('text','')):
            text=m.group(0).strip()
            if not text: continue
            rows.append(stage_claim_from_sentence(text, chunk=chunk, start=chunk.get('start',0)+m.start(), end=chunk.get('start',0)+m.end()))
    return rows

def stage_chunks_file(chunks_path: str|Path, output_path: str|Path|None=None) -> dict[str,Any]:
    chunks=[json.loads(l) for l in Path(chunks_path).read_text().splitlines() if l.strip()]
    rows=chunks_to_staging(chunks)
    out=Path(output_path) if output_path else ROOT/'05_OUTPUTS/product_claims/staging.jsonl'; out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,sort_keys=True)+'\n')
    rec={'status':'PASSED','staging_path':rel(out),'claim_count':len(rows),'rows':rows}
    rp=receipt('chunk_to_staging', rec, root='05_OUTPUTS/product_claims'); rec['receipt_path']=rel(rp); return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('chunks_path'); ap.add_argument('--output-path'); a=ap.parse_args(); stage_chunks_file(a.chunks_path,a.output_path); return 0
if __name__=='__main__': raise SystemExit(main())
