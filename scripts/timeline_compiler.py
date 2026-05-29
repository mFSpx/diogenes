#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
from spine_common import ROOT, rel, receipt, write_json
from chrono_from_inventory import claims_from_inventory_rows
from event_candidate_registry import build_event_registry

def _load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []

def compile_timeline_from_custody(package_path: str|Path, *, source_root: str|Path|None=None, output_dir: str|Path|None=None) -> dict[str,Any]:
    pkg=json.loads(Path(package_path).read_text())
    rows=_load_jsonl(ROOT/pkg['normal_manifest_path'])+_load_jsonl(ROOT/pkg['quarantine_manifest_path'])
    claims=claims_from_inventory_rows(rows, source_root=source_root)
    by_time={}
    for c in claims: by_time.setdefault(c['source_ref']['custody_id'], set()).add(c['candidate_timestamp'])
    disputed={k:v for k,v in by_time.items() if len(v)>1}
    for c in claims: c['conflict']=c['source_ref']['custody_id'] in disputed
    timeline=sorted(claims, key=lambda c:(c['candidate_timestamp'], c['claim_id']))
    out=Path(output_dir) if output_dir else ROOT/'05_OUTPUTS/product_timeline'; out.mkdir(parents=True, exist_ok=True)
    tl=out/'timeline.json'; cl=out/'chrono_claims.jsonl'
    write_json(tl, {'timeline':timeline,'claim_count':len(claims),'conflict_count':len(disputed)})
    with cl.open('w',encoding='utf-8') as f:
        for c in claims: f.write(json.dumps(c,sort_keys=True)+'\n')
    events=build_event_registry(chrono_claims=claims)
    ev=out/'event_candidates.json'; write_json(ev, {'events':events,'event_count':len(events)})
    rec={'status':'PASSED','timeline_path':rel(tl),'chrono_claims_path':rel(cl),'event_candidates_path':rel(ev),'claim_count':len(claims),'event_count':len(events),'conflict_count':len(disputed),'timeline':timeline,'events':events}
    rp=receipt('product_timeline',rec,root='05_OUTPUTS/product_timeline'); rec['receipt_path']=rel(rp); return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('package_path'); ap.add_argument('--source-root'); ap.add_argument('--output-dir'); a=ap.parse_args(); compile_timeline_from_custody(a.package_path, source_root=a.source_root, output_dir=a.output_dir); return 0
if __name__=='__main__': raise SystemExit(main())
