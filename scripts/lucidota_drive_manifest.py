#!/usr/bin/env python3
"""Local Drive/import manifest from existing records only; no Drive API access."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RECORDS=[ROOT/'00_PROJECT_BRAIN', ROOT/'02_RECORDS_OFFICE']
PATTERNS=['drive','pypeline','math','secret','scraper','template','persona','indy_reads']

def scan():
    hits=[]
    for d in RECORDS:
        for p in d.rglob('*.md'):
            txt=p.read_text(errors='ignore')
            low=txt.lower()
            score=sum(low.count(x) for x in PATTERNS)
            if score:
                heads=re.findall(r'^#+\s+(.+)$', txt, re.M)[:5]
                hits.append({'path':str(p.relative_to(ROOT)),'score':score,'headings':heads})
    return sorted(hits,key=lambda x:(-x['score'],x['path']))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); ap.add_argument('--limit',type=int,default=30); args=ap.parse_args()
    hits=scan()[:args.limit]
    report={'ok':True,'mode':'local_records_only_no_drive_api','targets':hits,'count':len(hits)}
    print(json.dumps(report,sort_keys=True) if args.json else '\n'.join(f"{h['score']:3} {h['path']}" for h in hits))
    return 0
if __name__=='__main__': raise SystemExit(main())
