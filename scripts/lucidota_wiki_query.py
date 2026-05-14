#!/usr/bin/env python3
"""File-backed VIBESCONTROL / Indy_Reads wiki query v0."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WIKI_DIRS=[ROOT/'00_PROJECT_BRAIN', ROOT/'02_RECORDS_OFFICE']

def sections():
    for d in WIKI_DIRS:
        for p in sorted(d.glob('*.md')):
            text=p.read_text(encoding='utf-8', errors='ignore')
            current='preamble'; buf=[]
            for line in text.splitlines():
                if line.startswith('#'):
                    if buf:
                        yield {'path':str(p.relative_to(ROOT)), 'heading':current, 'text':'\n'.join(buf).strip()}
                    current=line.lstrip('#').strip(); buf=[]
                else:
                    buf.append(line)
            if buf:
                yield {'path':str(p.relative_to(ROOT)), 'heading':current, 'text':'\n'.join(buf).strip()}

def score(sec, terms):
    hay=(sec['path']+' '+sec['heading']+' '+sec['text']).lower()
    return sum(hay.count(t) for t in terms)

def main():
    ap=argparse.ArgumentParser(prog='lucidota-wiki-query')
    ap.add_argument('query')
    ap.add_argument('--limit', type=int, default=8)
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()
    terms=[t.lower() for t in re.findall(r'[A-Za-z0-9_/-]+', args.query) if len(t)>1]
    hits=[]
    for sec in sections():
        s=score(sec, terms)
        if s:
            snippet=re.sub(r'\s+', ' ', sec['text'])[:360]
            hits.append({'score':s,'path':sec['path'],'heading':sec['heading'],'snippet':snippet})
    hits=sorted(hits, key=lambda h:(-h['score'],h['path'],h['heading']))[:args.limit]
    report={'ok':True,'query':args.query,'hits':hits,'searched_dirs':[str(d.relative_to(ROOT)) for d in WIKI_DIRS]}
    print(json.dumps(report, sort_keys=True) if args.json else '\n'.join(f"{h['score']} {h['path']} :: {h['heading']} :: {h['snippet']}" for h in hits))
    return 0
if __name__=='__main__': raise SystemExit(main())
