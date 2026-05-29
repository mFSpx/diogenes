#!/usr/bin/env python3
from __future__ import annotations
from collections import defaultdict
from typing import Any
from spine_common import sha256_json

def build_entity_registry(staging_rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    groups=defaultdict(lambda:{'aliases':set(),'source_refs':[],'confidence':[]})
    for r in staging_rows:
        for name in r.get('entity_candidates',[]):
            key=name.strip().lower()
            if not key: continue
            groups[key]['aliases'].add(name)
            groups[key]['source_refs'].append({'staging_id':r.get('staging_id'),'source_ref':r.get('source_ref'),'source_span':r.get('source_span')})
            groups[key]['confidence'].append(int(r.get('confidence_bps',0)))
    out=[]
    for key,g in sorted(groups.items()):
        canonical=sorted(g['aliases'], key=lambda x:(len(x),x))[0]
        out.append({'entity_id':'entitycand:'+sha256_json({'name':key})[:24], 'canonical_name':canonical, 'aliases':sorted(g['aliases']), 'source_refs':g['source_refs'], 'confidence_bps':max(g['confidence']) if g['confidence'] else 0, 'status':'CANDIDATE'})
    return out
