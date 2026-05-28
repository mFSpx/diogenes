#!/usr/bin/env python3
from __future__ import annotations
from collections import defaultdict
from typing import Any
from spine_common import sha256_json

def build_event_registry(*, chrono_claims: list[dict[str,Any]], staged_events: list[dict[str,Any]]|None=None) -> list[dict[str,Any]]:
    groups=defaultdict(lambda:{'source_refs':[],'conflict':False,'methods':set()})
    for c in chrono_claims:
        key=c['candidate_timestamp']
        groups[key]['source_refs'].append(c.get('source_ref'))
        groups[key]['conflict']=groups[key]['conflict'] or bool(c.get('conflict'))
        groups[key]['methods'].add(c.get('method','unknown'))
    for e in staged_events or []:
        key=e.get('event_time') or 'UNKNOWN'
        groups[key]['source_refs'].append(e.get('source_ref'))
        groups[key]['methods'].add(e.get('extraction_method','staged_event'))
    out=[]
    for key,g in sorted(groups.items()):
        out.append({'event_id':'eventcand:'+sha256_json({'time':key,'refs':g['source_refs']})[:24], 'event_time':key, 'source_refs':g['source_refs'], 'uncertainty':'CONFLICT' if g['conflict'] else ('UNKNOWN_TIME' if key=='UNKNOWN' else 'OBSERVED_CLAIM'), 'conflict':g['conflict'], 'methods':sorted(g['methods']), 'status':'CANDIDATE'})
    return out
