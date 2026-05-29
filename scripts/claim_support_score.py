#!/usr/bin/env python3
from __future__ import annotations
from collections import defaultdict
from typing import Any

def score_claims(claims: list[dict[str,Any]]) -> list[dict[str,Any]]:
    by_text=defaultdict(list)
    for c in claims: by_text[c['claim_text'].strip().lower()].append(c)
    scored=[]
    for c in claims:
        support_count=len({x.get('source_ref',{}).get('custody_id') for x in by_text[c['claim_text'].strip().lower()]})
        quote_bonus=1500 if c.get('quote_text') else 0
        confidence=int(c.get('confidence_bps',0))
        contradiction_penalty=2500 if c.get('contradiction_status')=='OPEN' else 0
        method_bonus=500 if c.get('staging_ref') else 0
        score=max(0,min(10000,confidence + quote_bonus + method_bonus + min(2000,support_count*500) - contradiction_penalty))
        row=dict(c); row['support_score']=score; row['support_reason']={'supporting_source_count':support_count,'quote_bonus':quote_bonus,'confidence_bps':confidence,'contradiction_penalty':contradiction_penalty,'method_bonus':method_bonus}
        scored.append(row)
    return scored
