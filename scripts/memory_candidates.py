#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
from spine_common import sha256_json
from operator_decisions import decision_allows

def candidates_from_claims(claims: list[dict[str,Any]]) -> list[dict[str,Any]]:
    out=[]
    for c in claims:
        if c.get('modality')=='FACT' and c.get('source_ref'):
            out.append({'memory_id':'memcand:'+sha256_json({'claim':c['claim_id']})[:24], 'claim_id':c['claim_id'], 'claim_text':c['claim_text'], 'source_refs':[c['source_ref']], 'status':'CANDIDATE'})
    return out

def apply_memory_decision(candidate: dict[str,Any], decision: dict[str,Any]) -> dict[str,Any]:
    if decision_allows(decision,target_ref=candidate['memory_id'],required='ACCEPT'):
        r=dict(candidate); r['status']='ACCEPTED'; r['decision_id']=decision['decision_id']; return r
    if decision_allows(decision,target_ref=candidate['memory_id'],required='REJECT'):
        r=dict(candidate); r['status']='REJECTED'; r['decision_id']=decision['decision_id']; return r
    raise ValueError('matching ACCEPT or REJECT decision required')
