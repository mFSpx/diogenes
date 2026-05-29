#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
from spine_common import sha256_json, now
VALID={'ACCEPT','REJECT','DISMISS','APPROVE'}

def make_decision(*, actor: str, decision_type: str, target_ref: str, reason: str, downstream_effect: str='none') -> dict[str,Any]:
    if decision_type not in VALID: raise ValueError('invalid decision_type')
    if not actor or not target_ref or not reason: raise ValueError('actor target_ref reason required')
    d={'actor':actor,'decision_type':decision_type,'target_ref':target_ref,'reason':reason,'downstream_effect':downstream_effect,'created_at':now()}
    d['decision_id']='decision:'+sha256_json({k:v for k,v in d.items() if k!='created_at'})[:24]
    return d

def decision_allows(decision: dict[str,Any], *, target_ref: str, required: str) -> bool:
    return decision.get('target_ref')==target_ref and decision.get('decision_type')==required
