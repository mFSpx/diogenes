#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
from spine_common import sha256_json

def missing_evidence_items(claims: list[dict[str,Any]]) -> list[dict[str,Any]]:
    items=[]
    for c in claims:
        if not c.get('source_ref') or c.get('modality') in {'HYPOTHESIS','UNKNOWN'} or not c.get('quote_text'):
            reason=[]
            if not c.get('source_ref'): reason.append('missing_source_ref')
            if c.get('modality') in {'HYPOTHESIS','UNKNOWN'}: reason.append('non_fact_modality')
            if not c.get('quote_text'): reason.append('missing_direct_quote')
            items.append({'missing_evidence_id':'missing:'+sha256_json({'claim':c.get('claim_id'),'reason':reason})[:24], 'claim_id':c.get('claim_id'), 'claim_cluster_id':c.get('cluster_id'), 'reasons':reason, 'closure_gate':'Add source ref and direct quote, or mark claim rejected.'})
    return items
