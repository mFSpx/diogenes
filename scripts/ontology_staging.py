#!/usr/bin/env python3
from __future__ import annotations
import re
from typing import Any
from spine_common import sha256_json
ENTITY_RE=re.compile(r'\b[A-Z][A-Za-z0-9_\-]{2,}\b')

def validate_staging_row(row: dict[str,Any]) -> None:
    if row.get('status') != 'STAGED':
        raise ValueError('ontology staging rows must remain STAGED')
    if not (0 <= int(row.get('confidence_bps',-1)) <= 10000):
        raise ValueError('confidence_bps out of bounds')
    if row.get('modality') not in {'FACT','HYPOTHESIS','UNKNOWN'}:
        raise ValueError('invalid modality')
    if not row.get('source_span') or row['source_span'].get('start') is None or row['source_span'].get('end') is None:
        raise ValueError('missing source span')
    if not row.get('source_ref') or not row['source_ref'].get('custody_id'):
        raise ValueError('missing source custody ref')

def stage_claim_from_sentence(sentence: str, *, chunk: dict[str,Any], start: int, end: int, method: str='deterministic_sentence_v1') -> dict[str,Any]:
    text=sentence.strip()
    lower=text.lower()
    if any(w in lower for w in ['maybe','might','possibly','hypothesis']): modality='HYPOTHESIS'
    elif text.endswith('?'): modality='UNKNOWN'
    else: modality='FACT'
    entities=sorted(set(ENTITY_RE.findall(text)))
    row={'staging_id':'staged:'+sha256_json({'chunk_id':chunk['chunk_id'],'start':start,'end':end,'text':text})[:24], 'claim_text':text, 'entity_candidates':entities, 'event_candidates':[], 'modality':modality, 'confidence_bps':7000 if modality=='FACT' else 4500, 'extraction_method':method, 'status':'STAGED', 'promotion_blocker':'operator_review_required', 'source_ref':chunk['source_ref'] | {'chunk_id':chunk['chunk_id']}, 'source_span':{'start':start,'end':end}}
    validate_staging_row(row)
    return row
