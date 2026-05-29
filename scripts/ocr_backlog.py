#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
from spine_common import sha256_json

def ocr_jobs_from_parse_records(records: list[dict[str,Any]], *, ocr_allowed: bool=False, ocr_available: bool=False) -> list[dict[str,Any]]:
    jobs=[]
    for r in records:
        if r.get('status') not in {'OCR_REQUIRED','OCR_BLOCKED'}: continue
        state='BLOCKED' if not (ocr_allowed and ocr_available) else 'CREATED'
        jobs.append({'work_order_id':'ocr:'+sha256_json({'source':r.get('source_path'),'custody':r.get('custody_id')})[:24], 'kind':'ocr_backlog', 'source_path':r.get('source_path'), 'custody_id':r.get('custody_id'), 'status':state, 'blocker':None if state=='CREATED' else 'OCR_NOT_ALLOWED_OR_BACKEND_MISSING', 'closure_gate':'Run OCR backend and attach parse text to custody record.'})
    return jobs
