#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
from spine_common import receipt
from content_store import ContentStore

def write_receipt(component: str, payload: dict[str,Any], *, case_id: str|None=None) -> dict[str,Any]:
    store=ContentStore(case_id=case_id)
    ref=store.put_json(payload, media_type='application/vnd.lucidota.receipt+json')
    payload=dict(payload); payload['content_ref']=ref['content_ref']; payload['content_sha256']=ref['sha256']
    p=receipt(component,payload,root='05_OUTPUTS/receipts')
    payload['receipt_path']=str(p)
    return payload
