#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
from spine_common import sha256_json

def chunk_text(text: str, *, source_ref: dict[str,Any], max_chars: int = 500, overlap: int = 0) -> list[dict[str,Any]]:
    if max_chars <= 0:
        raise ValueError('max_chars must be positive')
    if overlap < 0 or overlap >= max_chars:
        raise ValueError('overlap must be >=0 and < max_chars')
    chunks=[]; start=0; idx=0
    while start < len(text):
        end=min(len(text), start+max_chars)
        chunk=text[start:end]
        row={'chunk_id':'chunk:'+sha256_json({'source_ref':source_ref,'start':start,'end':end,'text':chunk})[:24],'chunk_index':idx,'start':start,'end':end,'text':chunk,'source_ref':source_ref}
        chunks.append(row)
        idx += 1
        if end == len(text): break
        start=end-overlap
    if not chunks:
        chunks.append({'chunk_id':'chunk:'+sha256_json({'source_ref':source_ref,'empty':True})[:24],'chunk_index':0,'start':0,'end':0,'text':'','source_ref':source_ref})
    return chunks
