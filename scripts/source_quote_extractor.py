#!/usr/bin/env python3
from __future__ import annotations
from typing import Any

def attach_quote_context(claim: dict[str,Any], chunk_text: str, *, window: int=24) -> dict[str,Any]:
    span=claim.get('source_span') or {}; start=int(span['start']); end=int(span['end'])
    local_start=max(0, start - int(claim.get('source_ref',{}).get('chunk_start',0)))
    local_end=max(local_start, end - int(claim.get('source_ref',{}).get('chunk_start',0)))
    quote=chunk_text[local_start:local_end]
    out=dict(claim)
    out['quote_text']=quote
    out['context_before']=chunk_text[max(0,local_start-window):local_start]
    out['context_after']=chunk_text[local_end:local_end+window]
    out['quote_context_window']=window
    return out
