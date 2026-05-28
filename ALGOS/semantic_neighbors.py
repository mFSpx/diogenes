#!/usr/bin/env python3
"""In-memory semantic-neighbor enclave."""
from __future__ import annotations
import math
_ENCLAVE: dict[str,list[float]]={}
def clear_enclave() -> None: _ENCLAVE.clear()
def register_document(doc_id: str, vector: list[float]) -> None: _ENCLAVE[doc_id]=vector
def _cos(a,b):
    den=math.sqrt(sum(x*x for x in a))*math.sqrt(sum(y*y for y in b)); return 0.0 if den==0 else sum(x*y for x,y in zip(a,b))/den
def semantic_neighbors(doc_id: str, k: int=5) -> list[tuple[str,float]]:
    v=_ENCLAVE[doc_id]
    return sorted(((d,_cos(v,w)) for d,w in _ENCLAVE.items() if d!=doc_id), key=lambda x:(-x[1],x[0]))[:k]
