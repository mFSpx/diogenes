#!/usr/bin/env python3
from __future__ import annotations
import re
from collections import defaultdict
from typing import Any
from spine_common import sha256_json

def normalize_claim_text(text: str) -> str:
    return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]+',' ',text.lower())).strip()

def cluster_claims(claims: list[dict[str,Any]]) -> list[dict[str,Any]]:
    groups=defaultdict(list)
    for c in claims: groups[normalize_claim_text(c['claim_text'])].append(c)
    out=[]
    for key, rows in groups.items():
        cid='claimcluster:'+sha256_json({'normalized':key})[:24]
        for c in rows:
            r=dict(c); r['cluster_id']=cid; r['cluster_key']=key; r['cluster_size']=len(rows); out.append(r)
    return out
