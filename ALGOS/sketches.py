#!/usr/bin/env python3
"""Count-min, HLL-lite, and MinHash LSH helpers."""
from __future__ import annotations
import hashlib
from collections import defaultdict
from typing import Iterable

def count_min_sketch(items: Iterable[str], width: int=64, depth: int=4) -> list[list[int]]:
    table=[[0]*width for _ in range(depth)]
    for item in items:
        for d in range(depth): table[d][int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(),16)%width]+=1
    return table
def hyperloglog_cardinality(items: Iterable[str]) -> int: return len(set(items))
def minhash_lsh_index(docs: dict[str,set[str]]) -> dict[str,list[str]]:
    buckets=defaultdict(list)
    for doc_id, shingles in docs.items():
        key=min((hashlib.sha1(s.encode()).hexdigest()[:6] for s in shingles), default='empty')
        buckets[key].append(doc_id)
    return dict(buckets)
