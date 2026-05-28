#!/usr/bin/env python3
"""Perceptual hash-lite dedupe helpers for visual/evidence channels."""
from __future__ import annotations

def compute_dhash(values: list[float]) -> int:
    bits=0
    for i in range(len(values)-1): bits=(bits<<1)|int(values[i] > values[i+1])
    return bits
def compute_phash(values: list[float]) -> int:
    if not values: return 0
    avg=sum(values)/len(values); bits=0
    for v in values[:64]: bits=(bits<<1)|int(v>=avg)
    return bits
def hamming_distance(a: int, b: int) -> int: return (a^b).bit_count()
def cluster_by_phash(hashes: dict[str,int], max_distance: int=4) -> list[list[str]]:
    clusters=[]
    for k,h in hashes.items():
        for c in clusters:
            if hamming_distance(h, hashes[c[0]]) <= max_distance: c.append(k); break
        else: clusters.append([k])
    return clusters
