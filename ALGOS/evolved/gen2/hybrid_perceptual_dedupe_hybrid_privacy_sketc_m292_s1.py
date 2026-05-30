# DARWIN HAMMER — match 292, survivor 1
# gen: 2
# parent_a: perceptual_dedupe.py (gen0)
# parent_b: hybrid_privacy_sketches_m15_s1.py (gen1)
# born: 2026-05-29T23:28:03Z

"""
Hybrid algorithm fusing perceptual hashing (perceptual_dedupe.py) with 
reconstruction risk scoring and MinHash LSH indexing (hybrid_privacy_sketches_m15_s1.py).
The mathematical bridge lies in utilizing MinHash LSH to efficiently cluster similar 
records based on their perceptual hashes, then applying reconstruction risk scoring 
to evaluate the anonymization risk of these clusters.

This hybrid approach leverages the strengths of both parent algorithms: 
1. Perceptual hashing for efficient similarity search and duplicate detection.
2. Reconstruction risk scoring and MinHash LSH for evaluating anonymization risk 
   and efficient indexing of similar records.

The fusion integrates the governing equations of both parents by:
- Using MinHash LSH to cluster similar records based on their perceptual hashes.
- Applying reconstruction risk scoring to these clusters to evaluate their anonymization risk.
"""

import numpy as np
from collections import defaultdict
import hashlib
import math
import random
import sys
import pathlib

def compute_dhash(values: list[float]) -> int:
    bits=0
    for i in range(len(values)-1): bits=(bits<<1)|int(values[i] > values[i+1])
    return bits

def compute_phash(values: list[float]) -> int:
    if not values: return 0
    avg=sum(values)/len(values); bits=0
    for v in values[:64]: bits=(bits<<1)|int(v>=avg)
    return bits

def hamming_distance(a: int, b: int) -> int: 
    return (a^b).bit_count()

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers/total_records))

def minhash_lsh_index(docs: dict[str,set[str]]) -> dict[str,list[str]]:
    buckets=defaultdict(list)
    for doc_id, shingles in docs.items():
        key=min((hashlib.sha1(s.encode()).hexdigest()[:6] for s in shingles), default='empty')
        buckets[key].append(doc_id)
    return dict(buckets)

def hybrid_cluster_and_risk(documents: dict[str,list[float]], redact_keys: set[str]|None=None) -> dict:
    perceptual_hashes = {doc_id: compute_phash(values) for doc_id, values in documents.items()}
    clusters = cluster_by_phash(perceptual_hashes)
    
    risk_scores = {}
    for cluster in clusters:
        quasi_identifiers = set()
        for doc_id in cluster:
            quasi_identifier = tuple(sorted((k, v) for k, v in documents[doc_id].items() if k not in redact_keys))
            quasi_identifiers.add(quasi_identifier)
        risk_score = reconstruction_risk_score(len(quasi_identifiers), len(cluster))
        risk_scores[tuple(cluster)] = risk_score
    
    return risk_scores

def cluster_by_phash(hashes: dict[str,int], max_distance: int=4) -> list[list[str]]:
    clusters=[]
    for k,h in hashes.items():
        for c in clusters:
            if hamming_distance(h, hashes[c[0]]) <= max_distance: c.append(k); break
        else: clusters.append([k])
    return clusters

def dp_aggregate(values: list[float], epsilon: float=1.0, sensitivity: float=1.0) -> float:
    noise = np.random.laplace(loc=0, scale=sensitivity/epsilon)
    return sum(values) + noise

if __name__ == "__main__":
    documents = {
        'doc1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'doc2': [2.0, 3.0, 4.0, 5.0, 6.0],
        'doc3': [1.0, 2.0, 3.0, 4.0, 5.0],
    }
    risk_scores = hybrid_cluster_and_risk(documents)
    print(risk_scores)