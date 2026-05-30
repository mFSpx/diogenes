# DARWIN HAMMER — match 740, survivor 1
# gen: 4
# parent_a: gini_coefficient.py (gen0)
# parent_b: hybrid_hybrid_infotaxis_min_hybrid_hybrid_distri_m20_s0.py (gen3)
# born: 2026-05-29T23:30:46Z

import math
import hashlib
import numpy as np
import random
import sys
import pathlib

def gini_entropy(probabilities: list[float]) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    entropy_val = -sum((p/total) * math.log(max(p, 1e-12)) for p in probabilities if p > 0)
    n = len(probabilities)
    gini = sum((2*i-n-1)*x for i,x in enumerate(sorted(probabilities, reverse=True),1))/(n*sum(probabilities))
    return (1 - entropy_val) * gini

def entropic_minhash(probabilities: list[float], k: int = 128) -> list[int]:
    tokens = [str(p) for p in probabilities]
    return signature(tokens, k)

def signature(tokens: list[str], k: int = 128) -> list[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')

def jensen_shannon_divergence(probabilities1: list[float], probabilities2: list[float]) -> float:
    total1 = sum(probabilities1)
    total2 = sum(probabilities2)
    if total1 <= 0 or total2 <= 0:
        raise ValueError('positive probability mass required')
    avg_probabilities = [(p1/total1 + p2/total2)/2 for p1, p2 in zip(probabilities1, probabilities2)]
    entropy1 = -sum((p1/total1) * math.log(max(p1, 1e-12)) for p1 in probabilities1 if p1 > 0)
    entropy2 = -sum((p2/total2) * math.log(max(p2, 1e-12)) for p2 in probabilities2 if p2 > 0)
    avg_entropy = -sum(p * math.log(max(p, 1e-12)) for p in avg_probabilities if p > 0)
    return 0.5 * (entropy1 + entropy2) - avg_entropy

def similarity(signature1: list[int], signature2: list[int]) -> float:
    return 1 - jensen_shannon_divergence([1 if x < 2**63 else 0 for x in signature1], [1 if x < 2**63 else 0 for x in signature2])

def chelydrid_strike(probabilities: list[float], k: int = 128, dt: float = 1.0) -> float:
    signature_val = entropic_minhash(probabilities, k)
    similarity_val = similarity(signature_val, [2**64 - 1] * k)
    return dt * (1 - similarity_val)

if __name__ == "__main__":
    probabilities = [0.2, 0.3, 0.5]
    print(gini_entropy(probabilities))
    print(entropic_minhash(probabilities))
    print(chelydrid_strike(probabilities))