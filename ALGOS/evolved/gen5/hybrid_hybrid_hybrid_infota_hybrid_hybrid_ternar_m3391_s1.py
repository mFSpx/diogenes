# DARWIN HAMMER — match 3391, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_infotaxis_min_hybrid_voronoi_parti_m453_s0.py (gen3)
# parent_b: hybrid_hybrid_ternary_lens__hybrid_hybrid_hybrid_m1735_s0.py (gen4)
# born: 2026-05-29T23:49:43Z

"""
Hybrid Multivector Ternary Liquid MinHash with Voronoi Diagram.

This module fuses the mathematical structures of the Multivector Geometric Product algorithm 
and the Hybrid MinHash with Voronoi Diagram algorithm. The bridge between the two structures 
lies in integrating the Multivector geometric product within the MinHash calculation, 
utilizing the Multivector's Shannon entropy to condition the input sequences of the MinHash 
algorithm, and applying Voronoi diagram to weight the importance of each point in the similarity 
calculation.

Parents:
- hybrid_hybrid_infotaxis_min_hybrid_voronoi_parti_m453_s0.py
- hybrid_hybrid_ternary_lens__hybrid_hybrid_hybrid_m1735_s0.py
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from collections import Counter
from typing import Iterable, List, Set, Tuple, Dict

MAX64 = (1 << 64) - 1

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length *k* for the given token set."""
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Approximate Jaccard similarity via MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list."""
    lst = list(indices)
    sign = 1
    # Bubble sort; track swaps
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # Duplicate: e_i * e_i = 1, remove both
                lst.pop(j)
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""
    def __init__(self, blades):
        self.blades = blades

    def __mul__(self, other):
        result = Multivector({})
        for blade_a, coeff_a in self.blades.items():
            for blade_b, coeff_b in other.blades.items():
                blade, sign = _multiply_blades(blade_a, blade_b)
                result.blades[blade] = result.blades.get(blade, 0) + sign * coeff_a * coeff_b
        return result

def multivector_entropy(multivector: Multivector) -> float:
    """Calculate Shannon entropy of a Multivector."""
    total = sum(abs(coeff) for coeff in multivector.blades.values())
    if total == 0:
        return 0.0
    probs = [abs(coeff) / total for coeff in multivector.blades.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def voronoi_weight(points: List[Tuple[float, float]], point: Tuple[float, float]) -> float:
    """Calculate Voronoi weight for a point."""
    dists = [math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2) for x, y in points]
    return 1 / min(dists)

def hybrid_similarity(tokens_a: Iterable[str], tokens_b: Iterable[str], points: List[Tuple[float, float]]) -> float:
    """Calculate hybrid similarity between two token sets."""
    sig_a = signature(tokens_a)
    sig_b = signature(tokens_b)
    similarity_score = similarity(sig_a, sig_b)
    
    multivector_a = Multivector({frozenset(range(len(tokens_a))): 1})
    multivector_b = Multivector({frozenset(range(len(tokens_b))): 1})
    multivector_product = multivector_a * multivector_b
    entropy_score = multivector_entropy(multivector_product)
    
    weights = [voronoi_weight(points, point) for point in points]
    weighted_similarity = similarity_score * sum(weights) / len(weights)
    weighted_entropy = entropy_score * sum(weights) / len(weights)
    
    return weighted_similarity * weighted_entropy

if __name__ == "__main__":
    tokens_a = ["apple", "banana", "orange"]
    tokens_b = ["banana", "orange", "grape"]
    points = [(0, 0), (1, 1), (2, 2)]
    print(hybrid_similarity(tokens_a, tokens_b, points))