# DARWIN HAMMER — match 5059, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_rectified_flo_hybrid_hybrid_endpoi_m519_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_percep_m1761_s1.py (gen5)
# born: 2026-05-29T23:59:34Z

"""
Hybrid Multivector-Perceptual RBF Allocation with Lexical Style Matrix
Parents:
- hybrid_hybrid_rectified_flo_hybrid_hybrid_endpoi_m519_s3.py (Lexical Style Matrix)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_percep_m1761_s1.py (Multivector-Perceptual RBF Allocation)

The mathematical bridge between the two parents lies in the distance-based similarity 
and the conversion of Multivector into a hyper-dimensional vector. 
The Lexical Style Matrix (LSM) from Parent A can be used to modulate the Multivector 
components from Parent B, which are then fed into the RBF kernel.

This hybrid algorithm integrates the LSM vector calculation with the Multivector 
processing and RBF allocation, creating a unified system that respects both 
algebraic structure and perceptual clustering.
"""

import numpy as np
import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, FrozenSet
from pathlib import Path

# ----------------------------------------------------------------------
# 1. Lexical function categories (unchanged)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

# ----------------------------------------------------------------------
# 2. Utility functions
# ----------------------------------------------------------------------
def words(text: str) -> List[str]:
    """Tokenise a string into alphabetic lower‑case words."""
    return [w for w in (text or "").lower().split() if w.isalpha()]

def lsm_vector(text: str) -> Dict[str, float]:
    """
    Compute a “lexical style matrix” (LSM) vector.
    Returns a probability for each FUNCTION_CATS entry; missing categories get 0.0.
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    lsm: Dict[str, float] = {}
    for cat, vocab in FUNCTION_CATS.items():
        lsm[cat] = sum(cnt[w] for w in vocab) / total
    return lsm

def stable_hash(text: str) -> int:
    """Deterministic 64‑bit hash based on SHA‑256."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)

# ----------------------------------------------------------------------
# Parent B – Multivector utilities (geometric algebra)
# ----------------------------------------------------------------------
def _blade_sign(indices: Tuple[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate index → blade vanishes
                lst.pop(j)
                lst.pop(j)  # second copy now at same position
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(tuple(combined))
    return frozenset(result), sign

def multivector_to_hypervector(multivector: Dict[FrozenSet[int], float]) -> np.ndarray:
    """Convert a Multivector into a hyper-dimensional vector."""
    dim = max(max(indices) for indices in multivector.keys()) + 1
    vector = np.zeros(dim)
    for blade, value in multivector.items():
        for index in blade:
            vector[index] += value
    return vector

def rbf_kernel(vector_a: np.ndarray, vector_b: np.ndarray, epsilon: float) -> float:
    """Compute the RBF kernel between two hyper-dimensional vectors."""
    distance = np.linalg.norm(vector_a - vector_b)
    return math.exp(-distance**2 / (2 * epsilon**2))

def hybrid_operation(text: str, multivector: Dict[FrozenSet[int], float], epsilon: float) -> float:
    """Perform the hybrid operation."""
    lsm = lsm_vector(text)
    hypervector = multivector_to_hypervector(multivector)
    modulated_hypervector = hypervector * np.array([lsm["pronoun"] + lsm["article"] for _ in range(len(hypervector))])
    return rbf_kernel(modulated_hypervector, hypervector, epsilon)

# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def test_hybrid_operation():
    text = "This is a test sentence."
    multivector = {frozenset([0, 1]): 0.5, frozenset([2]): 0.3}
    epsilon = 0.1
    result = hybrid_operation(text, multivector, epsilon)
    print(result)

def test_multivector_to_hypervector():
    multivector = {frozenset([0, 1]): 0.5, frozenset([2]): 0.3}
    hypervector = multivector_to_hypervector(multivector)
    print(hypervector)

def test_rbf_kernel():
    vector_a = np.array([1, 2, 3])
    vector_b = np.array([4, 5, 6])
    epsilon = 0.1
    result = rbf_kernel(vector_a, vector_b, epsilon)
    print(result)

if __name__ == "__main__":
    test_hybrid_operation()
    test_multivector_to_hypervector()
    test_rbf_kernel()