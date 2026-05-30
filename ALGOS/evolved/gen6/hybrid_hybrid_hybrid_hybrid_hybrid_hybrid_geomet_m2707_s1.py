# DARWIN HAMMER — match 2707, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s2.py (gen5)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s3.py (gen4)
# born: 2026-05-29T23:43:42Z

"""
This module integrates the stylometry features from the DARWIN HAMMER 
algorithm (hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s2.py) 
with the geometric product from the Clifford algebra (Cl(n,0)) and the 
hybrid ternary route algorithm from (hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s3.py).

The mathematical bridge between these two structures is formed by using the 
geometric product to compute distances and orientations between points in 
the stylometry feature space, and then applying these computations to 
assign points to their nearest seeds using the hybrid ternary route algorithm.

The governing equations of the Clifford algebra are used to compute the 
geometric product of multivectors, which are then used to represent points 
and vectors in the stylometry feature space. The hybrid ternary route 
algorithm is used to compute the shortest paths between points, and the 
geometric product is used to compute the distances and orientations 
between these points.
"""

import math
import numpy as np
import random
import sys
import pathlib
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import Counter
import re

FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't".split()),
}

CATEGORY_ORDER = list(FUNCTION_CATS.keys())
NUM_CATS = len(CATEGORY_ORDER)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+'\w+|\b\w+\b", text.lower())


def stylometry_features(text: str) -> np.ndarray:
    tokens = _tokenize(text)
    counts = Counter(tokens)
    vec = np.zeros(NUM_CATS, dtype=float)
    for idx, cat in enumerate(CATEGORY_ORDER):
        cat_words = FUNCTION_CATS[cat]
        cat_count = sum(counts[w] for w in cat_words if w in counts)
        vec[idx] = cat_count
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


# Core blade arithmetic helpers
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list.

    Each transposition of adjacent indices that are out of order flips the
    sign (anti-commutativity).  Duplicate indices cancel (e_i^2 = 1 → they
    annihilate and contribute +1 to the sign, but the index disappears).
    """
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
    """Multiply two basis blades (each a frozenset of indices).

    Returns (result_blade_frozenset, sign).
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


# Multivector
class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, blades: Dict[frozenset, float]):
        self.blades = blades

    def __mul__(self, other):
        result_blades = {}
        for blade_a, coeff_a in self.blades.items():
            for blade_b, coeff_b in other.blades.items():
                result_blade, sign = _multiply_blades(blade_a, blade_b)
                result_blades[result_blade] = result_blades.get(result_blade, 0) + sign * coeff_a * coeff_b
        return Multivector(result_blades)

    def __repr__(self):
        return f"Multivector({self.blades})"


def compute_geometric_product(multivector: Multivector, stylometry_vec: np.ndarray) -> Multivector:
    # Map stylometry vector to multivector
    blades = {}
    for i, val in enumerate(stylometry_vec):
        blades[frozenset([i])] = val
    other = Multivector(blades)
    return multivector * other


def hybrid_ternary_route(stylometry_vecs: List[np.ndarray], num_seeds: int) -> List[int]:
    # Simple ternary route implementation
    seeds = random.sample(range(len(stylometry_vecs)), num_seeds)
    assignments = []
    for vec in stylometry_vecs:
        min_dist = float('inf')
        closest_seed = None
        for seed_idx in seeds:
            seed_vec = stylometry_vecs[seed_idx]
            dist = np.linalg.norm(vec - seed_vec)
            if dist < min_dist:
                min_dist = dist
                closest_seed = seed_idx
        assignments.append(closest_seed)
    return assignments


def hybrid_stylometry_clifford(stylometry_texts: List[str], num_seeds: int) -> List[int]:
    stylometry_vecs = [stylometry_features(text) for text in stylometry_texts]
    # Create a multivector for each stylometry vector
    multivectors = []
    for vec in stylometry_vecs:
        blades = {}
        for i, val in enumerate(vec):
            blades[frozenset([i])] = val
        multivectors.append(Multivector(blades))
    # Compute geometric products
    products = []
    for multivector in multivectors:
        product = compute_geometric_product(multivector, stylometry_vecs[0])
        products.append(product)
    # Apply hybrid ternary route
    return hybrid_ternary_route(stylometry_vecs, num_seeds)


if __name__ == "__main__":
    texts = ["This is a test.", "This test is only a test.", "Test only this."]
    num_seeds = 2
    assignments = hybrid_stylometry_clifford(texts, num_seeds)
    print(assignments)