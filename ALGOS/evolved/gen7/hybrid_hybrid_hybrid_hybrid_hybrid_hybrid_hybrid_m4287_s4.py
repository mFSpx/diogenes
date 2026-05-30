# DARWIN HAMMER — match 4287, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sheaf__m529_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2494_s0.py (gen6)
# born: 2026-05-29T23:54:40Z

"""Hybrid Fusion Module

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sheaf__m529_s0.py (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2494_s0.py (Algorithm B)

Mathematical Bridge:
Both parents manipulate high‑dimensional vector‑like structures.
Algorithm A reduces data to a Count‑Min sketch (a low‑dimensional count vector) and
uses tropical max‑plus algebra to update decision scores. Algorithm B works with
geometric‑algebra multivectors that are convertible to physical “morphology”
objects.  

The fusion maps the aggregated sketch counts to a 4‑component multivector,
updates it with tropical max‑plus addition driven by Hoeffding‑bound confidence,
and finally combines the updated multivector with a morphology via a
sphericity‑based similarity metric weighted by the Shannon entropy of the sketch.
"""

import math
import random
import sys
import pathlib
import hashlib
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# Utilities from Algorithm A
# ----------------------------------------------------------------------
def count_min_sketch(items: List[int], width: int = 64, depth: int = 4) -> List[List[int]]:
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def shannon_entropy(probabilities: List[float]) -> float:
    return -sum(p * math.log(p, 2) for p in probabilities if p > 0)

def sketch_row_entropy(table: List[List[int]]) -> List[float]:
    entropies = []
    for row in table:
        total = sum(row)
        if total == 0:
            entropies.append(0.0)
            continue
        probs = [c / total for c in row if c > 0]
        entropies.append(shannon_entropy(probs))
    return entropies

def sketch_to_multivector(table: List[List[int]]) -> np.ndarray:
    """Aggregate each sketch row into a scalar and pack into a 4‑D multivector."""
    agg = [sum(row) for row in table[:4]]
    # Pad/truncate to length 4
    while len(agg) < 4:
        agg.append(0)
    return np.array(agg, dtype=float)

# ----------------------------------------------------------------------
# Utilities from Algorithm B
# ----------------------------------------------------------------------
class Morphology:
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

def multivector_to_morphology(multivector: np.ndarray) -> Morphology:
    length, width, height, mass = (float(v) for v in multivector[:4])
    return Morphology(length, width, height, mass)

def morphology_to_multivector(morphology: Morphology) -> np.ndarray:
    return np.array([morphology.length, morphology.width, morphology.height, morphology.mass], dtype=float)

def sphericity_index(morph: Morphology) -> float:
    """Approximate sphericity = (π^{1/3} * (6V)^{2/3}) / A."""
    V = morph.length * morph.width * morph.height
    A = 2 * (morph.length * morph.width + morph.length * morph.height + morph.width * morph.height)
    if A == 0:
        return 0.0
    return (math.pi ** (1.0 / 3.0) * (6 * V) ** (2.0 / 3.0)) / A

def tropical_max_plus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Tropical addition (max) and multiplication (plus) on vectors."""
    return np.maximum(a, b) + np.minimum(a, b)

# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def hybrid_update_multivector(
    multivector: np.ndarray,
    gain: float,
    epsilon: float,
) -> np.ndarray:
    """
    Update a multivector using tropical max‑plus algebra.
    If gain exceeds epsilon, we add gain to each component (tropical multiplication)
    and then take the element‑wise max with the original vector.
    """
    if gain > epsilon:
        added = multivector + gain
        updated = np.maximum(multivector, added)
    else:
        updated = multivector.copy()
    return updated

def hybrid_split_decision(
    best_gain: float,
    second_best_gain: float,
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
) -> Tuple[bool, float]:
    """
    Decide whether to split a Hoeffding tree node.
    Uses Hoeffding bound to compute epsilon and a tropical‑enhanced gap.
    Returns (should_split, epsilon).
    """
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    # tropical amplification of the gap
    tropical_gap = max(0.0, gap + eps)
    should = tropical_gap > tie_threshold
    return should, eps

def hybrid_fuse(
    multivector: np.ndarray,
    sketch: List[List[int]],
    morphology: Morphology,
    gain: float,
    r: float,
    delta: float,
    n: int,
) -> np.ndarray:
    """
    Full hybrid pipeline:
    1. Compute Hoeffding epsilon and decide on a split.
    2. Update the multivector with tropical max‑plus using the gain if split.
    3. Convert sketch rows to entropy weights.
    4. Compute similarity between updated multivector (as morphology) and the
       provided morphology via sphericity indices.
    5. Return a weighted multivector that blends both sources.
    """
    should_split, eps = hybrid_split_decision(best_gain=gain,
                                              second_best_gain=gain * 0.9,
                                              r=r,
                                              delta=delta,
                                              n=n)

    updated_mv = hybrid_update_multivector(multivector, gain if should_split else 0.0, eps)

    # Entropy weighting from the sketch
    entropies = sketch_row_entropy(sketch)
    entropy_weight = sum(entropies) / len(entropies) if entropies else 0.0

    # Similarity via sphericity
    mv_morph = multivector_to_morphology(updated_mv)
    sim = sphericity_index(mv_morph) * sphericity_index(morphology)

    # Blend: original multivector + weighted similarity * entropy
    blended = updated_mv + entropy_weight * sim * np.ones_like(updated_mv)
    return blended

# ----------------------------------------------------------------------
# Demonstration Functions (at least three)
# ----------------------------------------------------------------------
def demo_sketch_to_mv():
    items = [random.randint(0, 1000) for _ in range(500)]
    sketch = count_min_sketch(items)
    mv = sketch_to_multivector(sketch)
    print("Sketch → Multivector:", mv)

def demo_hybrid_split():
    should, eps = hybrid_split_decision(best_gain=0.12,
                                        second_best_gain=0.07,
                                        r=1.0,
                                        delta=0.05,
                                        n=2000)
    print(f"Split decision: {should} (epsilon={eps:.5f})")

def demo_full_fusion():
    # Random data generation
    items = [random.randint(0, 1000) for _ in range(800)]
    sketch = count_min_sketch(items)

    # Initial multivector from sketch
    mv = sketch_to_multivector(sketch)

    # Random morphology
    morph = Morphology(length=random.uniform(1, 10),
                       width=random.uniform(1, 10),
                       height=random.uniform(1, 10),
                       mass=random.uniform(0.5, 5.0))

    # Parameters for Hoeffding bound
    gain = random.uniform(0.05, 0.15)
    r = 1.0
    delta = 0.05
    n = len(items)

    fused = hybrid_fuse(mv, sketch, morph, gain, r, delta, n)
    print("Fused multivector:", fused)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_sketch_to_mv()
    demo_hybrid_split()
    demo_full_fusion()