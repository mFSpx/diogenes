# DARWIN HAMMER — match 4225, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m2718_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m1075_s2.py (gen5)
# born: 2026-05-29T23:54:20Z

"""Hybrid Morphology‑Geometric‑Allocation Algorithm

Parents:
- **Parent A** – Morphology‑based text analysis represented as multivectors
  and combined with geometric product signatures.
- **Parent B** – Adaptive work‑share allocation using Count‑Min Sketch,
  health‑score vectors, Gini fairness, and Hoeffding statistical bounds.

Mathematical Bridge:
The morphology of a text fragment is encoded as a 4‑dimensional vector
`m = (length, width, height, mass)`.  This vector is promoted to a multivector
and combined with a curvature scalar `κ` (derived from a semantic graph) via
the Clifford geometric product `G = m ⊗ (κ·1₄)`.  The flattened product
`g = vec(G)` becomes a *signature* that is fed to a Count‑Min Sketch, yielding
a compact frequency estimate `f`.  The frequency vector is merged with a
health‑score vector `h` (computed from reconstruction‑risk and recovery‑priority)
to form a demand‑health vector `d = f ⊙ h`.  Fairness of `d` is measured with the
Gini coefficient, while a Hoeffding bound on `d` decides whether a re‑allocation
is statistically justified.  The resulting allocation integrates the geometric
structure of Parent A directly into the adaptive logic of Parent B.
"""

import sys
import math
import random
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A concepts (morphology → multivector → geometric product)
# ----------------------------------------------------------------------
class Morphology:
    """Simple container for morphological dimensions."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

def morphological_vector(morph: Morphology) -> np.ndarray:
    """Encode a Morphology instance as a 4‑D vector."""
    return np.array([morph.length, morph.width, morph.height, morph.mass], dtype=float)

def geometric_product(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    """
    Clifford‑like geometric product for two 4‑D vectors.
    For simplicity we use the outer product plus the inner product on the diagonal:
        G_ij = a_i * b_j   for i != j
        G_ii = a_i * b_i   (inner product term)
    The result is flattened to a 16‑D signature vector.
    """
    if vec_a.shape != (4,) or vec_b.shape != (4,):
        raise ValueError("Both arguments must be 4‑dimensional vectors")
    outer = np.outer(vec_a, vec_b)            # 4×4 matrix
    # Replace diagonal with inner products (same as outer diagonal, kept for clarity)
    np.fill_diagonal(outer, vec_a * vec_b)
    return outer.ravel()                     # 16‑dim signature

# ----------------------------------------------------------------------
# Parent‑B concepts (curvature, Count‑Min Sketch, health, Gini, Hoeffding)
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Risk score in [0,1] based on uniqueness of quasi‑identifiers."""
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def health_score(reconstruction_risk: float, recovery_priority: float) -> float:
    """Higher health means lower risk and lower recovery priority."""
    return (1.0 - reconstruction_risk) * (1.0 - recovery_priority)

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for range r, confidence 1‑δ, and n observations."""
    if r <= 0 or not (0.0 < delta < 1.0) or n <= 0:
        raise ValueError("Invalid parameters for Hoeffding bound")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient for a list/array of non‑negative numbers."""
    arr = np.asarray(list(values), dtype=float)
    if np.any(arr < 0):
        raise ValueError("Gini coefficient is defined for non‑negative values")
    if arr.size == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(sorted_arr)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return gini

class CountMinSketch:
    """Very small Count‑Min Sketch using pairwise‑independent hash functions."""
    def __init__(self, depth: int = 4, width: int = 256, seed: int = 0):
        self.depth = depth
        self.width = width
        self.tables = np.zeros((depth, width), dtype=int)
        rng = random.Random(seed)
        self.seeds = [rng.randint(1, 2**31 - 1) for _ in range(depth)]

    def _hash(self, item: int, i: int) -> int:
        # Simple multiplicative hash modulo a large prime then modulo width
        prime = 2_147_483_647
        return ((item * self.seeds[i]) % prime) % self.width

    def update(self, item: int, increment: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.tables[i, idx] += increment

    def estimate(self, item: int) -> int:
        return min(self.tables[i, self._hash(item, i)] for i in range(self.depth))

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def compute_curvature(feature_vec: np.ndarray) -> float:
    """
    Placeholder curvature estimator.
    In a real system this would be Ollivier‑Ricci curvature on a semantic graph.
    Here we use a simple monotonic function of the vector norm.
    """
    norm = np.linalg.norm(feature_vec)
    return math.tanh(norm)  # maps to (0,1)

def hybrid_signature(morph: Morphology) -> np.ndarray:
    """
    Produce a 16‑dim hybrid signature by:
    1. Converting morphology to a vector.
    2. Computing curvature κ from that vector.
    3. Forming a curvature vector c = κ·[1,1,1,1].
    4. Applying the geometric product and flattening.
    """
    vec = morphological_vector(morph)
    κ = compute_curvature(vec)
    curvature_vec = np.full(4, κ, dtype=float)
    return geometric_product(vec, curvature_vec)  # 16‑dim

def build_frequency_sketch(signatures: List[np.ndarray]) -> CountMinSketch:
    """
    Insert each signature into a Count‑Min Sketch.
    The signature is first hashed to a 64‑bit integer via a stable hash.
    """
    cms = CountMinSketch()
    for sig in signatures:
        # Use a deterministic integer hash of the bytes representation
        h = hash(sig.tobytes()) & 0xFFFFFFFFFFFFFFFF
        cms.update(h, increment=1)
    return cms

def health_vector(models: List[str],
                  unique_ids: List[int],
                  total_records: List[int],
                  recovery_priorities: List[float]) -> Dict[str, float]:
    """
    Compute a health score per model.
    """
    if not (len(models) == len(unique_ids) == len(total_records) == len(recovery_priorities)):
        raise ValueError("All input lists must have the same length")
    healths = {}
    for name, uq, tot, rp in zip(models, unique_ids, total_records, recovery_priorities):
        risk = reconstruction_risk_score(uq, tot)
        healths[name] = health_score(risk, rp)
    return healths

def allocate_workshare(models: List[str],
                       signatures: List[np.ndarray],
                       healths: Dict[str, float],
                       delta: float = 0.05) -> Dict[str, float]:
    """
    Core hybrid allocation:
    1. Build a Count‑Min Sketch from the signatures.
    2. Estimate frequency for each model by hashing the model name.
    3. Combine frequency (demand) with health (capacity) via element‑wise product.
    4. Compute Gini coefficient of the combined vector.
    5. Compute Hoeffding bound; if the bound exceeds the observed range, keep
       the current allocation, otherwise normalize the combined vector to sum to 1.
    Returns a dict mapping model name → allocated share.
    """
    cms = build_frequency_sketch(signatures)

    # Estimate demand per model
    demand = {}
    for name in models:
        h = hash(name.encode('utf-8')) & 0xFFFFFFFFFFFFFFFF
        demand[name] = cms.estimate(h)

    # Combine demand with health (element‑wise multiplication)
    combined = np.array([demand[m] * healths.get(m, 0.0) for m in models], dtype=float)

    # Fairness and statistical check
    gini = gini_coefficient(combined)
    if len(combined) == 0:
        raise ValueError("No models provided")
    observed_range = combined.max() - combined.min()
    bound = hoeffding_bound(observed_range, delta, n=len(combined))

    # If Hoeffding bound is large relative to the range, we consider the estimate unstable
    if bound > observed_range:
        # Return uniform allocation as a safe fallback
        uniform = 1.0 / len(models)
        return {m: uniform for m in models}

    # Otherwise normalize combined vector to obtain allocation shares
    total = combined.sum()
    if total == 0:
        # Avoid division by zero – fallback to uniform
        uniform = 1.0 / len(models)
        return {m: uniform for m in models}
    allocation = {m: (demand[m] * healths.get(m, 0.0)) / total for m in models}
    # Attach diagnostic info as hidden attributes (optional)
    allocation["_gini"] = gini
    allocation["_hoeffding_bound"] = bound
    return allocation

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a few synthetic Morphology objects
    morphs = [
        Morphology(1.2, 0.5, 0.3, 2.0),
        Morphology(0.8, 0.7, 0.4, 1.5),
        Morphology(1.5, 0.6, 0.5, 2.2),
    ]

    # Compute hybrid signatures
    sigs = [hybrid_signature(m) for m in morphs]

    # Define dummy models and health‑related data
    model_names = ["model_A", "model_B", "model_C"]
    unique_ids = [10, 5, 8]
    total_records = [100, 100, 100]
    recovery_priorities = [0.2, 0.5, 0.1]

    # Compute health scores
    healths = health_vector(model_names, unique_ids, total_records, recovery_priorities)

    # Run allocation
    allocation = allocate_workshare(model_names, sigs, healths)

    # Print results
    print("Health scores:", healths)
    print("Allocation shares:")
    for m in model_names:
        print(f"  {m}: {allocation[m]:.4f}")
    print("Diagnostic - Gini:", allocation["_gini"])
    print("Diagnostic - Hoeffding bound:", allocation["_hoeffding_bound"])