# DARWIN HAMMER — match 4225, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m2718_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m1075_s2.py (gen5)
# born: 2026-05-29T23:54:20Z

"""Hybrid Morphology‑Geometric‑CountMin Allocation

Parents:
- **Parent A** – Morphology‑based multivector representation with a geometric
  product (from *hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m2718_s0.py*).
- **Parent B** – Health‑score driven allocation using a Count‑Min Sketch,
  Hoeffding bound and Gini coefficient (from *hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m1075_s2.py*).

Mathematical Bridge:
Each text token is described by a 4‑component morphology vector  
`m = (length, width, height, mass)`.  The geometric product of two such
vectors yields a 4×4 interaction matrix `G = m ⊗ m'`.  By projecting the
first three components of `m` (augmented with a curvature scalar κ) into a
3‑D point `p = (x, y, z)`, we can hash `p` into a Count‑Min Sketch.  The sketch
produces a frequency estimate `f(p)` that is interpreted as a *demand*
signal for the corresponding model.  The demand vector `f` is combined with
the health‑score vector `h` (computed from reconstruction‑risk and recovery‑priority)
into a single resource‑allocation vector `a = f ⊙ h` (element‑wise product).
Fairness of the allocation is measured with the Gini coefficient, while a
Hoeffding bound on the range of `a` decides whether a re‑allocation is
statistically justified.  This code implements the full hybrid pipeline."""

import sys
import math
import random
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A components (morphology + geometric product)
# ----------------------------------------------------------------------
class Morphology:
    """Simple 4‑dimensional morphological descriptor."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)
        self.mass = float(mass)

    def as_vector(self) -> np.ndarray:
        """Return the morphology as a length‑4 numpy column vector."""
        return np.array([self.length, self.width, self.height, self.mass], dtype=float)


def geometric_product(m1: Morphology, m2: Morphology) -> np.ndarray:
    """
    Compute a simplified geometric product of two morphology vectors.
    For the purposes of this hybrid algorithm we use the outer product
    followed by a symmetric contraction that mimics the Clifford product.
    The result is a 4×4 matrix.
    """
    v1 = m1.as_vector().reshape(4, 1)          # column
    v2 = m2.as_vector().reshape(1, 4)          # row
    outer = v1 @ v2                             # 4×4 outer product
    # Symmetric part (v·w) on the diagonal, antisymmetric part kept off‑diagonal
    sym = np.diag(np.diag(outer))
    anti = outer - sym
    return sym + anti  # simply returns the full outer product


def curvature_from_morph(m: Morphology) -> float:
    """
    Placeholder curvature κ computed from the morphology.
    A plausible surrogate is the normalized mass‑to‑volume ratio.
    """
    volume = max(m.length * m.width * m.height, 1e-9)
    return m.mass / volume


# ----------------------------------------------------------------------
# Parent‑B components (Count‑Min Sketch, health score, Hoeffding, Gini)
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def health_score(reconstruction_risk: float, recovery_priority: float) -> float:
    """Higher health means lower risk and lower recovery priority."""
    return (1.0 - reconstruction_risk) * (1.0 - recovery_priority)


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a range r, confidence 1‑δ, and n observations."""
    if r <= 0 or not (0.0 < delta < 1.0) or n <= 0:
        raise ValueError("Invalid parameters for Hoeffding bound")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def gini_coefficient(values: Iterable[float]) -> float:
    """Compute Gini coefficient of a 1‑D iterable of non‑negative numbers."""
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        raise ValueError("Gini coefficient is undefined for negative values")
    sorted_arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(sorted_arr)
    sum_ = cumulative[-1]
    if sum_ == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_) / n
    return gini


class CountMinSketch:
    """
    Simple Count‑Min Sketch implementation with pairwise‑independent hash
    functions generated from Python's built‑in ``hash`` combined with a seed.
    """
    def __init__(self, width: int = 200, depth: int = 5, seed: int = 0):
        self.width = int(width)
        self.depth = int(depth)
        self.seed = int(seed)
        self.table = np.zeros((self.depth, self.width), dtype=int)
        random.seed(self.seed)

    def _hash(self, item: Tuple[float, float, float], i: int) -> int:
        """Hash a 3‑D point into the i‑th row."""
        x, y, z = item
        combined = f"{x:.6f}:{y:.6f}:{z:.6f}:{i}"
        return (hash(combined) + i * 31) % self.width

    def add(self, item: Tuple[float, float, float], increment: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.table[i, idx] += increment

    def estimate(self, item: Tuple[float, float, float]) -> int:
        """Return the minimum count across the depth (standard CMS query)."""
        mins = []
        for i in range(self.depth):
            idx = self._hash(item, i)
            mins.append(self.table[i, idx])
        return int(min(mins))


# ----------------------------------------------------------------------
# Hybrid Functions (demonstrating the fused operation)
# ----------------------------------------------------------------------
def build_feature_point(m: Morphology, curvature_weight: float = 1.0) -> Tuple[float, float, float]:
    """
    Convert a morphology into a 3‑D point (x, y, z) where the curvature
    κ is injected into the x‑coordinate.
    """
    κ = curvature_from_morph(m)
    x = m.length + curvature_weight * κ
    y = m.width
    z = m.height
    return (x, y, z)


def compute_demand_vector(morphs: List[Morphology], sketch: CountMinSketch) -> np.ndarray:
    """
    Feed each morphology into the Count‑Min Sketch and collect the estimated
    frequencies into a vector.  The order follows the input list.
    """
    demand = []
    for m in morphs:
        pt = build_feature_point(m)
        sketch.add(pt, increment=1)
        demand.append(sketch.estimate(pt))
    return np.array(demand, dtype=float)


def hybrid_allocation(
    morphs: List[Morphology],
    reconstruction_risk: float,
    recovery_priority: float,
    delta: float = 0.05,
    curvature_weight: float = 1.0,
) -> Tuple[np.ndarray, float, bool]:
    """
    End‑to‑end hybrid allocation:

    1. Compute health score h.
    2. Build a Count‑Min Sketch from the morphologies and obtain demand vector f.
    3. Form allocation vector a = f ⊙ h (element‑wise product).
    4. Evaluate fairness with Gini coefficient.
    5. Use Hoeffding bound on the range of a to decide if re‑allocation is justified.

    Returns:
        a            – allocation vector (numpy array)
        gini         – fairness measure (float)
        reallocate   – boolean flag indicating statistical justification
    """
    # 1. health score (scalar, broadcast to each model)
    h_scalar = health_score(reconstruction_risk, recovery_priority)
    health_vec = np.full(len(morphs), h_scalar, dtype=float)

    # 2. demand vector via Count‑Min Sketch
    sketch = CountMinSketch(width=256, depth=7, seed=42)
    demand_vec = compute_demand_vector(morphs, sketch)

    # 3. allocation vector (element‑wise product)
    allocation = demand_vec * health_vec

    # 4. fairness metric
    gini = gini_coefficient(allocation)

    # 5. Hoeffding bound decision
    r = float(np.max(allocation) - np.min(allocation))
    n = max(1, len(allocation))
    bound = hoeffding_bound(r, delta, n)
    # If the bound is smaller than a small epsilon, we consider the allocation stable
    reallocate = bound > 0.1 * r

    return allocation, gini, reallocate


def morphology_interaction_matrix(morphs: List[Morphology]) -> np.ndarray:
    """
    Compute a block matrix where each block (i, j) is the geometric product
    of morphologies i and j.  The resulting matrix has shape (4·N, 4·N).
    """
    n = len(morphs)
    blocks = []
    for i in range(n):
        row_blocks = []
        for j in range(n):
            prod = geometric_product(morphs[i], morphs[j])
            row_blocks.append(prod)
        blocks.append(np.hstack(row_blocks))
    return np.vstack(blocks)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a modest list of synthetic morphologies
    random.seed(0)
    morph_list = [
        Morphology(
            length=random.uniform(1.0, 10.0),
            width=random.uniform(1.0, 5.0),
            height=random.uniform(0.5, 3.0),
            mass=random.uniform(0.1, 2.0),
        )
        for _ in range(8)
    ]

    # Arbitrary risk/priority values
    recon_risk = reconstruction_risk_score(unique_quasi_identifiers=23, total_records=150)
    recovery_prio = 0.3

    alloc_vec, fairness, need_realloc = hybrid_allocation(
        morphs=morph_list,
        reconstruction_risk=recon_risk,
        recovery_priority=recovery_prio,
        delta=0.05,
        curvature_weight=0.8,
    )

    print("Allocation vector:", alloc_vec)
    print("Gini coefficient (fairness):", round(fairness, 4))
    print("Re‑allocation justified?:", need_realloc)

    # Demonstrate interaction matrix size
    interaction = morphology_interaction_matrix(morph_list)
    print("Interaction matrix shape:", interaction.shape)