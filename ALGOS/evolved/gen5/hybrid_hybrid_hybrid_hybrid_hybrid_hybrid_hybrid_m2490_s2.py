# DARWIN HAMMER — match 2490, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_physarum_netw_m334_s0.py (gen4)
# born: 2026-05-29T23:42:32Z

"""Hybrid algorithm merging privacy‑risk/resource allocation (Parent A) with geometric‑algebraic
physarum network dynamics (Parent B).

Mathematical bridge:
- Parent A produces a scalar *risk_score* (0‑1) that quantifies reconstruction risk.
- Parent B represents network conductances as a *Multivector*; its grade‑1 (vector) part
  can be interpreted as a conductance vector **g** and its scalar part as a global
  conductance factor *c*.

The hybrid system scales the multivector by the risk_score, extracts a conductance
vector **ĝ**, and feeds it into a resource‑allocation matrix **R** (dot‑product).  The
resulting allocation vector is then privately aggregated with a differential‑privacy
mean (dp_aggregate).  An auxiliary sphericity index of a physical entity is combined
with the scalar part of the scaled multivector to produce a morphology‑aware
conductance score.

The three core functions below demonstrate this fused workflow."""
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Tuple, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Probability that a record can be re‑identified (0‑1)."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def dp_aggregate(values: Iterable[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """
    Differential‑privacy mean aggregator.
    (A real DP implementation would add Laplace noise; here we keep it deterministic.)
    """
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity: (geometric mean) / (max dimension)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


# ----------------------------------------------------------------------
# Parent B geometric algebra utilities
# ----------------------------------------------------------------------
class Multivector:
    """
    Simple Euclidean Clifford algebra (grade up to n) implementation.
    Blades are represented by frozenset of basis indices, e.g. frozenset({1,3}) == e13.
    The scalar (grade‑0) blade is frozenset().
    """

    def __init__(self, components: Dict[frozenset, float], n: int):
        self.n = int(n)
        # prune near‑zero components
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    # ------------------------------------------------------------------
    # Basic algebraic operations
    # ------------------------------------------------------------------
    def grade(self, k: int) -> "Multivector":
        """Return the k‑grade part of the multivector."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) component."""
        return self.components.get(frozenset(), 0.0)

    def vector_part(self) -> "Multivector":
        """Convenience: return the grade‑1 part."""
        return self.grade(1)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), x[0])):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        neg = Multivector({k: -v for k, v in other.components.items()}, other.n)
        return self + neg

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Multivector({k: v * other for k, v in self.components.items()}, self.n)
        if isinstance(other, Multivector):
            return geometric_product(self, other)
        raise TypeError("Unsupported multiplication")

    def __rmul__(self, scalar):
        return self * scalar


def _blade_sign(blade1: frozenset, blade2: frozenset) -> Tuple[frozenset, int]:
    """
    Compute the product blade and sign for two basis blades under Euclidean metric.
    Implements the rule: e_i * e_j = e_{i∧j} + δ_{ij}
    """
    # Convert to ordered lists for sign determination
    list1 = sorted(blade1)
    list2 = sorted(blade2)
    result_blade = list1.copy()
    sign = 1

    for idx in list2:
        if idx in result_blade:
            # e_i * e_i = 1 (scalar), remove the index and flip sign if needed
            result_blade.remove(idx)
        else:
            # Insert while counting swaps (grade‑1 vectors anticommute)
            # Determine how many existing elements are greater than idx
            swaps = sum(1 for existing in result_blade if existing > idx)
            sign *= (-1) ** swaps
            result_blade.append(idx)
            result_blade.sort()
    return frozenset(result_blade), sign


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """Full geometric product (Euclidean metric)."""
    result: Dict[frozenset, float] = {}
    for blade_a, coef_a in a.components.items():
        for blade_b, coef_b in b.components.items():
            blade_res, sign = _blade_sign(blade_a, blade_b)
            result[blade_res] = result.get(blade_res, 0.0) + sign * coef_a * coef_b
    return Multivector(result, a.n)


# ----------------------------------------------------------------------
# Hybrid core functions (the required three)
# ----------------------------------------------------------------------
def scale_multivector_by_risk(mv: Multivector, risk: float) -> Multivector:
    """
    Bridge A→B: treat the reconstruction risk as a uniform scalar weight.
    All components of the multivector are multiplied by ``risk``.
    """
    if not (0.0 <= risk <= 1.0):
        raise ValueError("risk must be in [0, 1]")
    return risk * mv


def allocate_resources(
    allocation_matrix: np.ndarray,
    conductance_mv: Multivector,
) -> np.ndarray:
    """
    Bridge B→A: extract the vector (grade‑1) part of the conductance multivector,
    treat it as a conductance vector **g**, and compute the dot‑product
    ``allocation_matrix @ g``.  The resulting raw allocation is then privately
    aggregated (DP mean) per row, yielding a final allocation vector.
    """
    if allocation_matrix.ndim != 2:
        raise ValueError("allocation_matrix must be 2‑dimensional")
    # Convert grade‑1 part to a dense numpy vector (order by basis index)
    vec_mv = conductance_mv.vector_part()
    # Determine the dimension from the matrix columns
    dim = allocation_matrix.shape[1]
    g = np.zeros(dim)
    for blade, coef in vec_mv.components.items():
        # blade is a frozenset with exactly one element
        if len(blade) != 1:
            continue
        idx = next(iter(blade)) - 1  # basis indices start at 1
        if 0 <= idx < dim:
            g[idx] = coef
    # Raw allocation per row
    raw = allocation_matrix @ g
    # Apply DP mean across the whole vector (could be per‑row, here we keep it simple)
    dp_mean = dp_aggregate(raw)
    return np.full_like(raw, dp_mean)


def morphology_conductance_score(
    morph: "Morphology",
    conductance_mv: Multivector,
) -> float:
    """
    Combine Parent A's geometric sphericity with Parent B's scalar conductance.
    The score = sphericity_index * (1 + scalar_part_of_mv).
    """
    sph = sphericity_index(morph.length, morph.width, morph.height)
    scalar = conductance_mv.scalar_part()
    return sph * (1.0 + scalar)


# ----------------------------------------------------------------------
# Simple data containers used by the hybrid functions
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float = 0.0  # optional, not used in current calculations


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Risk computation (Parent A)
    risk = reconstruction_risk_score(unique_quasi_identifiers=42, total_records=1000)
    print(f"Risk score: {risk:.4f}")

    # 2. Build a sample multivector (Parent B)
    # Basis: e1, e2, e3 (n=3)
    mv = Multivector(
        {
            frozenset(): 0.5,                 # scalar part
            frozenset({1}): 2.0,              # e1
            frozenset({2}): -1.5,             # e2
            frozenset({1, 2}): 0.3,           # e12
        },
        n=3,
    )
    print(f"Original multivector: {mv}")

    # 3. Scale by risk (Hybrid step)
    mv_scaled = scale_multivector_by_risk(mv, risk)
    print(f"Risk‑scaled multivector: {mv_scaled}")

    # 4. Resource allocation matrix (random but reproducible)
    rng = np.random.default_rng(0)
    R = rng.random((4, 3))  # 4 tasks, 3 conductance dimensions
    allocation = allocate_resources(R, mv_scaled)
    print(f"DP‑private allocation vector: {allocation}")

    # 5. Morphology + conductance score
    morph = Morphology(length=2.0, width=1.5, height=1.0)
    score = morphology_conductance_score(morph, mv_scaled)
    print(f"Morphology‑conductance score: {score:.4f}")

    # Ensure the script runs without raising exceptions
    sys.exit(0)