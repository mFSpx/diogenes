# DARWIN HAMMER — match 1394, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m871_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decrea_m527_s5.py (gen5)
# born: 2026-05-29T23:36:01Z

import math
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple, List, Iterable, Union, Optional

import numpy as np


# ----------------------------------------------------------------------
# Multivector core (enhanced from Parent A)
# ----------------------------------------------------------------------
Blade = Tuple[int, ...]  # sorted tuple of basis indices, e.g. (1,3)


def _canonical_blade(blade: Iterable[int]) -> Tuple[Blade, int]:
    """
    Return the canonical (sorted, duplicate‑free) blade and the sign
    resulting from the reordering required by the exterior algebra.
    """
    lst = list(blade)
    sign = 1
    i = 0
    while i < len(lst):
        # bubble sort style swap to enforce ascending order
        j = i
        while j > 0 and lst[j - 1] > lst[j]:
            lst[j - 1], lst[j] = lst[j], lst[j - 1]
            sign = -sign
            j -= 1
        # cancel duplicate indices (e_i ^ e_i = 0)
        if j > 0 and lst[j - 1] == lst[j]:
            lst.pop(j)      # remove the second
            lst.pop(j - 1)  # remove the first
            sign = sign  # sign unchanged by cancellation
            i = max(i - 2, 0)
            continue
        i += 1
    return tuple(lst), sign


def _geometric_product_blades(a: Blade, b: Blade) -> Tuple[Blade, int]:
    """
    Compute the geometric product of two basis blades.
    The result is a blade and an associated sign.
    """
    # concatenate the index lists and reduce to canonical form
    combined = list(a) + list(b)
    return _canonical_blade(combined)


class Multivector:
    """
    Simple dense multivector implementation supporting addition,
    scalar multiplication, grade extraction and the full geometric product.
    Blades are stored as sorted tuples of basis indices.
    """

    def __init__(self, components: Optional[Dict[Blade, float]] = None, n: int = 0):
        self.n = int(n)
        self.components: Dict[Blade, float] = {}
        if components:
            for blade, coeff in components.items():
                if abs(coeff) > 1e-15:
                    self.components[tuple(sorted(blade))] = float(coeff)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def scalar(cls, value: float, n: int = 0) -> "Multivector":
        return cls({(): value}, n)

    @classmethod
    def vector(cls, coords: Iterable[float], n: int) -> "Multivector":
        comps = { (i,): float(c) for i, c in enumerate(coords) if abs(c) > 1e-15 }
        return cls(comps, n)

    # ------------------------------------------------------------------
    # Basic algebra
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coeff in other.components.items():
            result[blade] = result.get(blade, 0.0) + coeff
            if abs(result[blade]) < 1e-15:
                del result[blade]
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self.components.items()}, self.n)

    def __rmul__(self, scalar: float) -> "Multivector":
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Multivector({b: scalar * c for b, c in self.components.items()}, self.n)

    def __mul__(self, other: Union["Multivector", float]) -> "Multivector":
        if isinstance(other, (int, float)):
            return other * self
        # geometric product
        result: Dict[Blade, float] = {}
        for a_blade, a_coeff in self.components.items():
            for b_blade, b_coeff in other.components.items():
                blade, sign = _geometric_product_blades(a_blade, b_blade)
                coeff = a_coeff * b_coeff * sign
                result[blade] = result.get(blade, 0.0) + coeff
        # prune near‑zero entries
        result = {b: c for b, c in result.items() if abs(c) > 1e-15}
        return Multivector(result, self.n)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def scalar_part(self) -> float:
        """Return the coefficient of the grade‑0 (scalar) blade."""
        return self.components.get((), 0.0)

    def grade(self, k: int) -> "Multivector":
        """Extract the sub‑multivector consisting of blades of grade k."""
        return Multivector(
            {b: c for b, c in self.components.items() if len(b) == k},
            self.n,
        )

    def norm(self) -> float:
        """Euclidean norm of the coefficient vector."""
        return math.sqrt(sum(c * c for c in self.components.values()))

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        terms = [f"{c:.3g}{''.join(map(str, b)) or '1'}" for b, c in self.components.items()]
        return f"Multivector({', '.join(terms)})"


# ----------------------------------------------------------------------
# Regret & Gini utilities (enhanced from Parent B)
# ----------------------------------------------------------------------
def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient for a 1‑D array.
    Returns 0 for a uniform distribution and 1 for maximal inequality.
    """
    if values.ndim != 1:
        raise ValueError("gini_coefficient expects a 1‑D array")
    if np.any(values < 0):
        raise ValueError("Gini coefficient is defined for non‑negative values")
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals, dtype=float)
    if cumulative[-1] == 0:
        return 0.0
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


def compute_regret_gini(costs: np.ndarray, risks: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Regret is defined as element‑wise product of costs and risks.
    The function returns a normalized regret vector (sums to 1) and its Gini.
    """
    if costs.shape != risks.shape:
        raise ValueError("costs and risks must share the same shape")
    regrets = costs * risks
    total = regrets.sum()
    if total == 0:
        normalized = np.full_like(regrets, 1.0 / regrets.size)
    else:
        normalized = regrets / total
    gini = gini_coefficient(normalized)
    return normalized, gini


# ----------------------------------------------------------------------
# Deeper hybrid integration
# ----------------------------------------------------------------------
def embed_regret_as_vector(regret: np.ndarray, n: int) -> Multivector:
    """
    Map a regret vector to a grade‑1 multivector (a vector) in an n‑dimensional space.
    If the regret length exceeds n, the excess dimensions are ignored.
    """
    if regret.ndim != 1:
        raise ValueError("regret must be a 1‑D array")
    coords = regret[:n]
    return Multivector.vector(coords, n)


def compute_hybrid_edge_weights(
    costs: np.ndarray,
    risks: np.ndarray,
    multivector_components: Dict[Blade, float],
    n: int,
) -> np.ndarray:
    """
    Hybrid edge weight computation:
      1. Obtain normalized regrets and their Gini.
      2. Embed regrets as a vector multivector.
      3. Geometric‑product interaction between the embedded regret and the
         supplied multivector; extract the scalar part as an interaction term.
      4. Modulate the normalized regrets by a factor that depends on the
         interaction term and the Gini coefficient.
    """
    normalized_regrets, gini = compute_regret_gini(costs, risks)

    # Step 2 – embed regrets
    regret_mv = embed_regret_as_vector(normalized_regrets, n)

    # Step 3 – geometric product with user‑provided multivector
    user_mv = Multivector(multivector_components, n)
    interaction_mv = regret_mv * user_mv
    interaction_scalar = interaction_mv.scalar_part()

    # Step 4 – modulation
    modulation = 1.0 + interaction_scalar * (1.0 - gini)
    weights = normalized_regrets * modulation

    # Guard against pathological zero‑sum cases
    total = weights.sum()
    if total == 0:
        return np.full_like(weights, 1.0 / weights.size)
    return weights / total


def hybrid_prune_and_rank(
    weights: np.ndarray,
    pruning_rate: float,
    alpha: float,
    t: float,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """
    Apply an exponential decay pruning schedule and renormalize.
    The schedule is bounded to [0, 1] to avoid negative scaling.
    """
    schedule = pruning_rate * math.exp(-alpha * t)
    schedule = min(max(schedule, 0.0), 1.0)
    pruned = weights * (1.0 - schedule)
    total = pruned.sum()
    if total < epsilon:
        # fallback to uniform distribution if everything is pruned away
        return np.full_like(pruned, 1.0 / pruned.size)
    return pruned / total


def fisher_score_localization(weights: np.ndarray) -> float:
    """
    Compute a Fisher‑information‑like scalar from the weight distribution.
    Here we use the quadratic form wᵀ·diag(w)·w = Σ w_i³, which emphasizes
    peakedness more strongly than Σ w_i². The angle is then arctan of that
    quantity, yielding a value in (0, π/2).
    """
    if weights.ndim != 1:
        raise ValueError("weights must be a 1‑D array")
    fisher_like = np.sum(weights ** 3)
    return math.atan(fisher_like)


# ----------------------------------------------------------------------
# Simple sanity test (executed when run as script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example data
    costs = np.array([1.0, 2.0, 3.0, 4.0])
    risks = np.array([0.5, 0.6, 0.7, 0.8])

    # Multivector: scalar 1.0 + vector (2, -1, 0, 0)
    mv_components = {
        (): 1.0,
        (0,): 2.0,
        (1,): -1.0,
    }
    n_dim = 4

    # Hybrid pipeline
    edge_weights = compute_hybrid_edge_weights(costs, risks, mv_components, n_dim)
    pruned = hybrid_prune_and_rank(edge_weights, pruning_rate=0.15, alpha=0.4, t=2.0)
    angle = fisher_score_localization(pruned)

    print(f"Edge weights (normalized): {edge_weights}")
    print(f"Pruned & renormalized:    {pruned}")
    print(f"Localization angle (rad): {angle:.6f}")