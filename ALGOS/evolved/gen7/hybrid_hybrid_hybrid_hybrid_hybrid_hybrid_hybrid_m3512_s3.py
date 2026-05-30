# DARWIN HAMMER — match 3512, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_ollivier_ricci_curva_m1848_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1844_s3.py (gen5)
# born: 2026-05-29T23:50:39Z

import math
import random
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Blade arithmetic
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return a sorted blade list and the sign produced by anti‑commutation.
    Duplicate indices cancel (e_i * e_i = 1)."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel the pair
                del lst[j : j + 2]
                n -= 2
                sign *= 1
                j -= 1  # stay on current position after removal
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    """Geometric product of two basis blades."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


# ----------------------------------------------------------------------
# Multivector class
# ----------------------------------------------------------------------
class Multivector:
    """Sparse representation of a multivector as a dict {blade: coefficient}."""

    def __init__(self, components: Dict[frozenset, float] | None = None):
        self.components: Dict[frozenset, float] = {}
        if components:
            # discard zero coefficients
            for b, c in components.items():
                if abs(c) > 1e-12:
                    self.components[frozenset(b)] = float(c)

    @staticmethod
    def basis_vector(idx: int) -> "Multivector":
        return Multivector({frozenset({idx}): 1.0})

    @staticmethod
    def scalar(value: float) -> "Multivector":
        return Multivector({frozenset(): float(value)})

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for b, c in other.components.items():
            result[b] = result.get(b, 0.0) + c
            if abs(result[b]) < 1e-12:
                del result[b]
        return Multivector(result)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self.components.items()})

    def geometric_product(self, other: "Multivector") -> "Multivector":
        """Geometric (Clifford) product."""
        result: Dict[frozenset, float] = {}
        for b1, c1 in self.components.items():
            for b2, c2 in other.components.items():
                b_res, sign = _multiply_blades(b1, b2)
                coeff = c1 * c2 * sign
                result[b_res] = result.get(b_res, 0.0) + coeff
        # prune near‑zero entries
        result = {b: v for b, v in result.items() if abs(v) > 1e-12}
        return Multivector(result)

    def inner_product(self, other: "Multivector") -> float:
        """Scalar inner product <A, B> = sum_{blade} a_blade * b_blade."""
        total = 0.0
        for b, c in self.components.items():
            if b in other.components:
                total += c * other.components[b]
        return total

    def norm(self) -> float:
        """Euclidean norm sqrt(<A, A>)."""
        return math.sqrt(self.inner_product(self))

    def __repr__(self) -> str:
        if not self.components:
            return "0"
        terms = []
        for blade, coeff in sorted(self.components.items(), key=lambda x: (len(x[0]), tuple(sorted(x[0])))):
            if not blade:
                term = f"{coeff:.3g}"
            else:
                basis = "e" + "".join(str(i) for i in sorted(blade))
                term = f"{coeff:.3g}{basis}"
            terms.append(term)
        return " + ".join(terms)


# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature
# ----------------------------------------------------------------------
def _distance_matrix(points: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix."""
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=2)


def ollivier_ricci_curvature(points: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Ollivier‑Ricci curvature.

    For each node i we compute curvature κ_i = 1 - (average distance to neighbours) / (global avg distance).
    The neighbour set is taken as the k‑nearest neighbours.

    Args:
    points: A 2D numpy array of shape (n_samples, n_dim)
    k: The number of nearest neighbours to consider

    Returns:
    A 1D numpy array of shape (n_samples,) containing the Ollivier‑Ricci curvature for each point
    """
    if points.ndim != 2:
        raise ValueError("points must be a 2‑D array (n_samples, n_dim)")
    n = points.shape[0]
    dist = _distance_matrix(points)
    np.fill_diagonal(dist, np.inf)
    knn_dist = np.partition(dist, k, axis=1)[:, :k]
    knn_avg = np.mean(knn_dist, axis=1)
    global_avg = np.mean(dist[dist != np.inf])
    curvature = 1.0 - knn_avg / (global_avg + 1e-12)
    return curvature  # shape (n,)


# ----------------------------------------------------------------------
# Regret‑based decision utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def compute_regret(actions: List[MathAction], counterfactuals: List[MathCounterfactual]) -> Dict[str, float]:
    """
    Compute raw regret for each action:
        regret(a) = Σ_{cf} p(cf) * max(0, cf.outcome - expected[a])
    """
    if not actions:
        return {}
    exp_map = {a.id: a.expected_value for a in actions}
    regrets: Dict[str, float] = {a.id: 0.0 for a in actions}
    for cf in counterfactuals:
        if cf.action_id not in exp_map:
            continue
        diff = cf.outcome_value - exp_map[cf.action_id]
        if diff > 0:
            regrets[cf.action_id] += cf.probability * diff  # Fix: probability instead of probabil
    return regrets


def hybrid_decision_pipeline(multivectors: List[Multivector], 
                             points: np.ndarray, 
                             actions: List[MathAction], 
                             counterfactuals: List[MathCounterfactual]) -> Dict[str, float]:
    health_scalars = [mv.norm() for mv in multivectors]
    curvature = ollivier_ricci_curvature(points)
    modulated_regrets = {}
    for i, (health, action) in enumerate(zip(health_scalars, actions)):
        regret = compute_regret([action], counterfactuals)[action.id]
        modulated_regret = regret * health * curvature[i]
        modulated_regrets[action.id] = modulated_regret
    return modulated_regrets