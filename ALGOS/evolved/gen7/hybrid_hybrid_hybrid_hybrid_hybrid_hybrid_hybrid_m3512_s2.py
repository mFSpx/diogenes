# DARWIN HAMMER — match 3512, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_ollivier_ricci_curva_m1848_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1844_s3.py (gen5)
# born: 2026-05-29T23:50:39Z

"""Hybrid algorithm combining geometric algebra (multivectors) with regret‑weighted decision making.
Parent A: hybrid_hybrid_hybrid_hybrid_ollivier_ricci_curva_m1848_s3.py – provides blade arithmetic,
geometric product and a placeholder for Ollivier‑Ricci curvature.
Parent B: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1844_s3.py – provides regret‑based
strategy, signature similarity and related utilities.

Mathematical bridge:
* The norm (Euclidean magnitude) of a multivector is interpreted as a “health” scalar.
* Ollivier‑Ricci curvature of a (synthetic) graph is used as an annealing temperature.
* The health scalar modulates the regret weights, while the curvature temperature scales
  the exploration‑exploitation trade‑off.  Thus the decision‑making pipeline becomes a
  function of both geometric‑algebraic structure and curvature‑driven dynamics.
"""

import math
import random
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Blade arithmetic (from Parent A)
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
# Multivector class (core of Parent A)
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
# Ollivier‑Ricci curvature placeholder (from Parent A)
# ----------------------------------------------------------------------
def _distance_matrix(points: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix."""
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=2)


def ollivier_ricci_curvature(points: np.ndarray) -> np.ndarray:
    """
    Very crude surrogate for Ollivier‑Ricci curvature.
    For each node i we compute curvature κ_i = 1 - (average distance to neighbours) / (global avg distance).
    The neighbour set is taken as the k‑nearest neighbours (k=3).
    """
    if points.ndim != 2:
        raise ValueError("points must be a 2‑D array (n_samples, n_dim)")
    n = points.shape[0]
    dist = _distance_matrix(points)
    np.fill_diagonal(dist, np.inf)
    k = min(3, n - 1)
    knn_avg = np.mean(np.partition(dist, k, axis=1)[:, :k], axis=1)
    global_avg = np.mean(dist[dist != np.inf])
    curvature = 1.0 - knn_avg / (global_avg + 1e-12)
    return curvature  # shape (n,)


# ----------------------------------------------------------------------
# Regret‑based decision utilities (from Parent B)
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
            regrets[cf.action_id] += cf.probability * diff
    return regrets


def regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    temperature: float,
) -> Dict[str, float]:
    """
    Convert regrets into a probability distribution using a soft‑max with temperature.
    Lower temperature → more deterministic (exploitation).
    """
    regrets = compute_regret(actions, counterfactuals)
    if not regrets:
        return {}
    # Shift regrets to be non‑negative
    min_reg = min(regrets.values())
    shifted = {k: v - min_reg for k, v in regrets.items()}
    # Soft‑max
    exp_vals = {k: math.exp(-v / (temperature + 1e-12)) for k, v in shifted.items()}
    total = sum(exp_vals.values())
    return {k: v / total for k, v in exp_vals.items()}


# ----------------------------------------------------------------------
# Hybrid functions (the required three+ functions)
# ----------------------------------------------------------------------
def health_from_multivector(mv: Multivector) -> float:
    """Interpret the multivector norm as a health scalar in [0, ∞)."""
    return mv.norm()


def annealing_temperature(curvature: np.ndarray) -> float:
    """
    Derive a global temperature from curvature values.
    We use the mean curvature magnitude; larger positive curvature → higher temperature.
    """
    if curvature.size == 0:
        return 1.0
    mean_kappa = np.mean(np.abs(curvature))
    # Map mean curvature to a temperature in (0.1, 10) via a simple affine transform
    return 0.1 + 9.9 * (mean_kappa / (1.0 + mean_kappa))


def hybrid_regret_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    mv: Multivector,
    points: np.ndarray,
) -> Dict[str, float]:
    """
    Full hybrid pipeline:
      1. Compute health = ||mv||.
      2. Compute curvature from geometric positions.
      3. Derive temperature = annealing_temperature(curvature) * (1 / (1 + health)).
      4. Run regret_weighted_strategy with the derived temperature.
    """
    health = health_from_multivector(mv)
    curvature = ollivier_ricci_curvature(points)
    base_temp = annealing_temperature(curvature)
    temperature = base_temp / (1.0 + health)
    return regret_weighted_strategy(actions, counterfactuals, temperature)


def hybrid_signature_similarity(actions: List[MathAction], points: np.ndarray) -> float:
    """
    Example of a cross‑domain metric:
      * Build a token set from action IDs.
      * Build a token set from quantized point coordinates.
      * Compute MinHash‑style signatures and return their Jaccard‑like similarity.
    """
    action_tokens = {a.id for a in actions}
    # Quantize points to strings like "x_y_z"
    point_tokens = {
        f"{int(round(p[0]))}_{int(round(p[1]))}_{int(round(p[2]))}"
        for p in points
    }
    sig_a = signature(action_tokens, k=64)
    sig_b = signature(point_tokens, k=64)
    return similarity(sig_a, sig_b)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a simple multivector: 3 + 2e1 + e2e3
    mv = Multivector({
        frozenset(): 3.0,
        frozenset({1}): 2.0,
        frozenset({2, 3}): 1.0,
    })
    print("Multivector:", mv)
    print("Norm (health):", health_from_multivector(mv))

    # Synthetic 5‑point cloud in 3‑D
    np.random.seed(0)
    points = np.random.randn(5, 3)

    # Dummy actions and counterfactuals
    actions = [
        MathAction(id="a1", expected_value=10.0),
        MathAction(id="a2", expected_value=8.5),
        MathAction(id="a3", expected_value=9.2),
    ]
    counterfactuals = [
        MathCounterfactual(action_id="a1", outcome_value=12.0, probability=0.4),
        MathCounterfactual(action_id="a2", outcome_value=7.0, probability=0.6),
        MathCounterfactual(action_id="a3", outcome_value=11.0, probability=0.5),
    ]

    strategy = hybrid_regret_strategy(actions, counterfactuals, mv, points)
    print("Hybrid regret strategy distribution:", strategy)

    sim = hybrid_signature_similarity(actions, points)
    print("Hybrid signature similarity:", sim)