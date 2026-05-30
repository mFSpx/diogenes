# DARWIN HAMMER — match 5022, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m84_s3.py (gen4)
# parent_b: hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s3.py (gen3)
# born: 2026-05-29T23:59:19Z

"""Hybrid algorithm merging geometric algebra (Parent A) and NLMS adaptive filtering with
date‑based learning‑rate modulation (Parent B).

Mathematical bridge:
- A Multivector is a linear combination of basis blades. Its coefficient vector can be
  treated as the weight vector of an NLMS adaptive filter.
- The NLMS prediction is the inner product between the weight‑vector and the input‑vector
  (coefficients of an input multivector). The error drives an update of the coefficients.
- The learning‑rate μ is modulated by the Doomsday rule (weekday of the current date),
  providing a deterministic, date‑dependent scaling that links the two parent algorithms.

The module therefore:
1. Provides a full `Multivector` implementation (geometric algebra).
2. Supplies NLMS prediction / update functions that operate on multivector coefficients.
3. Uses the Doomsday rule to adapt μ each step, completing the hybrid fusion.
"""

import math
import random
import sys
from datetime import date
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple, List

# ----------------------------------------------------------------------
# Parent A – Geometric Algebra core
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list.

    If duplicate indices appear the blade vanishes (square = 0) and the function
    returns the reduced list with the current sign.
    """
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
                # cancel duplicate basis vectors
                lst.pop(j)
                lst.pop(j)  # after first pop the second duplicate shifts to j
                n -= 2
                sign *= 0  # blade disappears; sign becomes irrelevant
                return lst, sign
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-12}
        self.n = int(n)

    # ------------------------------------------------------------------
    # Basic algebraic operations
    # ------------------------------------------------------------------
    def grade(self, k: int) -> "Multivector":
        """Return a new Multivector keeping only grade‑k blades."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the scalar (grade‑0) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if abs(v) > 1e-12}, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                blade_c, sign = _multiply_blades(blade_a, blade_b)
                if sign == 0:  # blade vanished due to duplicate indices
                    continue
                result[blade_c] = result.get(blade_c, 0.0) + sign * coef_a * coef_b
        return Multivector({k: v for k, v in result.items() if abs(v) > 1e-12}, self.n)

    # ------------------------------------------------------------------
    # Utility helpers for the hybrid NLMS part
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        terms = [f"{v:.3g}*e{sorted(list(k)) if k else '0'}" for k, v in self.components.items()]
        return " + ".join(terms) if terms else "0"

    def copy(self) -> "Multivector":
        return Multivector(dict(self.components), self.n)


def _all_blades(n: int) -> List[FrozenSet[int]]:
    """Return a list of all possible blades for an n‑dimensional space,
    ordered first by grade then lexicographically."""
    blades: List[FrozenSet[int]] = [frozenset()]  # scalar
    for grade in range(1, n + 1):
        # generate combinations of indices 0..n-1 of size grade
        from itertools import combinations

        for combo in combinations(range(n), grade):
            blades.append(frozenset(combo))
    return blades


def multivector_to_numpy(mv: Multivector, ordering: List[FrozenSet[int]]) -> np.ndarray:
    """Map a Multivector to a dense numpy vector according to `ordering`."""
    vec = np.zeros(len(ordering), dtype=float)
    for i, blade in enumerate(ordering):
        vec[i] = mv.components.get(blade, 0.0)
    return vec


def numpy_to_multivector(vec: np.ndarray, ordering: List[FrozenSet[int]], n: int) -> Multivector:
    """Inverse of `multivector_to_numpy`."""
    comps = {blade: float(coef) for blade, coef in zip(ordering, vec) if abs(coef) > 1e-12}
    return Multivector(comps, n)


# ----------------------------------------------------------------------
# Parent B – Doomsday‑guided NLMS core
# ----------------------------------------------------------------------
def doomsday_rule(year: int, month: int, day: int) -> int:
    """Return weekday (0=Sunday … 6=Saturday) using the Doomsday algorithm."""
    return (date(year, month, day).weekday() + 1) % 7


def bayesian_information_criterion(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """Standard BIC = -2*logL + k*log(N)."""
    return -2.0 * log_likelihood + n_params * math.log(n_samples)


def nlms_predict(weights_vec: np.ndarray, x_vec: np.ndarray) -> float:
    """Standard NLMS prediction: inner product w·x."""
    return float(weights_vec @ x_vec)


def nlms_update(
    weights_vec: np.ndarray,
    x_vec: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> np.ndarray:
    """Return updated weight vector using the NLMS rule."""
    pred = nlms_predict(weights_vec, x_vec)
    error = target - pred
    norm_sq = np.dot(x_vec, x_vec) + eps
    step = (mu / norm_sq) * error
    return weights_vec + step * x_vec


# ----------------------------------------------------------------------
# Hybrid operations (functions that combine both parents)
# ----------------------------------------------------------------------
def nlms_predict_mv(weights: Multivector, x: Multivector, ordering: List[FrozenSet[int]]) -> float:
    """Predict using NLMS on the coefficient vectors of two multivectors."""
    w_vec = multivector_to_numpy(weights, ordering)
    x_vec = multivector_to_numpy(x, ordering)
    return nlms_predict(w_vec, x_vec)


def nlms_update_mv(
    weights: Multivector,
    x: Multivector,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
    ordering: List[FrozenSet[int]] = None,
) -> Multivector:
    """Update a Multivector weight using NLMS, returning a new Multivector."""
    if ordering is None:
        ordering = _all_blades(weights.n)
    w_vec = multivector_to_numpy(weights, ordering)
    x_vec = multivector_to_numpy(x, ordering)
    w_new_vec = nlms_update(w_vec, x_vec, target, mu, eps)
    return numpy_to_multivector(w_new_vec, ordering, weights.n)


def hybrid_step(
    today: date,
    weights: Multivector,
    x: Multivector,
    target: float,
    base_mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[Multivector, float]:
    """Perform one hybrid adaptation step.

    The learning rate μ is scaled by a factor derived from the Doomsday rule:
        μ_eff = base_mu * (1 + weekday/6)

    Returns the updated weight Multivector and the prediction error.
    """
    weekday = doomsday_rule(today.year, today.month, today.day)  # 0‑6
    mu_eff = base_mu * (1.0 + weekday / 6.0)

    ordering = _all_blades(weights.n)

    pred = nlms_predict_mv(weights, x, ordering)
    error = target - pred
    new_weights = nlms_update_mv(weights, x, target, mu=mu_eff, eps=eps, ordering=ordering)
    return new_weights, error


def evaluate_bic_on_sequence(
    predictions: List[float],
    targets: List[float],
    n_params: int,
) -> float:
    """Compute BIC for a sequence of predictions vs targets."""
    if len(predictions) != len(targets):
        raise ValueError("Length mismatch between predictions and targets.")
    residuals = np.array(targets) - np.array(predictions)
    # Assuming Gaussian noise, log‑likelihood = -0.5 * N * (log(2πσ²) + 1)
    # σ² estimated by MSE
    mse = np.mean(residuals ** 2) + 1e-12
    log_likelihood = -0.5 * len(targets) * (math.log(2 * math.pi * mse) + 1)
    return bayesian_information_criterion(log_likelihood, n_params, len(targets))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a 3‑dimensional geometric algebra (8 blades total)
    n_dim = 3
    ordering = _all_blades(n_dim)

    # Random initial weight multivector
    rng = np.random.default_rng(seed=42)
    init_coeffs = rng.normal(size=len(ordering))
    weights_mv = numpy_to_multivector(init_coeffs, ordering, n_dim)

    # Random input multivector
    x_coeffs = rng.normal(size=len(ordering))
    x_mv = numpy_to_multivector(x_coeffs, ordering, n_dim)

    # Synthetic target (e.g., scalar part of geometric product)
    target = (weights_mv * x_mv).scalar_part()

    # Perform a hybrid adaptation step
    today = date.today()
    new_weights, err = hybrid_step(today, weights_mv, x_mv, target)

    print("Initial weights (first 5 coeffs):", init_coeffs[:5])
    print("Input vector (first 5 coeffs):", x_coeffs[:5])
    print("Target scalar:", target)
    print("Prediction error after step:", err)

    # Run a short sequence to demonstrate BIC computation
    preds = []
    trgs = []
    w = weights_mv
    for _ in range(10):
        pred = nlms_predict_mv(w, x_mv, ordering)
        preds.append(pred)
        trgs.append(target)
        w = nlms_update_mv(w, x_mv, target, mu=0.5, ordering=ordering)

    bic_score = evaluate_bic_on_sequence(preds, trgs, n_params=len(ordering))
    print("BIC score for 10‑step sequence:", bic_score)