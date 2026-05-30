# DARWIN HAMMER — match 5022, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m84_s3.py (gen4)
# parent_b: hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s3.py (gen3)
# born: 2026-05-29T23:59:19Z

"""Hybrid module combining geometric algebra (Parent A) and NLMS adaptive filtering with
date‑based learning‑rate modulation (Parent B).

Mathematical bridge:
- In geometric algebra the scalar part of the geometric product of two grade‑1
  multivectors equals their Euclidean dot product.
- The NLMS algorithm relies on this dot product for prediction and on a scalar
  learning rate μ for weight updates.
- We obtain μ from the Doomsday rule (Parent B) and feed it into the NLMS
  update that operates on multivector‑encoded weight and input vectors.
Thus the two parent topologies are fused: multivector algebra provides the
linear algebraic substrate, while the calendar‑derived μ controls the adaptive
learning dynamics.
"""

import math
import random
import sys
from datetime import date
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Geometric Algebra core
# ----------------------------------------------------------------------


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting index list.
    Identical indices cancel (Grassmann algebra property)."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign *= -1
            i = max(i - 1, 0)
        elif lst[i] == lst[i + 1]:
            # cancel pair
            lst.pop(i)
            lst.pop(i)  # next element shifts into i
            n -= 2
            i = max(i - 1, 0)
        else:
            i += 1
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components: dict[frozenset, float], n: int):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    # ------------------------------------------------------------------
    # Basic grade‑filtering utilities
    # ------------------------------------------------------------------
    def grade(self, k: int):
        """Return a new Multivector keeping only grade‑k blades."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items()
             if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the scalar (grade‑0) coefficient."""
        return self.components.get(frozenset(), 0.0)

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------
    def __add__(self, other):
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if v != 0.0}, self.n)

    def __mul__(self, other):
        """Geometric product."""
        result = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                blade_c, sign = _multiply_blades(blade_a, blade_b)
                result[blade_c] = result.get(blade_c, 0.0) + sign * coef_a * coef_b
        return Multivector({k: v for k, v in result.items() if v != 0.0}, self.n)

    # ------------------------------------------------------------------
    # Conversion helpers for NLMS (grade‑1 ↔ vector)
    # ------------------------------------------------------------------
    def to_numpy_vector(self) -> np.ndarray:
        """Return a dense NumPy vector of the grade‑1 coefficients (size n)."""
        vec = np.zeros(self.n, dtype=float)
        for blade, coef in self.components.items():
            if len(blade) == 1:
                idx = next(iter(blade))
                vec[idx] = coef
        return vec

    @staticmethod
    def from_numpy_vector(vec: np.ndarray) -> "Multivector":
        """Create a grade‑1 multivector from a NumPy vector."""
        n = int(vec.shape[0])
        comps = {frozenset({i}): float(v) for i, v in enumerate(vec) if v != 0.0}
        return Multivector(comps, n)

    # ------------------------------------------------------------------
    def __repr__(self):
        return f"Multivector({self.components}, n={self.n})"


# ----------------------------------------------------------------------
# Parent B – Doomsday‑driven NLMS core
# ----------------------------------------------------------------------


def doomsday_rule(year: int, month: int, day: int) -> int:
    """Return day of week (0=Sunday … 6=Saturday) using the Doomsday algorithm."""
    return (date(year, month, day).weekday() + 1) % 7


def bayesian_information_criterion(log_likelihood: float,
                                   n_params: int,
                                   n_samples: int) -> float:
    """Standard BIC:  -2*logL + k*log(N)."""
    return -2.0 * log_likelihood + n_params * math.log(n_samples)


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Standard NLMS prediction (dot product)."""
    return float(weights @ x)


def nlms_update(weights: np.ndarray,
                x: np.ndarray,
                target: float,
                mu: float = 0.5,
                eps: float = 1e-9) -> np.ndarray:
    """Standard NLMS weight update."""
    error = target - nlms_predict(weights, x)
    norm_sq = max(eps, float(x @ x))
    step = (mu / norm_sq) * error
    return weights + step * x


# ----------------------------------------------------------------------
# Hybrid operations (the fusion)
# ----------------------------------------------------------------------


def geometric_dot(v1: Multivector, v2: Multivector) -> float:
    """Scalar part of the geometric product of two grade‑1 multivectors.
    This equals the Euclidean dot product of their coefficient vectors."""
    if len(v1.components) == 0 or len(v2.components) == 0:
        return 0.0
    prod = v1 * v2
    return prod.scalar_part()


def nlms_predict_mv(weights_mv: Multivector,
                    x_mv: Multivector) -> float:
    """NLMS prediction where weights and input are encoded as grade‑1 multivectors."""
    return geometric_dot(weights_mv, x_mv)


def nlms_update_mv(weights_mv: Multivector,
                   x_mv: Multivector,
                   target: float,
                   today: date = date.today(),
                   base_mu: float = 0.5,
                   eps: float = 1e-9) -> Multivector:
    """
    NLMS weight update with a learning rate μ modulated by the Doomsday rule.

    Parameters
    ----------
    weights_mv : Multivector
        Current weight vector (grade‑1).
    x_mv : Multivector
        Current input vector (grade‑1).
    target : float
        Desired scalar output.
    today : date
        Calendar date used to compute μ via the Doomsday rule.
    base_mu : float
        Baseline learning‑rate constant.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    Multivector
        Updated weight vector (grade‑1).
    """
    # Convert to dense vectors for the NLMS arithmetic
    w = weights_mv.to_numpy_vector()
    x = x_mv.to_numpy_vector()

    # Doomsday‑derived scaling factor (0 … 1) and final μ
    dow = doomsday_rule(today.year, today.month, today.day)  # 0‑6
    mu = base_mu * (1.0 + dow / 6.0)  # stretch μ up to 2× base_mu

    # Standard NLMS update in Euclidean space
    w_new = nlms_update(w, x, target, mu=mu, eps=eps)

    # Return as a grade‑1 multivector
    return Multivector.from_numpy_vector(w_new)


def bic_for_mv_model(log_likelihood: float,
                     weights_mv: Multivector,
                     n_samples: int) -> float:
    """
    Compute BIC for a model whose parameters are stored in a multivector.

    The number of free parameters is the number of non‑zero grade‑1 coefficients.
    """
    n_params = sum(1 for blade in weights_mv.components if len(blade) == 1)
    return bayesian_information_criterion(log_likelihood, n_params, n_samples)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dimensionality
    dim = 5

    # Random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Initialise weight and input as random grade‑1 multivectors
    w_vec = np.random.randn(dim)
    x_vec = np.random.randn(dim)

    weights_mv = Multivector.from_numpy_vector(w_vec)
    x_mv = Multivector.from_numpy_vector(x_vec)

    # Desired target (scalar)
    target = 0.7

    # Perform prediction
    pred_before = nlms_predict_mv(weights_mv, x_mv)
    print(f"Prediction before update: {pred_before:.6f}")

    # Perform hybrid NLMS update
    updated_weights_mv = nlms_update_mv(weights_mv, x_mv, target)

    # Prediction after update
    pred_after = nlms_predict_mv(updated_weights_mv, x_mv)
    print(f"Prediction after update:  {pred_after:.6f}")

    # Compute a dummy log‑likelihood and BIC
    log_like = -0.5 * (target - pred_after) ** 2  # simple Gaussian log‑likelihood
    bic = bic_for_mv_model(log_like, updated_weights_mv, n_samples=100)
    print(f"BIC of the updated model: {bic:.4f}")

    # Verify that the update changed at least one coefficient
    diff = np.linalg.norm(updated_weights_mv.to_numpy_vector() - w_vec)
    assert diff > 1e-8, "Weights did not change after update"
    print("Smoke test passed.")