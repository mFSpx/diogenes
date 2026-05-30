# DARWIN HAMMER — match 3438, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_distri_m1385_s3.py (gen4)
# born: 2026-05-29T23:50:15Z

"""Hybrid Algorithm combining Clifford geometric product, Bayesian updating, and
Test‑Time Training (TTT).

Parent A contributes:
- Clifford algebra utilities (`_blade_sign`, `_multiply_blades`, `geometric_product`).
- Linear adaptation utilities (`init_ttt`, `ttt_loss`, `ttt_grad`).
- Image similarity metric (`ssim`).

Parent B contributes:
- Bayesian update mechanism for probability‑like coefficients.
- Use of SSIM as a data‑driven likelihood.

**Mathematical bridge**
The coefficients of a multivector (a dict mapping basis blades to scalars) are
interpreted as a discrete probability distribution (they sum to 1 after
normalisation).  The geometric product of two multivectors yields a new set of
coefficients; these are taken as the *likelihood* of observing the current data.
Multiplying the prior distribution by this likelihood and renormalising gives a
Bayesian posterior.  The posterior then drives a single step of Test‑Time
Training: the gradient of the TTT loss is scaled by the posterior confidence
and used to update the linear map `W`.  This creates a tightly coupled hybrid
system where algebraic structure, probabilistic reasoning and online gradient
descent interact in a single mathematical loop.
"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import Any, Dict, Tuple, FrozenSet

# ----------------------------------------------------------------------
# Clifford algebra helpers (Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # Cancel duplicate index (e_i * e_i = 1)
                lst.pop(j)
                lst.pop(j)  # second element shifts to j after first pop
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def geometric_product(a: Dict[FrozenSet[int], float],
                     b: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """
    Full Clifford (geometric) product of two multivectors `a` and `b`.
    Each multivector is a dict mapping a basis blade (frozenset of ints) to a scalar.
    """
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coef_a in a.items():
        for blade_b, coef_b in b.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            result[blade_res] = result.get(blade_res, 0.0) + sign * coef_a * coef_b
    # Remove near‑zero entries for cleanliness
    return {blade: c for blade, c in result.items() if abs(c) > 1e-12}

# ----------------------------------------------------------------------
# Linear adaptation utilities (Parent A)
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize weight matrix `W` of shape (d_out, d_in)."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> float:
    """Self‑supervised loss for Test‑Time Training."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    """Gradient of `ttt_loss` w.r.t. `W`."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)

# ----------------------------------------------------------------------
# SSIM similarity (Parent A) – used as likelihood in Bayesian step
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index Measure for 1‑D arrays."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) *
                                                             (sigma_x ** 2 + sigma_y ** 2 + c2))

# ----------------------------------------------------------------------
# Bayesian utilities (Parent B)
# ----------------------------------------------------------------------
def bayesian_update(prior: Dict[FrozenSet[int], float],
                    likelihood: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """
    Perform element‑wise Bayesian update: posterior ∝ prior * likelihood.
    The result is renormalised to sum to 1 (interpreted as a probability mass).
    """
    unnorm: Dict[FrozenSet[int], float] = {}
    for blade, p in prior.items():
        l = likelihood.get(blade, 0.0)
        unnorm[blade] = p * l
    total = sum(unnorm.values())
    if total == 0.0:
        # Avoid division by zero – fallback to uniform prior over observed blades
        n = len(unnorm)
        return {blade: 1.0 / n for blade in unnorm}
    return {blade: v / total for blade, v in unnorm.items()}

def vector_to_multivector(v: np.ndarray) -> Dict[FrozenSet[int], float]:
    """
    Encode a real vector as a multivector where each component i becomes the
    grade‑1 blade {i} with coefficient v[i].
    """
    return {frozenset({int(i)}): float(coeff) for i, coeff in enumerate(v)}

def multivector_to_vector(mv: Dict[FrozenSet[int], float], dim: int) -> np.ndarray:
    """
    Decode a multivector back to a dense vector of length `dim`. Only grade‑1
    blades contribute; other grades are ignored for this simple projection.
    """
    vec = np.zeros(dim, dtype=float)
    for blade, coeff in mv.items():
        if len(blade) == 1:
            idx = next(iter(blade))
            if idx < dim:
                vec[idx] = coeff
    return vec

# ----------------------------------------------------------------------
# Hybrid operation (core of the fused algorithm)
# ----------------------------------------------------------------------
def hybrid_step(W: np.ndarray,
                x: np.ndarray,
                prior_mv: Dict[FrozenSet[int], float],
                target: np.ndarray = None,
                lr: float = 0.1) -> Tuple[np.ndarray, Dict[FrozenSet[int], float]]:
    """
    Perform one hybrid iteration:
    1. Linear prediction `p = W @ x`.
    2. Encode `p` as a multivector and compute geometric product with the prior.
       The product acts as a *likelihood*.
    3. Update the prior via Bayesian rule, using SSIM(p, x) as an additional
       scalar likelihood factor (applied uniformly to all blades).
    4. Compute TTT gradient and update `W` with a step size scaled by the
       posterior confidence (the max posterior coefficient).
    Returns the updated weight matrix and the new posterior multivector.
    """
    # 1. Linear prediction
    pred = W @ x

    # 2. Likelihood from geometric product
    pred_mv = vector_to_multivector(pred)
    likelihood_mv = geometric_product(prior_mv, pred_mv)

    # 3. SSIM‑based scalar likelihood
    sim = ssim(pred, x if target is None else target)
    # Scale every blade of the likelihood by the similarity (clamp to [0,1])
    sim = max(0.0, min(1.0, sim))
    scaled_likelihood = {blade: coeff * sim for blade, coeff in likelihood_mv.items()}

    posterior_mv = bayesian_update(prior_mv, scaled_likelihood)

    # 4. Confidence‑scaled TTT update
    confidence = max(posterior_mv.values()) if posterior_mv else 0.0
    grad = ttt_grad(W, x, target)
    W_new = W - lr * confidence * grad

    return W_new, posterior_mv

def hybrid_predict(W: np.ndarray,
                   x: np.ndarray,
                   prior_mv: Dict[FrozenSet[int], float]) -> Tuple[np.ndarray, Dict[FrozenSet[int], float]]:
    """
    Produce a prediction together with an updated posterior without performing a
    gradient step.  Useful for inference where learning is disabled.
    """
    pred = W @ x
    pred_mv = vector_to_multivector(pred)
    likelihood_mv = geometric_product(prior_mv, pred_mv)
    sim = ssim(pred, x)
    sim = max(0.0, min(1.0, sim))
    scaled_likelihood = {blade: coeff * sim for blade, coeff in likelihood_mv.items()}
    posterior_mv = bayesian_update(prior_mv, scaled_likelihood)
    return pred, posterior_mv

def initialize_hybrid(dimension: int,
                      scale: float = 0.01,
                      seed: int = 0) -> Tuple[np.ndarray, Dict[FrozenSet[int], float]]:
    """
    Initialise the linear map `W` and a uniform prior multivector over the
    `dimension`‑dimensional space.
    """
    W = init_ttt(dimension, scale=scale, seed=seed)
    uniform_coeff = 1.0 / dimension
    prior_mv = {frozenset({i}): uniform_coeff for i in range(dimension)}
    return W, prior_mv

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dim = 5
    W, prior = initialize_hybrid(dim, scale=0.05, seed=42)

    # Random input vector
    rng = np.random.default_rng(1)
    x = rng.standard_normal(dim)

    # Run a few hybrid steps
    for step in range(3):
        W, prior = hybrid_step(W, x, prior, lr=0.05)
        pred, posterior = hybrid_predict(W, x, prior)
        loss_val = ttt_loss(W, x)
        print(f"Step {step+1}: loss={loss_val:.4f}, pred={pred.round(3)}, max posterior={max(posterior.values()):.3f}")