# DARWIN HAMMER — match 3616, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m1514_s0.py (gen6)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s2.py (gen3)
# born: 2026-05-29T23:50:56Z

"""
Hybrid Fusion Module
====================

This module fuses the core topologies of the two parent algorithms:

* **Parent A** – geometric‑algebra multivector representation evolved by a
  Koopman operator and updated with an NLMS‑style rule.
* **Parent B** – Fisher‑score based weighting, SSIM similarity routing and a
  decreasing‑rate pruning schedule.

**Mathematical bridge**

1. The Koopman operator provides a linear (matrix) evolution of the
   multivector coefficient vector  **c** → **K·c**.  The resulting
   coefficient magnitudes are interpreted as a probability distribution
   **p** over basis blades.

2. The Fisher score `F(θ)` (derivative‑over‑intensity of a Gaussian beam)
   is used as a *likelihood* for a Bayesian update of **p**:
   `p̂ = normalize(p * F)`.

3. The updated probabilities weight the hygiene score used in the
   pruning schedule and also modulate the SSIM‑based routing decision.

Thus the algorithm alternates between a linear Koopman evolution,
a Bayesian Fisher‑score update, and a similarity‑driven routing/pruning
step, yielding a single unified hybrid system.

The implementation below provides three representative functions:
`koopman_step`, `nlms_update`, and `hybrid_route_and_prune`.
"""

import sys
import math
import random
import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Geometric Algebra utilities (Parent A)
# ----------------------------------------------------------------------


def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
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
                # duplicate index → cancels (e_i^2 = 1)
                del lst[j : j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(
    blade_a: FrozenSet[int], blade_b: FrozenSet[int]
) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Clifford algebra element represented as a sparse dict of blades."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # prune near‑zero entries
        self.components = {
            blade: val for blade, val in components.items() if abs(val) > 1e-15
        }

    def coefficient_vector(self) -> np.ndarray:
        """Return dense coefficient vector of length 2**n ordered by binary index."""
        size = 1 << self.n
        vec = np.zeros(size, dtype=float)
        for blade, val in self.components.items():
            # binary index: bit i set ⇔ basis vector e_i present
            idx = sum(1 << i for i in blade)
            vec[idx] = val
        return vec

    @classmethod
    def from_vector(cls, vec: np.ndarray, n: int) -> "Multivector":
        comps = {}
        for idx, val in enumerate(vec):
            if abs(val) > 1e-15:
                # decode blade from binary index
                blade = frozenset(
                    i for i in range(n) if (idx >> i) & 1
                )
                comps[blade] = float(val)
        return cls(comps, n)

    def norm(self) -> float:
        return math.sqrt(sum(v * v for v in self.components.values()))

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        return f"Multivector(n={self.n}, components={self.components})"


# ----------------------------------------------------------------------
# Koopman operator (linear evolution) – bridge to Parent B
# ----------------------------------------------------------------------


def koopman_step(mv: Multivector, K: np.ndarray) -> Multivector:
    """
    Apply a Koopman operator matrix K to the multivector coefficient vector.
    The resulting vector is re‑interpreted as a new Multivector.
    """
    if K.shape[0] != K.shape[1]:
        raise ValueError("Koopman matrix must be square")
    if K.shape[0] != (1 << mv.n):
        raise ValueError("Koopman dimension must match 2**n of the multivector")
    new_vec = K @ mv.coefficient_vector()
    return Multivector.from_vector(new_vec, mv.n)


# ----------------------------------------------------------------------
# NLMS update rule (Parent A)
# ----------------------------------------------------------------------


def nlms_update(
    mv: Multivector,
    x: np.ndarray,
    d: float,
    mu: float = 0.01,
    epsilon: float = 1e-6,
) -> Multivector:
    """
    Normalised Least‑Mean‑Squares (NLMS) adaptation of the multivector.
    Treat the coefficient vector as a linear filter w, input x, desired d.
    """
    w = mv.coefficient_vector()
    y = np.dot(w, x)
    e = d - y
    norm_x = np.dot(x, x) + epsilon
    w_new = w + (mu / norm_x) * e * x
    return Multivector.from_vector(w_new, mv.n)


# ----------------------------------------------------------------------
# Fisher score & SSIM (Parent B)
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


# ----------------------------------------------------------------------
# Hybrid decision & pruning (integration of both parents)
# ----------------------------------------------------------------------


def bayesian_fisher_update(prob_vec: np.ndarray, theta: float,
                           center: float, width: float) -> np.ndarray:
    """
    Treat the Fisher score as a likelihood for each basis blade and
    perform a Bayesian update of the probability vector.
    """
    likelihood = np.vectorize(lambda _: fisher_score(theta, center, width))(prob_vec)
    posterior_unnorm = prob_vec * likelihood
    if posterior_unnorm.sum() == 0:
        # avoid division by zero – fallback to uniform
        return np.full_like(posterior_unnorm, 1.0 / posterior_unnorm.size)
    return posterior_unnorm / posterior_unnorm.sum()


def decreasing_prune(weights: np.ndarray, step: int, decay_rate: float = 0.9) -> np.ndarray:
    """
    Apply a decreasing‑rate pruning schedule to a weight vector.
    The schedule follows w_i ← w_i * decay_rate^(step / (i+1)).
    """
    factors = np.power(decay_rate, step / (np.arange(weights.size) + 1))
    return weights * factors


def hybrid_route_and_prune(
    packet: dict,
    mv: Multivector,
    K: np.ndarray,
    reference_vec: np.ndarray,
    step: int,
    fisher_center: float = 0.0,
    fisher_width: float = 1.0,
) -> dict:
    """
    Complete hybrid step:
    1. Koopman evolution of the multivector.
    2. Bayesian Fisher‑score update of the blade‑probability distribution.
    3. SSIM similarity between packet payload and a reference vector.
    4. Pruning of the evolved coefficients using a decreasing schedule.
    Returns a dictionary with the decision payload.
    """
    # 1. Linear evolution
    mv_next = koopman_step(mv, K)

    # 2. Convert to probability vector (non‑negative, normalized)
    coeffs = mv_next.coefficient_vector()
    prob_vec = np.abs(coeffs)
    prob_sum = prob_vec.sum()
    if prob_sum == 0:
        prob_vec = np.full_like(prob_vec, 1.0 / prob_vec.size)
    else:
        prob_vec = prob_vec / prob_sum

    # 3. Fisher‑score Bayesian update
    theta = float(packet.get("theta", 0.0))
    prob_vec = bayesian_fisher_update(prob_vec, theta, fisher_center, fisher_width)

    # 4. SSIM similarity (payload assumed numeric list)
    payload = np.asarray(packet.get("payload", []), dtype=float)
    if payload.shape != reference_vec.shape:
        # pad/truncate to match length
        min_len = min(payload.size, reference_vec.size)
        payload = payload[:min_len]
        ref = reference_vec[:min_len]
    else:
        ref = reference_vec
    similarity = ssim(payload, ref)

    # 5. Pruning of coefficients using the updated probabilities as weights
    pruned_coeffs = decreasing_prune(prob_vec * coeffs, step)

    # 6. Re‑assemble multivector
    mv_pruned = Multivector.from_vector(pruned_coeffs, mv.n)

    decision = {
        "similarity": similarity,
        "probabilities": prob_vec.tolist(),
        "pruned_norm": mv_pruned.norm(),
        "theta": theta,
        "step": step,
    }
    return decision


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Simple 3‑dimensional Clifford algebra (n=3 → 8 coefficients)
    n = 3
    init_components = {
        frozenset(): 0.5,                     # scalar
        frozenset({0}): 0.2,                  # e1
        frozenset({1}): -0.1,                 # e2
        frozenset({0, 1}): 0.05,              # e12
    }
    mv = Multivector(init_components, n)

    # Random Koopman operator (must be 8×8)
    rng = np.random.default_rng(42)
    K = rng.normal(size=(1 << n, 1 << n))
    # Force stability by scaling eigenvalues < 1 (optional)
    K = K / np.max(np.abs(np.linalg.eigvals(K))) * 0.9

    # Dummy packet
    packet = {
        "theta": 0.3,
        "payload": [random.random() for _ in range(8)],
    }

    # Reference vector for SSIM (fixed)
    reference_vec = np.linspace(0, 1, 8)

    # Run a few hybrid steps
    for step in range(1, 4):
        decision = hybrid_route_and_prune(
            packet=packet,
            mv=mv,
            K=K,
            reference_vec=reference_vec,
            step=step,
            fisher_center=0.0,
            fisher_width=0.5,
        )
        # Update multivector for next iteration (using NLMS as an example)
        x_input = rng.normal(size=1 << n)
        desired = decision["similarity"]  # treat similarity as desired response
        mv = nlms_update(mv, x_input, desired, mu=0.05)

        print(f"Step {step}: similarity={decision['similarity']:.4f}, "
              f"pruned_norm={decision['pruned_norm']:.4f}")