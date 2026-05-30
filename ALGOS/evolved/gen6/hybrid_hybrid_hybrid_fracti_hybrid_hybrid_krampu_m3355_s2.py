# DARWIN HAMMER — match 3355, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_fractional_hd_hybrid_hybrid_nlms_o_m1127_s1.py (gen5)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_krampus_brain_m74_s2.py (gen2)
# born: 2026-05-29T23:49:40Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class CausalEffect:
    """Immutable container for a causal effect estimate."""
    effect_id: str
    treatment: str
    outcome: str
    confounders: Tuple[str, ...]
    ate_estimate: float | None
    ate_confidence_interval: Tuple[float, float] | None
    refutation_passed: bool
    refutation_methods: Tuple[str, ...]
    heterogeneous_effects: Dict[str, float]

# ----------------------------------------------------------------------
# 1. Hyperdimensional utilities (Parent A)
# ----------------------------------------------------------------------

def random_hv(d: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    """
    Generate a random hypervector of dimension *d*.

    - ``kind='complex'``  → unit‑modulus complex numbers e^{iθ}
    - ``kind='bipolar'``  → values in {‑1, +1}
    - ``kind='real'``     → L2‑normalised real Gaussian vector
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice(np.array([-1.0, 1.0]), size=d)
    if kind == "real":
        v = rng.standard_normal(d)
        norm = np.linalg.norm(v) + 1e-30
        return v / norm
    raise ValueError(f"kind must be 'complex', 'bipolar', or 'real'; got {kind!r}")

def fractional_bind(hv_a: np.ndarray, hv_b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Fractional binding between two hypervectors.

    The operation is defined as:
        bind = hv_a * (hv_b ** alpha)

    For complex hypervectors this corresponds to a rotation by a fraction
    ``alpha`` of the phase of ``hv_b``.
    """
    if hv_a.shape != hv_b.shape:
        raise ValueError("Hypervectors must share the same shape for binding.")
    return hv_a * (hv_b ** alpha)

def generate_causal_hypervector(effect: CausalEffect,
                                dim: int = 10000,
                                alpha: float = 0.5,
                                seed: int | None = None) -> np.ndarray:
    """
    Encode a ``CausalEffect`` as a hypervector.

    Steps:
    1. Create base hypervectors for treatment, outcome and each confounder.
    2. Bind them together using fractional binding.  The binding exponent ``alpha``
       controls the “strength” of each component.
    3. If an ATE estimate is available, rotate the resulting vector by an angle
       proportional to the estimate (mod 2π) to embed magnitude information.
    """
    rng = np.random.default_rng(seed)
    # Base vectors
    hv_treat = random_hv(dim, kind="complex", seed=rng.integers(0, 2**63))
    hv_out = random_hv(dim, kind="complex", seed=rng.integers(0, 2**63))
    hv = fractional_bind(hv_treat, hv_out, alpha)

    # Incorporate confounders
    for conf in effect.confounders:
        hv_conf = random_hv(dim, kind="complex", seed=rng.integers(0, 2**63))
        hv = fractional_bind(hv, hv_conf, alpha)

    # Encode the ATE magnitude as a global phase shift
    if effect.ate_estimate is not None:
        angle = (effect.ate_estimate % (2 * math.pi))
        hv = hv * np.exp(1j * angle)

    # Normalise to unit magnitude to keep the representation on the hypersphere
    hv = hv / (np.abs(hv).mean() + 1e-30)
    return hv

# ----------------------------------------------------------------------
# 2. Deterministic / Stochastic feature extraction (Parent B)
# ----------------------------------------------------------------------

def _deterministic_features(text: str, size: int = 128) -> np.ndarray:
    """
    Produce a reproducible feature vector by seeding ``random.Random`` with the
    hash of the input text.
    """
    rnd = random.Random(hash(text))
    return np.fromiter((rnd.random() for _ in range(size)), dtype=float)

def _stochastic_features(text: str, size: int = 128) -> np.ndarray:
    """
    Produce a stochastic feature vector that depends on the global RNG state.
    """
    # Use the text only to diversify the pattern a bit; the global RNG still
    # introduces stochasticity.
    base = sum(ord(c) for c in text) % 1000
    random.seed(base)  # affect the global RNG in a lightweight way
    return np.random.rand(size)

def fuse_text_features(text: str,
                       alpha: float = 0.5,
                       size: int = 128) -> np.ndarray:
    """
    Fuse deterministic and stochastic feature streams.

    The fused vector is ``alpha * det + (1‑alpha) * stoch``.
    """
    det = _deterministic_features(text, size)
    stoch = _stochastic_features(text, size)
    return alpha * det + (1.0 - alpha) * stoch

# ----------------------------------------------------------------------
# 3. Geometry: pairwise distances and Ollivier‑Ricci curvature
# ----------------------------------------------------------------------

def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise Euclidean distance matrix for rows of ``X``.
    """
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis=2)

def _neighbour_mask(distances: np.ndarray, cutoff: float) -> np.ndarray:
    """
    Boolean mask where ``True`` indicates that the distance is ≤ ``cutoff`` and
    not the diagonal element.
    """
    mask = (distances <= cutoff) & (distances > 0.0)
    return mask

def approximate_ollivier_ricci(distances: np.ndarray,
                               cutoff: float = 1.0) -> np.ndarray:
    """
    Approximate the Ollivier‑Ricci curvature matrix κ using the scheme described
    in Parent B.

    For each node i we define a lazy random‑walk measure μᵢ that is uniform over
    the neighbours within ``cutoff``.  The 1‑Wasserstein distance W₁(μᵢ, μⱼ) is
    approximated by the average absolute difference of the neighbour vectors.
    """
    n = distances.shape[0]
    curvature = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            mask_i = _neighbour_mask(distances, cutoff)
            mask_j = _neighbour_mask(distances, cutoff)

            if not mask_i.any() or not mask_j.any():
                continue

            W1 = np.mean(np.abs(distances[mask_i] - distances[mask_j].T))
            d_ij = distances[i, j]

            if d_ij > 0:
                curvature[i, j] = 1 - W1 / d_ij
                curvature[j, i] = curvature[i, j]

    return curvature

def nlms_curvature_update(weights: np.ndarray,
                          error: float,
                          curvature: np.ndarray,
                          learning_rate: float = 0.1) -> np.ndarray:
    """
    Update the weights using the NLMS algorithm with Ollivier‑Ricci curvature.

    The update rule is:
        w_new = w_old - learning_rate * error * (1 + κ) * w_old
    """
    return weights - learning_rate * error * (1 + curvature) * weights

if __name__ == "__main__":
    # Smoke test
    effect = CausalEffect("effect1", "treatment1", "outcome1", (), 0.5, None, True, (), {})
    hv = generate_causal_hypervector(effect)
    text_features = fuse_text_features("example text")
    distances = pairwise_euclidean(np.array([hv, text_features]))
    curvature = approximate_ollivier_ricci(distances)
    weights = np.random.rand(10)
    error = 0.1
    updated_weights = nlms_curvature_update(weights, error, curvature)