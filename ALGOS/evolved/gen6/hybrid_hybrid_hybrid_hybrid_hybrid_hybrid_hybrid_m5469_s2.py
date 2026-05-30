# DARWIN HAMMER — match 5469, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1811_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_hard_t_m92_s0.py (gen4)
# born: 2026-05-30T00:02:17Z

"""
Hybrid Decision Hygiene + Temperature Modulation ↔ RBF‑Surrogate Bridge

Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1811_s2.py) provides:
* a high‑dimensional “hygiene” vector built from weighted decision‑feature scores,
* a temperature‑dependent developmental rate (SchoolfieldParams) that modulates
  any scalar quantity.

Parent B (hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_hard_t_m92_s0.py) provides:
* an RBF‑surrogate that learns a mapping  x → y  by solving the dense linear system
  **K·w = y**, where **Kᵢⱼ = exp(‑ε²·‖xᵢ‑xⱼ‖²)**.

**Mathematical bridge** – the hygiene vector (dimension 10 000) is projected to a
low‑dimensional feature space and then supplied as the input **x** to the
RBF‑surrogate.  The temperature‑dependent rate λ(T) scales the RBF kernel width
ε, i.e. ε = ε₀·λ(T).  Consequently the surrogate learns under a temperature‑aware
similarity metric while still exploiting the rich decision‑hygiene representation.

The module implements:
* `temperature_rate` – Schoolfield temperature modulation.
* `hygiene_vector` – construction of the high‑dimensional decision vector.
* `project_vector` – random linear projection from 10 000 → low_dim.
* `hybrid_fit` – builds K with temperature‑scaled ε and solves for weights.
* `hybrid_predict` – kernel evaluation against the trained support set.
* `region_blade_product` – combines a stylometric fingerprint (character‑frequency
  vector) with a projected hygiene vector via element‑wise product, then feeds
  the result to the surrogate.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared type alias
# ----------------------------------------------------------------------
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Parent A – decision hygiene & temperature model
# ----------------------------------------------------------------------
_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 150, 150, 150], dtype=np.int64)

DIM = 10_000                # high‑dimensional HDC space
_SUB_DIM = DIM // len(_FEATURE_ORDER)   # chunk size per feature


@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters of the Schoolfield temperature model."""
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0      # J·mol⁻¹
    t_low: float = 283.15                     # K
    t_high: float = 307.15                    # K
    delta_h_low: float = -45_000.0            # J·mol⁻¹
    delta_h_high: float = 65_000.0            # J·mol⁻¹
    r_cal: float = 1.987                      # cal·mol⁻¹·K⁻¹ (≈8.314 J·mol⁻¹·K⁻¹ / 1000)


def temperature_rate(T: float, p: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Compute the temperature‑dependent developmental rate λ(T) using the
    full Schoolfield equation (simplified to avoid overflow).
    """
    R = p.r_cal * 1000.0  # convert to J·mol⁻¹·K⁻¹
    term_act = math.exp(-p.delta_h_activation / R * (1.0 / T - 1.0 / 298.15))
    term_low = 1.0 + math.exp(p.delta_h_low / R * (1.0 / p.t_low - 1.0 / T))
    term_high = 1.0 + math.exp(p.delta_h_high / R * (1.0 / p.t_high - 1.0 / T))
    return p.rho_25 * term_act / (term_low * term_high)


def hygiene_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Build a bipolar high‑dimensional vector from the supplied feature dict.
    Each feature occupies a contiguous sub‑vector of length _SUB_DIM.
    Positive weights add, negative weights subtract.
    """
    vec = np.zeros(DIM, dtype=np.float32)
    for idx, name in enumerate(_FEATURE_ORDER):
        val = float(features.get(name, 0.0))
        start = idx * _SUB_DIM
        end = start + _SUB_DIM
        # Apply positive and negative weighting
        pos = _POSITIVE_WEIGHTS[idx] * max(val, 0.0)
        neg = _NEGATIVE_WEIGHTS[idx] * max(-val, 0.0)
        vec[start:end] = pos - neg
    # Binarize to bipolar (+1 / –1) representation
    vec = np.sign(vec)
    vec[vec == 0] = 1.0
    return vec


# ----------------------------------------------------------------------
# Parent B – RBF surrogate
# ----------------------------------------------------------------------
def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))


@dataclass
class RBFSurrogate:
    """Trained RBF surrogate."""
    support_vectors: np.ndarray   # shape (n_samples, low_dim)
    weights: np.ndarray           # shape (n_samples,)
    epsilon: float                # kernel width


# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
_LOW_DIM = 64   # dimensionality after random projection (tunable)

# Fixed random projection matrix (Gaussian)
_rng = np.random.default_rng(seed=42)
_PROJECTION = _rng.normal(size=(DIM, _LOW_DIM)).astype(np.float32)


def project_vector(high_vec: np.ndarray) -> np.ndarray:
    """
    Linear random projection from DIM → _LOW_DIM.
    Normalizes to unit length to keep kernel scales comparable.
    """
    low = high_vec @ _PROJECTION
    norm = np.linalg.norm(low)
    return low / norm if norm > 0 else low


def hybrid_fit(
    hygiene_vectors: List[np.ndarray],
    targets: List[float],
    temperature_K: float,
    epsilon_base: float = 1.0,
) -> RBFSurrogate:
    """
    Fit an RBF surrogate on projected hygiene vectors.
    The kernel width ε is scaled by the temperature‑dependent rate λ(T).
    """
    if len(hygiene_vectors) != len(targets):
        raise ValueError("mismatched number of samples and targets")
    # Project and stack
    X = np.vstack([project_vector(v) for v in hygiene_vectors])   # (n, low_dim)
    n = X.shape[0]

    # Temperature‑scaled kernel width
    lam = temperature_rate(temperature_K)
    epsilon = epsilon_base * lam

    # Build dense kernel matrix K_{ij} = exp(-ε²·‖x_i‑x_j‖²)
    K = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            dist = euclidean(X[i], X[j])
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val   # symmetry

    # Solve K·w = y
    y = np.asarray(targets, dtype=np.float64)
    w = np.linalg.solve(K, y)

    return RBFSurrogate(support_vectors=X, weights=w, epsilon=epsilon)


def hybrid_predict(
    model: RBFSurrogate,
    hygiene_vector_new: np.ndarray,
) -> float:
    """
    Predict a scalar output for a new hygiene vector using the trained surrogate.
    """
    x_new = project_vector(hygiene_vector_new)          # (low_dim,)
    # Compute kernel between new point and each support vector
    diffs = model.support_vectors - x_new               # (n, low_dim)
    dists = np.linalg.norm(diffs, axis=1)               # (n,)
    ks = np.exp(-((model.epsilon * dists) ** 2))        # (n,)
    return float(np.dot(ks, model.weights))


def stylometric_fingerprint(text: str, dim: int = _LOW_DIM) -> np.ndarray:
    """
    Very simple stylometric feature: normalized character frequency vector,
    projected to the same low‑dimensional space as the hygiene vectors.
    """
    counts = np.zeros(256, dtype=np.float32)
    for ch in text.encode('utf-8', errors='ignore'):
        counts[ch] += 1.0
    if counts.sum() == 0:
        return np.zeros(dim, dtype=np.float32)
    freq = counts / counts.sum()
    # Random projection to low_dim (reuse the same matrix shape)
    proj = _rng.normal(size=(256, dim)).astype(np.float32)
    return freq @ proj


def region_blade_product(
    model: RBFSurrogate,
    hygiene_vec: np.ndarray,
    text: str,
) -> float:
    """
    Combine a stylometric fingerprint with a projected hygiene vector via
    element‑wise product (the “blade” product) and feed the result to the surrogate.
    """
    h_proj = project_vector(hygiene_vec)          # (low_dim,)
    s_proj = stylometric_fingerprint(text)       # (low_dim,)
    combined = h_proj * s_proj                    # element‑wise product
    # Use the same kernel machinery but with the combined vector as query
    diffs = model.support_vectors - combined
    dists = np.linalg.norm(diffs, axis=1)
    ks = np.exp(-((model.epsilon * dists) ** 2))
    return float(np.dot(ks, model.weights))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic decision‑feature dictionaries
    def random_features() -> Dict[str, float]:
        return {name: random.uniform(-1.0, 1.0) for name in _FEATURE_ORDER}

    sample_size = 8
    hygiene_vecs = [hygiene_vector(random_features()) for _ in range(sample_size)]
    # Synthetic targets could be e.g. expected reward from a bandit
    targets = [random.uniform(0.0, 10.0) for _ in range(sample_size)]

    # Fit at a moderate temperature (298 K ≈ 25 °C)
    model = hybrid_fit(hygiene_vecs, targets, temperature_K=298.15)

    # Predict for a fresh sample
    new_vec = hygiene_vector(random_features())
    pred = hybrid_predict(model, new_vec)
    print(f"Prediction (pure hygiene): {pred:.4f}")

    # Predict using stylometric‑augmented query
    sample_text = "The quick brown fox jumps over the lazy dog."
    pred2 = region_blade_product(model, new_vec, sample_text)
    print(f"Prediction (hygiene + stylometry): {pred2:.4f}")

    # Basic sanity check: predictions should be finite numbers
    assert math.isfinite(pred) and math.isfinite(pred2), "Non‑finite prediction encountered"
    print("Smoke test completed successfully.")