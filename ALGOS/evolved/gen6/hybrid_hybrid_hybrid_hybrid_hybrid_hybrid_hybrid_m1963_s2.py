# DARWIN HAMMER — match 1963, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m613_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_percyphon_hyb_m1057_s0.py (gen4)
# born: 2026-05-29T23:40:11Z

"""
Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m613_s1.py (path signature & lead‑lag transform)
- Parent B: hybrid_hybrid_hybrid_hybrid_percyphon_hyb_m1057_s0.py (morphology metrics & reconstruction risk)

Mathematical Bridge:
Both parents expose information‑theoretic quantities that can be expressed as entropy.
Parent A’s level‑2 signature matrix S₂ encodes path variability; its eigenvalue
distribution is a proxy for Shannon entropy of the trajectory.  
Parent B’s sphericity (σ) and flatness (φ) indices form a 2‑element probability‑like
vector after normalization, yielding a morphology entropy.  

The hybrid algorithm therefore:
1. Extracts a trajectory entropy from S₂.
2. Extracts a morphology entropy from (σ, φ).
3. Combines the two entropies (weighted sum) and feeds the result into the
   reconstruction‑risk model, producing a unified “hybrid risk score”.

The three core functions below demonstrate this integration.
"""

import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A components (path signatures)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Lead‑lag embedding of a 2‑D path."""
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path: np.ndarray) -> np.ndarray:
    """First level signature (net displacement)."""
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path: np.ndarray) -> np.ndarray:
    """Second level signature (iterated integral matrix)."""
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running = path[:-1] - path[0]               # (T‑1, d)
    return running.T @ increments               # (d, d)

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Simple Cox–de Boor implementation for B‑spline basis of order k.
    Returns a (len(x), len(grid)+k‑1) matrix where each column is a basis function.
    """
    x = np.asarray(x, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n = len(grid)
    m = len(x)

    # Augmented knot vector
    t = np.concatenate((
        np.full(k, grid[0]),
        grid,
        np.full(k, grid[-1])
    ))

    # Initial piecewise constant basis (order 1)
    N = np.zeros((m, n + k - 1), dtype=float)
    for i in range(n + k - 1):
        left = t[i]
        right = t[i + 1]
        N[:, i] = np.where((x >= left) & (x < right), 1.0, 0.0)
    # The last knot includes the right endpoint
    N[:, -1] = np.where(x == t[-1], 1.0, N[:, -1])

    # Recurrence for higher orders
    for order in range(2, k + 1):
        for i in range(n + k - order):
            denom1 = t[i + order - 1] - t[i]
            term1 = 0.0 if denom1 == 0 else ((x - t[i]) / denom1) * N[:, i]
            denom2 = t[i + order] - t[i + 1]
            term2 = 0.0 if denom2 == 0 else ((t[i + order] - x) / denom2) * N[:, i + 1]
            N[:, i] = term1 + term2
    return N[:, :n + k - 1]

# ----------------------------------------------------------------------
# Parent B components (morphology & risk)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int

TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1", 1024)
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2", 2048)
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2", 2048)
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3", 4096)

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    """Normalized ratio of the geometric mean to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    """Ratio of average planar size to vertical size."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def reconstruction_risk_score(
    unique_quasi_identifiers: int,
    total_records: int,
    morphology: Morphology,
) -> float:
    """
    Base risk score mixing re‑identification probability with shape factors.
    """
    if total_records <= 0:
        raise ValueError("total_records must be positive")
    base = unique_quasi_identifiers / total_records
    sigma = sphericity_index(morphology.length, morphology.width, morphology.height)
    phi = flatness_index(morphology.length, morphology.width, morphology.height)
    shape_factor = (1.0 - sigma) * phi
    return base * (1.0 + shape_factor)

# ----------------------------------------------------------------------
# Hybrid functions (mathematical fusion)
# ----------------------------------------------------------------------
def signature_entropy(path: np.ndarray) -> float:
    """
    Compute Shannon entropy of the eigenvalue distribution of the level‑2 signature.
    """
    S2 = signature_level2(path)
    eigvals = np.linalg.eigvalsh(S2)  # real symmetric → real eigenvalues
    # Ensure non‑negative and avoid log(0)
    eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)
    probs = eigvals / eigvals.sum()
    return -np.sum(probs * np.log(probs + 1e-12))

def morphology_entropy(morph: Morphology) -> float:
    """
    Compute entropy from normalized sphericity and flatness indices.
    """
    sigma = sphericity_index(morph.length, morph.width, morph.height)
    phi = flatness_index(morph.length, morph.width, morph.height)
    # Convert to a probability‑like vector
    raw = np.array([sigma, phi], dtype=float)
    raw = np.clip(raw, a_min=1e-12, a_max=None)
    probs = raw / raw.sum()
    return -np.sum(probs * np.log(probs + 1e-12))

def hybrid_risk_score(
    path: np.ndarray,
    morphology: Morphology,
    unique_quasi_identifiers: int,
    total_records: int,
    alpha: float = 0.6,
) -> float:
    """
    Unified risk score:
        H = α * Entropy_signature + (1‑α) * Entropy_morphology
        Risk = reconstruction_risk_score * (1 + H)
    """
    ent_sig = signature_entropy(path)
    ent_morph = morphology_entropy(morphology)
    H = alpha * ent_sig + (1.0 - alpha) * ent_morph
    base_risk = reconstruction_risk_score(
        unique_quasi_identifiers, total_records, morphology
    )
    return base_risk * (1.0 + H)

def hybrid_entity_generator(
    path: np.ndarray,
    morphology: Morphology,
    tier: ModelTier,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic entity (set of points) whose spatial spread is
    modulated by the hybrid risk score. The higher the risk, the larger the
    variance of the generated points, emulating a need for more robust
    anonymisation.
    """
    random.seed(seed)
    np.random.seed(seed)

    risk = hybrid_risk_score(
        path,
        morphology,
        unique_quasi_identifiers=5,
        total_records=1000,
    )

    # Scale variance with risk (bounded)
    var_scale = min(max(risk, 0.01), 5.0)
    mean = np.array([morphology.length / 2, morphology.width / 2, morphology.height / 2])
    cov = np.diag([var_scale, var_scale, var_scale])

    points = np.random.multivariate_normal(mean, cov, size=tier.ram_mb // 10)
    return points

def hybrid_path_morphology_transform(
    path: np.ndarray,
    morphology: Morphology,
) -> np.ndarray:
    """
    Apply lead‑lag transform to the path and then scale each dimension by
    morphological factors (sphericity and flatness) to embed shape information
    into the trajectory representation.
    """
    ll = lead_lag_transform(path)                     # (2T‑1, 2d)
    sigma = sphericity_index(morphology.length, morphology.width, morphology.height)
    phi = flatness_index(morphology.length, morphology.width, morphology.height)
    scale = np.array([sigma, phi] * (ll.shape[1] // 2))
    return ll * scale

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic path (3‑dimensional, 10 time steps)
    T = 10
    dim = 3
    np.random.seed(0)
    path = np.cumsum(np.random.randn(T, dim), axis=0)

    # Example morphology
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=10.0)

    # Compute hybrid risk
    risk = hybrid_risk_score(
        path,
        morph,
        unique_quasi_identifiers=7,
        total_records=5000,
    )
    print(f"Hybrid risk score: {risk:.6f}")

    # Generate synthetic entity
    points = hybrid_entity_generator(path, morph, TIER_T2_REASONING)
    print(f"Generated {points.shape[0]} points, first row: {points[0]}")

    # Transform path with morphology‑aware scaling
    transformed = hybrid_path_morphology_transform(path, morph)
    print(f"Transformed shape: {transformed.shape}")
    print(f"First transformed row: {transformed[0]}")