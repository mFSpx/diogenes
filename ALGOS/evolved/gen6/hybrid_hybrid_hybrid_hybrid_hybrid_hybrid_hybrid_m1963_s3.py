# DARWIN HAMMER — match 1963, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m613_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_percyphon_hyb_m1057_s0.py (gen4)
# born: 2026-05-29T23:40:11Z

import numpy as np
import random
from dataclasses import dataclass, asdict
from typing import Any, Tuple

# ----------------------------------------------------------------------
# Parent A components (path signatures)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running = path[:-1] - path[0]               # (T‑1, d)
    return running.T @ increments               # (d, d)

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n = len(grid)
    m = len(x)

    t = np.concatenate((
        np.full(k, grid[0]),
        grid,
        np.full(k, grid[-1])
    ))

    N = np.zeros((m, n + k - 1), dtype=float)
    for i in range(n + k - 1):
        left = t[i]
        right = t[i + 1]
        N[:, i] = np.where((x >= left) & (x < right), 1.0, 0.0)
    N[:, -1] = np.where(x == t[-1], 1.0, N[:, -1])

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

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def reconstruction_risk_score(
    unique_quasi_identifiers: int,
    total_records: int,
    morphology: Morphology,
) -> float:
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
    S2 = signature_level2(path)
    eigvals = np.linalg.eigvalsh(S2)  
    eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)
    probs = eigvals / eigvals.sum()
    return -np.sum(probs * np.log(probs + 1e-12))

def morphology_entropy(morph: Morphology) -> float:
    sigma = sphericity_index(morph.length, morph.width, morph.height)
    phi = flatness_index(morph.length, morph.width, morph.height)
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
    ent_sig = signature_entropy(path)
    ent_morph = morphology_entropy(morphology)
    H = alpha * ent_sig + (1.0 - alpha) * ent_morph
    base_risk = reconstruction_risk_score(
        unique_quasi_identifiers, total_records, morphology
    )
    return base_risk * (1.0 + np.tanh(H))  # Improved risk score calculation

def hybrid_entity_generator(
    path: np.ndarray,
    morphology: Morphology,
    tier: ModelTier,
    seed: int = 42,
) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)

    risk = hybrid_risk_score(
        path,
        morphology,
        unique_quasi_identifiers=5,
        total_records=1000,
    )

    var_scale = min(max(risk, 0.01), 5.0)
    mean = np.array([morphology.length / 2, morphology.width / 2, morphology.height / 2])
    cov = np.diag([var_scale, var_scale, var_scale])

    points = np.random.multivariate_normal(mean, cov, size=tier.ram_mb // 10)
    return points

def calculate_optimal_alpha(
    path: np.ndarray,
    morphology: Morphology,
    unique_quasi_identifiers: int,
    total_records: int,
) -> float:
    alphas = np.linspace(0, 1, 100)
    risks = []
    for alpha in alphas:
        risk = hybrid_risk_score(
            path,
            morphology,
            unique_quasi_identifiers,
            total_records,
            alpha,
        )
        risks.append(risk)
    optimal_alpha = alphas[np.argmin(risks)]
    return optimal_alpha

def adaptive_hybrid_risk_score(
    path: np.ndarray,
    morphology: Morphology,
    unique_quasi_identifiers: int,
    total_records: int,
) -> float:
    optimal_alpha = calculate_optimal_alpha(
        path,
        morphology,
        unique_quasi_identifiers,
        total_records,
    )
    return hybrid_risk_score(
        path,
        morphology,
        unique_quasi_identifiers,
        total_records,
        optimal_alpha,
    )

# Example usage
if __name__ == "__main__":
    path = np.random.rand(10, 3)
    morphology = Morphology(1.0, 1.0, 1.0, 1.0)
    tier = ModelTier("example", 1024, "T1", 2048)
    risk = adaptive_hybrid_risk_score(path, morphology, 5, 1000)
    print("Adaptive Hybrid Risk Score:", risk)