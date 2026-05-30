# DARWIN HAMMER — match 5505, survivor 4
# gen: 7
# parent_a: hybrid_fisher_localization_hybrid_hybrid_path_s_m20_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m1258_s1.py (gen6)
# born: 2026-05-30T00:02:32Z

import numpy as np
import math
import hashlib
import random
from typing import List, Dict, Tuple, Any


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_information(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Scalar Fisher information for a 1‑D Gaussian with known variance."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def extract_features(theta: float, center: float, width: float) -> np.ndarray:
    """Return a feature vector for a single theta."""
    return np.array(
        [
            fisher_information(theta, center, width),
            gaussian_beam(theta, center, width),
            (theta - center) / width,
        ],
        dtype=float,
    )


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform of a path.
    Input shape: (T, d)
    Output shape: (2*T-1, 2*d)
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time × dimension)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])          # lead
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])  # lag
    out[-1] = np.concatenate([path[-1], path[-1]])
    return out


def first_level_signature(transformed: np.ndarray) -> np.ndarray:
    """Sum over time – the first level signature."""
    return transformed.sum(axis=0)


def second_level_signature(transformed: np.ndarray) -> np.ndarray:
    """Simple approximation of the second level signature via cumulative outer products."""
    cum = np.cumsum(transformed, axis=0)
    # outer product of cumulative sums at each step, then sum
    outer = np.einsum("ij,ik->ijk", cum, cum)
    return outer.sum(axis=0).reshape(-1)


def hybrid_path_signature(theta_vals: np.ndarray, center: float, width: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a deeper hybrid signature:
    - lead‑lag transformed feature path
    - concatenated first‑ and second‑level signatures
    """
    features = np.vstack([extract_features(t, center, width) for t in theta_vals])
    transformed = lead_lag_transform(features)
    lvl1 = first_level_signature(transformed)
    lvl2 = second_level_signature(transformed)
    return transformed, np.concatenate([lvl1, lvl2])


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for bounded random variables in [-r, r]."""
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")
    if n <= 0:
        raise ValueError("n must be positive")
    return math.sqrt((r ** 2) * math.log(2.0 / delta) / (2.0 * n))


def count_sketch_expand(values: List[float], m: int, seed: int = 0) -> np.ndarray:
    """
    Count‑Sketch style sparse expansion.
    Each value contributes to three random buckets with random sign.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    rng = random.Random(seed)
    out = np.zeros(m, dtype=float)
    for i, v in enumerate(values):
        for r in range(3):
            # deterministic hash based on i, r and seed
            h = hashlib.sha256(f"{seed}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if (h[8] & 1) else -1.0
            out[j] += sign * v
    return out


def prune_by_hoeffding(expanded: np.ndarray, delta: float, n_samples: int) -> np.ndarray:
    """
    Keep dimensions whose absolute value exceeds the Hoeffding threshold.
    """
    r = np.max(np.abs(expanded))  # worst‑case bound for each coordinate
    thresh = hoeffding_bound(r, delta, n_samples)
    mask = np.abs(expanded) > thresh
    return expanded * mask.astype(float)


def hybrid_compute_health_scores(endpoints: List[Dict[str, Any]], center: float, width: float) -> List[Dict[str, Any]]:
    """
    Compute health scores for a list of endpoints.
    Returns a list preserving original order and an explicit 'id' field.
    """
    out = []
    for idx, ep in enumerate(endpoints):
        theta = float(ep["health_score"])
        base = fisher_information(theta, center, width)
        health = base * (1.0 - float(ep["failure_rate"])) * float(ep["recovery_priority"])
        out.append(
            {
                "id": idx,
                "theta": theta,
                "base_fisher": base,
                "health_score": health,
                "failure_rate": ep["failure_rate"],
                "recovery_priority": ep["recovery_priority"],
            }
        )
    return out


def hybrid_update_endpoint(endpoint: Dict[str, Any], new_request: Dict[str, Any], center: float, width: float) -> Dict[str, Any]:
    """
    Update an endpoint with a new request.
    The health_score is recomputed using the latest Fisher information.
    """
    theta = float(endpoint["theta"])
    # simulate stochastic adjustments
    failure_rate = max(0.0, float(endpoint["failure_rate"]) * (1.0 - random.random() * 0.1))
    recovery_priority = float(endpoint["recovery_priority"]) * (1.0 + random.random() * 0.05)
    new_fisher = fisher_information(theta, center, width)
    new_health = new_fisher * (1.0 - failure_rate) * recovery_priority
    updated = {
        "id": endpoint["id"],
        "theta": theta,
        "base_fisher": new_fisher,
        "health_score": new_health,
        "failure_rate": failure_rate,
        "recovery_priority": recovery_priority,
    }
    return updated


def hybrid_sparse_signal_expansion(
    endpoints: List[Dict[str, Any]], center: float, width: float, m: int, delta: float = 0.05
) -> np.ndarray:
    """
    Full pipeline:
    1. Compute health scores.
    2. Expand via Count‑Sketch.
    3. Prune dimensions using Hoeffding bound.
    """
    health_objs = hybrid_compute_health_scores(endpoints, center, width)
    raw_scores = [obj["health_score"] for obj in health_objs]
    expanded = count_sketch_expand(raw_scores, m, seed=42)
    pruned = prune_by_hoeffding(expanded, delta=delta, n_samples=len(raw_scores))
    return pruned


if __name__ == "__main__":
    # Demo of the deeper hybrid integration
    theta_vals = np.array([0.1, 0.2, 0.3, 0.25])
    center = 0.2
    width = 0.1

    transformed_path, signature = hybrid_path_signature(theta_vals, center, width)
    print("Lead‑lag transformed path shape:", transformed_path.shape)
    print("Hybrid signature (first + second level) length:", signature.shape[0])

    endpoints = [
        {"health_score": 0.1, "failure_rate": 0.01, "recovery_priority": 0.1},
        {"health_score": 0.2, "failure_rate": 0.02, "recovery_priority": 0.2},
        {"health_score": 0.25, "failure_rate": 0.015, "recovery_priority": 0.15},
    ]

    updated = [hybrid_update_endpoint(ep, {}, center, width) for ep in hybrid_compute_health_scores(endpoints, center, width)]
    print("\nUpdated endpoint records:")
    for u in updated:
        print(u)

    m_dim = 16
    expanded_scores = hybrid_sparse_signal_expansion(endpoints, center, width, m=m_dim, delta=0.01)
    print("\nPruned expanded health scores (dimension {}):".format(m_dim))
    print(expanded_scores)