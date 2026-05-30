# DARWIN HAMMER — match 626, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_regret_m236_s1.py (gen4)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m29_s2.py (gen2)
# born: 2026-05-29T23:30:13Z

"""
Hybrid algorithm merging hybrid_hybrid_ternary_route_hybrid_bandit_router_m56_s0.py and hybrid_sparse_wta_hybrid_privacy_model_m29_s2.py.

Mathematical bridge:
1. The Structural Similarity Index (SSIM) from the first parent informs the selection of sparse expansions in the second parent.
2. The top-k sparse expansions are projected onto a high-dimensional space using locality-sensitive hashing.
3. The resulting expanded vectors are treated as queries whose aggregate (sum) is perturbed with Laplace noise to satisfy differential privacy.
4. The noisy aggregate is normalised and fed into the reconstruction-risk function risk = unique_quasi_identifiers / total_records.
5. This risk score is then used as the scale of a second Laplace noise term that governs whether a model may be admitted to the pool.
"""
import numpy as np
import math
import random
import sys
from pathlib import Path

# Constants
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)

# SSIM implementation
def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)

# Hybrid sparse expansion utilities
def hybrid_sparse_expansion(values: List[float], m: int, salt: str = "") -> List[float]:
    """Hash‑based sparse expansion of `values` into a vector of length `m`."""
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    # Use SSIM to inform the selection of sparse expansions
    ssim = compute_ssim(out, PROTOTYPE_VECTOR, dynamic_range=1.0)
    top_k = np.argsort(ssim)[::-1][:m]
    masked_out = [1 if i in top_k else 0 for i in range(m)]
    return [sign * v for i, sign in enumerate(masked_out) if sign > 0]

def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    return hybrid_sparse_expansion(values, m, salt)

# Regret-matching utilities
class MathAction:
    def __init__(self, id: str, expected_value: float, cost: float = 0.0, risk: float = 0.0):
        self.id = id
        self.expected_value = expected_value
        self.cost = cost
        self.risk = risk

# Hybrid privacy model utilities
def hybrid_privacy_model(
    expanded_vectors: List[List[float]], epsilon: float, delta: float
) -> List[float]:
    """Differentially private aggregation of `expanded_vectors`."""
    aggregates = [sum(vector) for vector in expanded_vectors]
    noisy_aggregates = [aggregate + np.random.laplace(0, 1 / epsilon, 1) for aggregate in aggregates]
    return noisy_aggregates

def reconstruction_risk(
    noisy_aggregates: List[float], unique_quasi_identifiers: int, total_records: int
) -> float:
    """Reconstruction risk function."""
    return unique_quasi_identifiers / total_records

def hybrid_admission_decision(
    risk_scores: List[float], epsilon: float, delta: float
) -> List[bool]:
    """Hybrid admission decision."""
    noisy_risk_scores = [score + np.random.laplace(0, 1 / epsilon, 1) for score in risk_scores]
    return [score < delta for score in noisy_risk_scores]

# Smoke test
if __name__ == "__main__":
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    m = 5
    salt = "test"
    expanded = hybrid_sparse_expansion(values, m, salt)
    print(expanded)
    epsilon = 1.0
    delta = 0.1
    expanded_vectors = [[v * 2 for v in expanded], [v * 3 for v in expanded]]
    noisy_aggregates = hybrid_privacy_model(expanded_vectors, epsilon, delta)
    risk_scores = [reconstruction_risk(noisy_aggregates, 2, 10) for _ in range(3)]
    admission_decisions = hybrid_admission_decision(risk_scores, epsilon, delta)
    print(admission_decisions)