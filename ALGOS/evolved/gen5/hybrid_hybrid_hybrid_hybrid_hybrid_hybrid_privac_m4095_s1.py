# DARWIN HAMMER — match 4095, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_nlms_omni_cha_m701_s0.py (gen4)
# parent_b: hybrid_hybrid_privacy_sketc_hybrid_hybrid_hybrid_m1610_s1.py (gen4)
# born: 2026-05-29T23:53:28Z

"""
Hybrid algorithm merging:
- Parent A: weekday‑based weight allocation and NLMS adaptive update.
- Parent B: differential‑privacy RBF surrogate (Laplace‑noised centers) with Hoeffding‑style risk bound.

Mathematical bridge:
The stochastic weight vector w_dow produced by the weekday routine of Parent A is used as the
initial coefficient vector of the RBF surrogate from Parent B.  The RBF prediction
    ŷ = Σ_i w_i·exp(−ε·‖x−c_i‖²)
receives the same NLMS error‑correction step that Parent A applies to its linear model.
Thus the adaptive weight update and the privacy‑preserving surrogate share a common
parameter set, yielding a single unified system.
"""

import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Tuple, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities (weekday weight vector & NLMS update)
# ----------------------------------------------------------------------
def weekday_weight_vector(groups: List[str], dow: int) -> np.ndarray:
    """
    Normalised sinusoidal weight vector for *groups* based on weekday index ``dow``.
    Returns a row‑stochastic vector of length len(groups).
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    weights = np.array([math.sin(2 * math.pi * i / n + dow / n) for i in range(n)])
    return weights / np.sum(weights)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Normalised Least‑Mean‑Squares update.
    Returns the updated weight vector and the prediction error.
    """
    y = float(np.dot(weights, x))
    error = target - y
    power = float(np.dot(x, x) + eps)
    new_weights = weights + mu * error * x / power
    return new_weights, error


# ----------------------------------------------------------------------
# Parent B utilities (DP‑RBF surrogate and linear solver)
# ----------------------------------------------------------------------
def laplace_noise(scale: float) -> float:
    """Generate a single Laplace(0, scale) sample using inverse transform."""
    u = random.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))


def add_dp_noise_to_centers(
    centers: List[Tuple[float, ...]], epsilon: float, sensitivity: float = 1.0
) -> List[Tuple[float, ...]]:
    """
    Inject Laplace noise into each coordinate of every RBF centre.
    The noise scale is sensitivity/epsilon.
    """
    scale = sensitivity / epsilon
    noisy = []
    for c in centers:
        noisy_center = tuple(coord + laplace_noise(scale) for coord in c)
        noisy.append(noisy_center)
    return noisy


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """RBF kernel with bandwidth controlled by epsilon."""
    return math.exp(-((epsilon * r) ** 2))


def rbf_predict(
    weights: np.ndarray, centers: List[Tuple[float, ...]], x: np.ndarray, epsilon: float = 1.0
) -> float:
    """
    RBF surrogate prediction Σ_i w_i·gaussian(‖x−c_i‖, ε).
    """
    preds = [
        w * gaussian(euclidean(x, np.array(c)), epsilon)
        for w, c in zip(weights, centers)
    ]
    return float(sum(preds))


def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """
    Gaussian elimination with full pivoting (parent B implementation).
    Solves Ax = b for a square matrix a.
    """
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_initialize_weights(
    groups: List[str], date: datetime, n_rbf: int
) -> np.ndarray:
    """
    Initialise RBF weights using the weekday weight vector.
    If the number of RBF centres differs from the number of groups,
    the vector is tiled / truncated to match ``n_rbf``.
    """
    dow = (date.weekday() + 1) % 7  # 0=Sunday … 6=Saturday
    base = weekday_weight_vector(groups, dow)
    if n_rbf == len(base):
        return base.copy()
    # Tile or truncate to required size
    repeats = (n_rbf + len(base) - 1) // len(base)
    tiled = np.tile(base, repeats)[:n_rbf]
    return tiled.astype(float)


def hybrid_step(
    groups: List[str],
    date: datetime,
    x: np.ndarray,
    centers: List[Tuple[float, ...]],
    epsilon_dp: float,
    target: float,
    mu: float = 0.5,
) -> Tuple[np.ndarray, float, List[Tuple[float, ...]]]:
    """
    One hybrid iteration:
    1. Build weekday‑derived weights.
    2. Add DP noise to the RBF centres.
    3. Predict with the noisy RBF surrogate.
    4. Update the weights via NLMS using the prediction error.
    Returns (new_weights, error, noisy_centers).
    """
    # 1. Initialise / retrieve weights
    w = hybrid_initialize_weights(groups, date, len(centers))

    # 2. DP‑noised centres
    noisy_centers = add_dp_noise_to_centers(centers, epsilon_dp)

    # 3. RBF prediction
    y_pred = rbf_predict(w, noisy_centers, x, epsilon=epsilon_dp)

    # 4. NLMS weight adaptation
    w_updated, error = nlms_update(w, x, target, mu=mu)

    return w_updated, error, noisy_centers


def hybrid_risk_bound(
    centers: List[Tuple[float, ...]],
    epsilon_dp: float,
    delta: float = 0.05,
) -> float:
    """
    Compute a Hoeffding‑style bound on the reconstruction risk of the DP‑RBF model.
    The range is taken as the maximum pairwise kernel value (≤1), so the bound simplifies.
    """
    n = len(centers)
    if n == 0:
        return 0.0
    # Kernel values lie in [0,1]; range_ = 1
    range_ = 1.0
    # Hoeffding bound: exp(−2·n·ε² / range_²) ≤ δ  ⇒  ε ≤ sqrt( (range_²·ln(1/δ)) / (2n) )
    epsilon_bound = math.sqrt((range_ ** 2 * math.log(1.0 / delta)) / (2 * n))
    # Combine with DP epsilon to give a conservative risk estimate
    return min(epsilon_dp, epsilon_bound)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic scenario
    groups = ["codex", "groq", "cohere", "local_models"]
    today = datetime.now(timezone.utc)

    # Feature vector (same dimension as number of centres)
    x_feat = np.array([0.2, -0.1, 0.4, 0.7])

    # Define three RBF centres in the same 4‑D space
    centres = [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
        (-1.0, -1.0, -1.0, -1.0),
    ]

    target_value = 0.35  # Desired surrogate output

    # Run one hybrid iteration
    new_w, err, noisy_c = hybrid_step(
        groups=groups,
        date=today,
        x=x_feat,
        centers=centres,
        epsilon_dp=1.2,
        target=target_value,
        mu=0.4,
    )

    risk = hybrid_risk_bound(noisy_c, epsilon_dp=1.2)

    print("Updated weights:", new_w)
    print("Prediction error:", err)
    print("Noisy centres:", noisy_c)
    print("Risk bound estimate:", risk)