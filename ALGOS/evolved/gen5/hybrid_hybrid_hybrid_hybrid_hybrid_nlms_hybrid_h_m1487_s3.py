# DARWIN HAMMER — match 1487, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s2.py (gen3)
# parent_b: hybrid_nlms_hybrid_hybrid_rbf_su_m223_s0.py (gen4)
# born: 2026-05-29T23:36:53Z

import sys
import random
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, List, Dict, Iterable, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A utilities (resource allocation)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where 0 = Sunday … 6 = Saturday."""
    return (datetime(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Produce a row‑stochastic weight vector for *groups* based on the day‑of‑week ``dow``.
    A sinusoidal rotation gives each group a smooth periodic weight.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    # Base sinusoid shifted by the weekday
    angles = 2 * math.pi * (np.arange(n) + dow) / n
    raw = np.sin(angles) + 1.0          # shift to non‑negative
    raw = np.maximum(raw, 0.0)          # guard against tiny negatives due to float error
    weight_vec = raw / raw.sum()        # normalise to sum‑to‑one
    return weight_vec.astype(np.float64)


def allocate_hybrid(groups: Sequence[str] = GROUPS) -> np.ndarray:
    """
    Allocate resources for the given *groups* based on the current UTC weekday.
    Returns a normalised weight vector.
    """
    now = datetime.now(timezone.utc)
    dow = doomsday(now.year, now.month, now.day)
    return weekday_weight_vector(groups, dow)


# ----------------------------------------------------------------------
# Parent‑B utilities (NLMS, RBF kernel, Hoeffding bound)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian (RBF) kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rbf_kernel_matrix(features: Dict[int, Sequence[float]], epsilon: float = 1.0) -> np.ndarray:
    """
    Build a symmetric RBF kernel matrix K_ij = exp(-epsilon^2 * ||x_i - x_j||^2).
    """
    n = len(features)
    K = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            dist = euclidean(features[i], features[j])
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """
    Hoeffding bound for a Bernoulli‑like random variable with range ``r``.
    Returns the half‑width of the confidence interval.
    """
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def predict(weights: Iterable[float], x: Iterable[float]) -> float:
    """Linear prediction ŷ = w·x."""
    return sum(w * xi for w, xi in zip(weights, x))


# ----------------------------------------------------------------------
# Hybrid core: resource‑weighted NLMS with kernel & Hoeffding adaptation
# ----------------------------------------------------------------------
def hybrid_nlms_update(
    weights: List[float],
    x: List[float],
    target: float,
    resource_weights: np.ndarray,
    K: np.ndarray,
    sample_idx: int,
    delta: float = 0.1,
    n_obs: int = 100,
    eps: float = 1e-9,
) -> Tuple[List[float], float]:
    """
    Perform a single NLMS weight update where the per‑coordinate step size is:

        μ_i = min( μ_base_i , HoeffdingBound )

    with

        μ_base_i = resource_weights[i] * similarity_i

    ``similarity_i`` is taken from the kernel matrix diagonal K_ii (self‑similarity)
    for the current sample ``sample_idx``.  The Hoeffding bound caps the learning
    rate to guarantee a statistical confidence on the error estimate.

    Returns the updated weight list and the instantaneous squared error.
    """
    if len(weights) != len(x):
        raise ValueError("weights and input vector must have the same length")
    if len(resource_weights) != len(x):
        raise ValueError("resource_weights length must match input dimension")

    # Linear prediction and error
    y_hat = predict(weights, x)
    error = target - y_hat
    sq_error = error ** 2

    # Normalisation term of NLMS (||x||^2 + eps)
    norm_x_sq = sum(xi * xi for xi in x) + eps

    # Similarity for each dimension (using diagonal of K)
    # Guard against out‑of‑range index
    if not (0 <= sample_idx < K.shape[0]):
        raise IndexError("sample_idx out of bounds for kernel matrix")
    similarity_vec = np.diag(K)[sample_idx]  # shape ()

    # Adaptive per‑coordinate learning rates
    mu_vec = np.multiply(resource_weights, similarity_vec)
    # Hoeffding bound as a global cap (same for all coordinates)
    bound = hoeffding_bound(r=1.0, delta=delta, n=n_obs)
    mu_vec = np.minimum(mu_vec, bound)

    # NLMS update: w_i ← w_i + μ_i * error * x_i / (||x||^2 + eps)
    new_weights = [
        w + mu * error * xi / norm_x_sq
        for w, mu, xi in zip(weights, mu_vec, x)
    ]

    return new_weights, sq_error


def hybrid_workshare_liquid_time(
    groups: Sequence[str],
    features: Dict[int, Sequence[float]],
    x: List[float],
    target: float,
    sample_idx: int,
    init_weights: List[float] = None,
) -> Tuple[List[float], float]:
    """
    End‑to‑end hybrid routine:

    1. Allocate weekday‑dependent resource weights for *groups*.
    2. Build the RBF kernel matrix from *features*.
    3. Update NLMS weights using the resource‑weighted, kernel‑modulated rule.

    Returns the updated NLMS weight vector and the squared error.
    """
    # 1. Resource allocation (same length as input vector is assumed)
    res_weights = allocate_hybrid(groups)
    if len(res_weights) != len(x):
        # If the number of groups differs from input dimension, broadcast by repetition
        repeats = math.ceil(len(x) / len(res_weights))
        res_weights = np.tile(res_weights, repeats)[: len(x)]

    # 2. Kernel matrix
    K = rbf_kernel_matrix(features)

    # 3. Use provided or zero initial weights
    if init_weights is None:
        init_weights = [0.0] * len(x)

    # 4. Perform the hybrid update
    updated_weights, error = hybrid_nlms_update(
        weights=init_weights,
        x=x,
        target=target,
        resource_weights=res_weights,
        K=K,
        sample_idx=sample_idx,
    )

    return updated_weights, error


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    groups = ("codex", "groq", "cohere")
    features = {0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]}
    x = [0.5, 0.6, 0.7]
    target = 1.0
    sample_idx = 0

    updated_weights, error = hybrid_workshare_liquid_time(
        groups, features, x, target, sample_idx
    )
    print(updated_weights)
    print(error)