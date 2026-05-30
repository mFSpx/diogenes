# DARWIN HAMMER — match 4468, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_bandit_router_m512_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2013_s1.py (gen5)
# born: 2026-05-29T23:56:05Z

import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Sequence

import numpy as np


@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters for the Schoolfield temperature‑dependence model."""
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


def ssim(
    x: np.ndarray,
    y: np.ndarray,
    *,
    dynamic_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    One‑dimensional Structural Similarity Index Measure.

    The implementation follows the original SSIM definition but adapts the
    dynamic range to the data if it is not supplied.  The returned value lies
    in ``[-1, 1]``; for weighting purposes we later map it to ``[0, 1]``.
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.size == 0:
        raise ValueError("inputs must not be empty")

    if dynamic_range is None:
        # Use the combined range of both signals; avoid zero range.
        data_range = max(x.max(), y.max()) - min(x.min(), y.min())
        dynamic_range = data_range if data_range > 0 else 1.0

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)

    return float(numerator / denominator)


def caputo_kernel(alpha: float, delta: int) -> float:
    """
    Discrete Caputo fractional‑difference kernel.

    Parameters
    ----------
    alpha: order of the fractional derivative, 0 < alpha < 1
    delta: non‑negative integer lag
    """
    if delta < 0:
        raise ValueError("delta must be non‑negative")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in the open interval (0, 1)")

    gamma_term = math.gamma(2.0 - alpha)
    return ((delta + 1) ** (1.0 - alpha) - delta ** (1.0 - alpha)) / gamma_term


def fractional_memory_sum(alpha: float, allocations: Sequence[float]) -> float:
    """
    Fractional‑order memory accumulation of a sequence of allocation values.
    """
    total = 0.0
    t = len(allocations) - 1
    for k, a in enumerate(allocations):
        delta = t - k
        total += caputo_kernel(alpha, delta) * a
    return total


def developmental_rate(
    temp_k: float, params: SchoolfieldParams = SchoolfieldParams()
) -> float:
    """
    Schoolfield model for temperature‑dependent developmental rate.
    """
    if temp_k <= 0:
        raise ValueError("temperature must be positive Kelvin")
    if params.rho_25 < 0:
        raise ValueError("rho_25 must be non‑negative")

    # Helper to avoid repeated computation
    inv_temp = 1.0 / temp_k

    numerator = (
        params.rho_25
        * (temp_k / 298.15)
        * math.exp(
            (params.delta_h_activation / params.r_cal)
            * ((1.0 / 298.15) - inv_temp)
        )
    )
    low = math.exp(
        (params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - inv_temp)
    )
    high = math.exp(
        (params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - inv_temp)
    )
    return numerator / (low * high)


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


@dataclass
class RBFSurrogate:
    """
    Radial‑basis‑function surrogate that stores a collection of signal pairs
    (x_i, y_i) together with associated allocation weights.
    The SSIM between a query pair and each stored pair modulates the RBF
    contribution, providing a deeper coupling between the bandit‑router
    similarity measure and the surrogate model.
    """
    centers: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    allocations: List[float] = field(default_factory=list)
    epsilon: float = 1.0

    def add_center(self, x: np.ndarray, y: np.ndarray, allocation: float) -> None:
        """Append a new training example."""
        if x.shape != y.shape:
            raise ValueError("x and y must share the same shape")
        self.centers.append((x.copy(), y.copy()))
        self.allocations.append(allocation)

    def _distance(self, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> float:
        """
        Euclidean distance between concatenated signal vectors.
        This respects the original RBF notion of distance while keeping the
        dimensionality low.
        """
        v1 = np.concatenate([x1.ravel(), y1.ravel()])
        v2 = np.concatenate([x2.ravel(), y2.ravel()])
        return float(np.linalg.norm(v1 - v2))

    def predict(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the surrogate output for a query pair (x, y).

        The contribution of each stored centre is:
            w_i * SSIM(x, x_i) * SSIM(y, y_i) * φ(‖[x,y]‑[x_i,y_i]‖)

        where w_i is the original allocation weight and φ is the Gaussian RBF.
        """
        if not self.centers:
            raise RuntimeError("RBFSurrogate has no training data")

        # Normalise SSIM to [0, 1] for safe weighting
        ssim_norm = lambda a, b: (ssim(a, b) + 1.0) / 2.0

        total = 0.0
        for (cx, cy), w in zip(self.centers, self.allocations):
            sim_x = ssim_norm(x, cx)
            sim_y = ssim_norm(y, cy)
            r = self._distance(x, y, cx, cy)
            total += w * sim_x * sim_y * gaussian(r, self.epsilon)
        return total


def hybrid_rbf_surrogate(
    surrogate: RBFSurrogate,
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Wrapper that mirrors the original API but delegates to an RBFSurrogate
    instance.  This preserves backward compatibility while exposing the richer
    internal mechanics.
    """
    return surrogate.predict(x, y)


def hybrid_fusion(
    alpha: float,
    allocations: Sequence[float],
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Combine fractional memory with similarity‑weighted scaling.
    The SSIM term is mapped to [0, 1] to avoid negative scaling.
    """
    fms = fractional_memory_sum(alpha, allocations)
    sim = (ssim(x, y) + 1.0) / 2.0
    return fms * sim


def _demo() -> None:
    """Simple demonstration of the improved hybrid model."""
    # Build a surrogate with three training examples
    surrogate = RBFSurrogate(epsilon=0.8)

    training_data = [
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), 0.2),
        (np.array([2.0, 3.0, 4.0]), np.array([2.0, 3.0, 4.0]), 0.3),
        (np.array([0.5, 1.5, 2.5]), np.array([0.5, 1.5, 2.5]), 0.5),
    ]

    for x, y, alloc in training_data:
        surrogate.add_center(x, y, alloc)

    query_x = np.array([1.1, 2.1, 3.1])
    query_y = np.array([1.0, 2.0, 3.0])

    print("Hybrid RBF surrogate output:", hybrid_rbf_surrogate(surrogate, query_x, query_y))

    alpha = 0.5
    allocations = [0.2, 0.3, 0.5]
    print("Hybrid fusion output:", hybrid_fusion(alpha, allocations, query_x, query_y))


if __name__ == "__main__":
    _demo()