# DARWIN HAMMER — match 2628, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_fisher_locali_jepa_energy_m52_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m140_s3.py (gen4)
# born: 2026-05-29T23:43:11Z

"""Hybrid Fisher‑JEPA‑Certainty Module
===================================

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – ``hybrid_hybrid_fisher_locali_jepa_energy_m52_s0.py``  
  Provides a *Gaussian beam* intensity model, a *Fisher information* scorer and a
  JEPA‑style energy term that measures representation mismatch.

* **Parent B** – ``hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m140_s3.py``  
  Supplies a *certainty* flag (scaled to ``[0,1]``), a *weighted Fisher* aggregator,
  a *regularized loss curvature* (RLCT) estimator based on a log‑log regression,
  and a lightweight *count‑min sketch* for streaming frequency estimation.

**Mathematical bridge**

Both parents treat *information* as a scalar that can be combined multiplicatively.
We therefore:

1. Scale the Fisher information of each observation by its epistemic certainty
   ``c ∈ [0,1]`` – yielding a **weighted Fisher score**.
2. Use the same certainty to weight the contribution of each loss term in the
   RLCT regression, producing a **certainty‑weighted RLCT estimate**.
3. Treat the JEPA energy ``E = ‖z_pred – z_true‖²`` as a base loss and multiply it
   by the weighted Fisher score, thus regularising the representation space with
   both statistical sensitivity (Fisher) and epistemic trust (certainty).

The three functions below – ``weighted_fisher_score``, ``weighted_rlct_estimate``,
and ``hybrid_metric`` – implement this unified view, while ``jepa_energy`` shows
the direct fusion of Fisher information with the JEPA loss.

All operations rely only on the Python standard library and NumPy."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Iterable, Tuple, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared statistical core (Gaussian beam & Fisher information)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single observation of a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Epistemic certainty handling
# ----------------------------------------------------------------------
def certainty_to_weight(confidence_bps: int) -> float:
    """
    Convert a confidence expressed in basis points (0‑10000) to a
    multiplicative weight in the interval [0, 1].
    """
    return max(0.0, min(1.0, confidence_bps / 10000.0))


# ----------------------------------------------------------------------
# Weighted Fisher aggregation
# ----------------------------------------------------------------------
def weighted_fisher_score(
    thetas: Iterable[float],
    centers: Iterable[float],
    widths: Iterable[float],
    confidences_bps: Iterable[int],
) -> float:
    """
    Compute the certainty‑weighted average Fisher information over a dataset.

    Parameters
    ----------
    thetas, centers, widths : iterables of equal length
        Observation angles and the corresponding beam parameters.
    confidences_bps : iterable of int
        Epistemic confidence for each observation, expressed in basis points.

    Returns
    -------
    float
        Weighted mean Fisher score.
    """
    thetas = list(thetas)
    centers = list(centers)
    widths = list(widths)
    confidences = [certainty_to_weight(c) for c in confidences_bps]

    if not (len(thetas) == len(centers) == len(widths) == len(confidences)):
        raise ValueError("All input iterables must have the same length")

    weighted_scores = [
        fisher_score(t, c, w) * conf
        for t, c, w, conf in zip(thetas, centers, widths, confidences)
    ]
    total_weight = sum(confidences) or 1.0
    return sum(weighted_scores) / total_weight


# ----------------------------------------------------------------------
# RLCT (regularized loss curvature) estimator with certainty weighting
# ----------------------------------------------------------------------
def weighted_rlct_estimate(
    losses: Iterable[float],
    confidences_bps: Iterable[int],
) -> Tuple[float, float]:
    """
    Estimate the RLCT exponent α and coefficient β from a log‑log regression
    of loss vs. index, weighting each point by epistemic confidence.

    The model is  log(loss) ≈ α·log(i) + log(β), where i is the 1‑based index.

    Returns
    -------
    (alpha, beta) : tuple of floats
        The fitted exponent and coefficient.
    """
    losses = np.asarray(list(losses), dtype=float)
    confidences = np.asarray([certainty_to_weight(c) for c in confidences_bps], dtype=float)

    if losses.shape != confidences.shape:
        raise ValueError("losses and confidences must have the same length")
    if np.any(losses <= 0):
        raise ValueError("All loss values must be positive for log‑log regression")

    indices = np.arange(1, losses.size + 1, dtype=float)

    # Weighted least‑squares solution for [α, logβ]
    X = np.vstack([np.log(indices), np.ones_like(indices)]).T
    W = np.diag(confidences)
    XtW = X.T @ W
    theta = np.linalg.inv(XtW @ X) @ (XtW @ np.log(losses))

    alpha, log_beta = theta
    beta = math.exp(log_beta)
    return float(alpha), float(beta)


# ----------------------------------------------------------------------
# Simple count‑min sketch (streaming frequency estimator)
# ----------------------------------------------------------------------
class CountMinSketch:
    """
    Very small count‑min sketch with a fixed number of hash functions.
    It stores integer counters in a 2‑D NumPy array.
    """

    def __init__(self, depth: int = 4, width: int = 256, seed: int = 0):
        self.depth = depth
        self.width = width
        self.tables = np.zeros((depth, width), dtype=np.int64)
        rng = np.random.default_rng(seed)
        # Generate a distinct random seed for each hash function
        self.seeds = rng.integers(low=1, high=2**31, size=depth, dtype=np.int64)

    def _hash(self, item: str, i: int) -> int:
        h = hash((item, int(self.seeds[i])))  # built‑in hash, deterministic per run
        return h % self.width

    def add(self, item: str, count: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.tables[i, idx] += count

    def estimate(self, item: str) -> int:
        return int(min(self.tables[i, self._hash(item, i)] for i in range(self.depth)))


# ----------------------------------------------------------------------
# JEPA energy regularised by Fisher and certainty
# ----------------------------------------------------------------------
def jepa_energy(
    z_pred: np.ndarray,
    z_true: np.ndarray,
    fisher: float,
    certainty_weight: float,
) -> float:
    """
    Compute a JEPA‑style energy term and regularise it with Fisher information
    and epistemic certainty.

    The base energy is the squared Euclidean distance between prediction and target.
    The regularised energy is:

        E_reg = (||z_pred - z_true||²) * (1 + certainty_weight * fisher)

    Parameters
    ----------
    z_pred, z_true : np.ndarray
        Representation vectors (same shape).
    fisher : float
        Fisher information for the observation that generated ``z_pred``.
    certainty_weight : float
        Epistemic certainty in [0, 1].

    Returns
    -------
    float
        Regularised JEPA energy.
    """
    if z_pred.shape != z_true.shape:
        raise ValueError("z_pred and z_true must have the same shape")
    base = np.linalg.norm(z_pred - z_true) ** 2
    return float(base * (1.0 + certainty_weight * fisher))


# ----------------------------------------------------------------------
# Hybrid metric that combines all three pillars
# ----------------------------------------------------------------------
def hybrid_metric(
    thetas: List[float],
    centers: List[float],
    widths: List[float],
    confidences_bps: List[int],
    representations_pred: List[np.ndarray],
    representations_true: List[np.ndarray],
) -> Dict[str, float]:
    """
    Build a count‑min sketch of the stringified observations, compute the
    weighted Fisher score, the weighted RLCT estimate of the JEPA energies,
    and finally return a consolidated metric.

    Returns a dictionary with keys:
        - "weighted_fisher"
        - "rlct_alpha"
        - "rlct_beta"
        - "average_jepa_energy"
        - "sketch_estimate" (total count of distinct items)
    """
    # 1. Count‑min sketch over a simple identifier for each observation
    cms = CountMinSketch()
    for i, (t, c, w) in enumerate(zip(thetas, centers, widths)):
        identifier = f"{t:.4f}_{c:.4f}_{w:.4f}"
        cms.add(identifier)

    sketch_estimate = sum(cms.estimate(f"{t:.4f}_{c:.4f}_{w:.4f}") for t, c, w in zip(thetas, centers, widths))

    # 2. Weighted Fisher score across the dataset
    w_fisher = weighted_fisher_score(thetas, centers, widths, confidences_bps)

    # 3. JEPA energies for each pair, regularised by Fisher and certainty
    energies = []
    for z_pred, z_true, theta, center, width, conf_bps in zip(
        representations_pred,
        representations_true,
        thetas,
        centers,
        widths,
        confidences_bps,
    ):
        fisher = fisher_score(theta, center, width)
        certainty = certainty_to_weight(conf_bps)
        energies.append(jepa_energy(z_pred, z_true, fisher, certainty))

    avg_energy = float(np.mean(energies))

    # 4. RLCT estimate on the (regularised) energies
    alpha, beta = weighted_rlct_estimate(energies, confidences_bps)

    return {
        "weighted_fisher": w_fisher,
        "rlct_alpha": alpha,
        "rlct_beta": beta,
        "average_jepa_energy": avg_energy,
        "sketch_estimate": float(sketch_estimate),
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic dataset
    N = 50
    random.seed(42)
    np.random.seed(42)

    thetas = [random.uniform(-math.pi, math.pi) for _ in range(N)]
    centers = [0.0 for _ in range(N)]                     # beam centred at 0
    widths = [random.uniform(0.1, 1.0) for _ in range(N)]
    confidences_bps = [random.randint(2000, 10000) for _ in range(N)]

    # Random representation vectors (dim = 8)
    dim = 8
    representations_true = [np.random.randn(dim) for _ in range(N)]
    # Add small noise to obtain predictions
    representations_pred = [
        true + 0.1 * np.random.randn(dim) for true in representations_true
    ]

    result = hybrid_metric(
        thetas,
        centers,
        widths,
        confidences_bps,
        representations_pred,
        representations_true,
    )

    print("Hybrid metric results:")
    for k, v in result.items():
        print(f"  {k}: {v:.6f}")

    # Simple sanity checks (should not raise)
    assert 0.0 <= result["weighted_fisher"]
    assert result["rlct_alpha"] != 0.0
    assert result["average_jepa_energy"] >= 0.0
    assert result["sketch_estimate"] > 0.0

    sys.exit(0)