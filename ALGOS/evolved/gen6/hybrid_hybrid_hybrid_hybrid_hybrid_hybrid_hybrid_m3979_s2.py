# DARWIN HAMMER — match 3979, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m2681_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_nlms_omni_cha_m701_s1.py (gen4)
# born: 2026-05-29T23:52:54Z

"""Hybrid Algorithm: Fusion of Weak‑Supervision/Fisher JEPA with NLMS‑Adapted
Allocation Scheduler

Parents
-------
* **Parent A** – weak‑supervision labeling, Fisher‑based score and an
  RBF‑surrogate that maps a signal vector to a scalar output.
* **Parent B** – weekday‑aware allocation vector `w` and a Normalised
  Least‑Mean‑Squares (NLMS) rule that adapts `w` based on an error signal.

Mathematical Bridge
-------------------
Both parents operate on a *vector that sums to a constant*:
`w` in Parent B and the weight vector used by the RBF surrogate in Parent A.
We therefore apply the NLMS update **directly to the surrogate’s weight
vector**.  The error signal is defined as the mismatch between a target
Fisher score (derived from a weak‑supervision label) and the current Fisher
score computed from the scheduler’s memory‑usage vector.  The updated `w`
is then used both for the next scheduling step and as the weight vector of
the RBF surrogate, closing the loop between allocation learning and the
JEPA‑style scoring machinery.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Sequence, Callable

import numpy as np

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Core primitives from Parent A
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Radial‑basis Gaussian."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """Gaussian‑shaped Fisher information score."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def rbf_surrogate_predict(centers: np.ndarray, weights: np.ndarray, x: np.ndarray, epsilon: float = 1.0) -> float:
    """
    RBF surrogate output:
        y = Σ_i w_i * φ(‖x - c_i‖)
    where φ is the Gaussian kernel.
    """
    if centers.shape[0] != weights.shape[0]:
        raise ValueError("Number of centers must match number of weights")
    diffs = np.linalg.norm(centers - x, axis=1)
    kernels = np.vectorize(gaussian)(diffs, epsilon)
    return float(np.dot(weights, kernels))

# ----------------------------------------------------------------------
# Core primitives from Parent B
# ----------------------------------------------------------------------
def weekday_base_weights(day_index: int, groups: int = 8) -> np.ndarray:
    """
    Base allocation vector for a given weekday (0‑Monday … 6‑Sunday).
    Uses a sinusoidal pattern summed over `groups` harmonics and normalised
    to sum to 1.
    """
    d = day_index % 7
    angles = np.arange(groups)
    raw = np.sin(2 * math.pi * (d + angles) / groups) + 1.0
    normalized = raw / raw.sum()
    return normalized

def schedule_usage(allocation: np.ndarray, total_demand: np.ndarray, available: np.ndarray) -> np.ndarray:
    """
    Element‑wise schedule:
        usage_i = min( allocation_i * total_demand_i , available_i )
    """
    return np.minimum(allocation * total_demand, available)

def nlms_update(w: np.ndarray, error: float, x: np.ndarray, mu: float = 0.1, eps: float = 1e-12) -> np.ndarray:
    """
    NLMS weight adaptation:
        norm_x = x·x + eps
        w_new = w + mu * error * x / norm_x
    """
    norm_x = float(np.dot(x, x) + eps)
    return w + (mu * error * x) / norm_x

# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_step(
    day_index: int,
    total_demand: np.ndarray,
    available: np.ndarray,
    rbf_centers: np.ndarray,
    target_fisher: float,
    mu: float = 0.05,
    epsilon_rbf: float = 1.0,
) -> tuple[np.ndarray, float, float, float]:
    """
    Perform one hybrid iteration.

    Returns
    -------
    w_new          – updated allocation / RBF weight vector
    usage_mean     – mean memory usage after scheduling
    fisher_cur     – Fisher score computed from usage_mean
    rbf_output     – RBF surrogate output for the current usage vector
    """
    # 1. Build base allocation (weekday‑aware) and initialise weights
    w = weekday_base_weights(day_index, groups=rbf_centers.shape[0])
    # 2. Schedule memory usage
    usage = schedule_usage(w, total_demand, available)
    usage_mean = float(usage.mean())
    # 3. Compute current Fisher score from usage statistics
    fisher_cur = fisher_score(usage_mean, center=0.0, width=1.0)
    # 4. RBF surrogate prediction using the same vector `w` as weights
    rbf_output = rbf_surrogate_predict(rbf_centers, w, usage, epsilon=epsilon_rbf)
    # 5. Error signal: drive Fisher score towards the target
    error = target_fisher - fisher_cur
    # 6. NLMS adaptation of the allocation / weight vector
    w_new = nlms_update(w, error, w, mu=mu)
    # 7. Re‑normalise to keep the vector summing to 1 (allocation constraint)
    w_new = w_new / (w_new.sum() + 1e-12)
    return w_new, usage_mean, fisher_cur, rbf_output

def hybrid_scheduler(
    days: int,
    total_demand: np.ndarray,
    available: np.ndarray,
    rbf_centers: np.ndarray,
    target_fisher: float,
    mu: float = 0.05,
) -> list[dict]:
    """
    Run the hybrid algorithm over a sequence of weekdays.

    Parameters
    ----------
    days          – number of scheduling steps (each step uses the next weekday)
    total_demand  – 1‑D array of per‑task memory demand
    available     – 1‑D array of per‑task available memory
    rbf_centers   – array of shape (K, N) where K is number of RBF centres,
                    N matches `total_demand` dimension
    target_fisher – desired Fisher score to converge to
    mu            – NLMS step size

    Returns
    -------
    history – list of dictionaries with keys `day`, `w`, `usage_mean`,
              `fisher`, `rbf`.
    """
    history = []
    w = None
    for i in range(days):
        day_idx = (datetime.now(timezone.utc).weekday() + i) % 7
        w, usage_mean, fisher_cur, rbf_out = hybrid_step(
            day_idx,
            total_demand,
            available,
            rbf_centers,
            target_fisher,
            mu=mu,
        )
        history.append({
            "day": day_idx,
            "w": w.copy(),
            "usage_mean": usage_mean,
            "fisher": fisher_cur,
            "rbf": rbf_out,
        })
    return history

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic problem with 5 tasks
    TASKS = 5
    np.random.seed(42)
    total_demand = np.random.uniform(0.5, 2.0, size=TASKS)
    available = np.random.uniform(1.0, 3.0, size=TASKS)

    # Create 8 RBF centres randomly distributed in the same space
    K = 8
    rbf_centers = np.random.uniform(0.0, 2.5, size=(K, TASKS))

    target_fisher = 0.8  # Desired Fisher score

    # Run 10 days of hybrid scheduling
    hist = hybrid_scheduler(
        days=10,
        total_demand=total_demand,
        available=available,
        rbf_centers=rbf_centers,
        target_fisher=target_fisher,
        mu=0.07,
    )

    # Print a concise summary
    for entry in hist:
        print(
            f"Day {entry['day']}: "
            f"w={entry['w'].round(3)} | "
            f"usage_mean={entry['usage_mean']:.3f} | "
            f"fisher={entry['fisher']:.3f} | "
            f"rbf={entry['rbf']:.3f}"
        )