# DARWIN HAMMER — match 4934, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_semant_m1698_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_path_s_m851_s4.py (gen4)
# born: 2026-05-29T23:58:52Z

"""Hybrid NLMS‑Lead‑Lag‑Semantic Algorithm
========================================

This module fuses the two parent algorithms:

* **Parent A** – Normalised Least‑Mean‑Squares (NLMS) with curvature‑ and
  pheromone‑modulated learning rates.
* **Parent B** – Lead‑lag time‑series transform, weekday‑based group weighting
  and B‑spline basis utilities.

**Mathematical bridge**

For each training sample *i* we build a feature vector **xᵢ** by applying the
lead‑lag transform (Parent B) to a raw path and flattening the result.
A weekday‑dependent weight **wᵢ** (from ``weekday_weight_vector``) and a
graph‑derived curvature factor **cᵢ** (from a simple Ollivier‑Ricci‑like
approximation) modulate the base NLMS step size **μ₀** together with a
pheromone probability **πᵢ**:


μᵢ = μ₀ · cᵢ · πᵢ · wᵢ


The NLMS batch update then becomes


w ← w + Σ_i ( μᵢ·eᵢ / (‖xᵢ‖²+ε) )·xᵢ ,   eᵢ = yᵢ – w·xᵢ


Thus the algorithm simultaneously exploits graph geometry, semantic‑style
temporal features and a stochastic pheromone exploration mechanism.

The module provides three high‑level functions that illustrate this hybrid
behaviour:

1. ``hybrid_feature_matrix`` – builds the per‑sample feature matrix from
   lead‑lag transformed paths and applies a weekday‑group weighting.
2. ``sample_learning_rates`` – computes the per‑sample NLMS step sizes
   μᵢ using curvature, pheromone and weekday weights.
3. ``hybrid_nlms_batch_update`` – performs the NLMS batch update with the
   fused learning rates.

A minimal smoke test is executed when the file is run as a script.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import datetime as dt

# ----------------------------------------------------------------------
# Parent B – lead‑lag transform and weekday weighting
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Lead‑lag embedding of a 2‑D (time × dim) path."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (T, d)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Produce a normalised weight for each group based on the day‑of‑week.
    ``dow`` follows Python's ``datetime.weekday()`` convention (Mon=0).
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


# ----------------------------------------------------------------------
# Parent A – curvature and pheromone utilities
# ----------------------------------------------------------------------
def simple_ollivier_ricci_curvature(adj: np.ndarray) -> np.ndarray:
    """
    Very coarse Ollivier‑Ricci‑like curvature:
    c_i = 1 / (deg_i + 1)
    where deg_i is the (weighted) degree of node i.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency matrix must be square")
    deg = np.sum(np.abs(adj), axis=1)
    curvature = 1.0 / (deg + 1.0)
    return curvature.astype(np.float64)


def random_pheromone_vector(n: int, seed: int | None = None) -> np.ndarray:
    """
    Generate a stochastic pheromone probability vector π of length *n*
    using a Dirichlet distribution (α = 1 for uniformity).
    """
    rng = np.random.default_rng(seed)
    vec = rng.random(n) + 1e-6  # avoid exact zeros
    return (vec / vec.sum()).astype(np.float64)


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def hybrid_feature_matrix(
    paths: List[np.ndarray],
    groups: Sequence[str],
    date: dt.date,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the feature matrix X (samples × features) and a weekday weight vector w.

    * Each path is transformed by ``lead_lag_transform`` and flattened.
    * The resulting rows are stacked into X.
    * ``weekday_weight_vector`` is evaluated for the supplied groups and the
      weekday of *date*, yielding a per‑sample multiplicative weight.
    """
    if len(paths) != len(groups):
        raise ValueError("paths and groups must have the same length")

    transformed = [lead_lag_transform(p).ravel() for p in paths]
    X = np.vstack(transformed).astype(np.float64)

    dow = date.weekday()
    w = weekday_weight_vector(groups, dow)  # shape (n_samples,)
    return X, w


def sample_learning_rates(
    mu0: float,
    curvature: np.ndarray,
    pheromone: np.ndarray,
    weekday_weights: np.ndarray,
) -> np.ndarray:
    """
    Compute per‑sample NLMS step sizes μᵢ = μ₀·cᵢ·πᵢ·wᵢ.
    All inputs must be 1‑D arrays of equal length.
    """
    if not (curvature.shape == pheromone.shape == weekday_weights.shape):
        raise ValueError("All input vectors must have the same shape")
    return mu0 * curvature * pheromone * weekday_weights


def hybrid_nlms_batch_update(
    weights: np.ndarray,
    X: np.ndarray,
    targets: np.ndarray,
    mu0: float = 0.5,
    eps: float = 1e-9,
    curvature: np.ndarray | None = None,
    pheromone: np.ndarray | None = None,
    weekday_weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one NLMS batch update with the hybrid per‑sample learning rates.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (shape (n_features,)).
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    targets : np.ndarray
        Desired output vector (n_samples,).
    mu0, eps : float
        Global base learning rate and regularisation term.
    curvature, pheromone, weekday_weights : np.ndarray or None
        If ``None`` they are generated internally:
        * curvature → uniform 1.0
        * pheromone → random Dirichlet vector
        * weekday_weights → uniform 1.0

    Returns
    -------
    (new_weights, errors) : Tuple[np.ndarray, np.ndarray]
        Updated weight vector and the per‑sample error vector.
    """
    n_samples, n_features = X.shape
    if weights.shape != (n_features,):
        raise ValueError("weights shape mismatch with feature dimension")

    # Default auxiliary vectors
    if curvature is None:
        curvature = np.ones(n_samples, dtype=np.float64)
    if pheromone is None:
        pheromone = random_pheromone_vector(n_samples)
    if weekday_weights is None:
        weekday_weights = np.ones(n_samples, dtype=np.float64)

    mu_vec = sample_learning_rates(mu0, curvature, pheromone, weekday_weights)

    errors = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        x_i = X[i]
        y_i = targets[i]
        e_i = y_i - float(weights @ x_i)
        errors[i] = e_i
        denom = float(np.dot(x_i, x_i) + eps)
        weights = weights + (mu_vec[i] * e_i / denom) * x_i

    return weights, errors


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic graph (5 nodes) for curvature
    adj = np.array(
        [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ],
        dtype=float,
    )
    curvature = simple_ollivier_ricci_curvature(adj)

    # Generate dummy paths (time series) – 5 samples, each length 4, dim 2
    rng = np.random.default_rng(42)
    paths = [rng.random((4, 2)) for _ in range(5)]
    groups = ["A", "B", "C", "D", "E"]
    today = dt.date.today()

    # Build feature matrix and weekday weights
    X, weekday_w = hybrid_feature_matrix(paths, groups, today)

    # Random targets
    targets = rng.random(X.shape[0])

    # Initialise NLMS weight vector (zero)
    w = np.zeros(X.shape[1], dtype=np.float64)

    # Run hybrid NLMS update
    w_new, err = hybrid_nlms_batch_update(
        weights=w,
        X=X,
        targets=targets,
        mu0=0.3,
        curvature=curvature,
        pheromone=random_pheromone_vector(len(paths), seed=7),
        weekday_weights=weekday_w,
    )

    # Simple sanity check output
    print("Updated weights norm:", np.linalg.norm(w_new))
    print("Mean absolute error after update:", np.mean(np.abs(err)))
    sys.exit(0)