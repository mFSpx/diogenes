# DARWIN HAMMER — match 4213, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_koopman_operator_m632_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2500_s0.py (gen5)
# born: 2026-05-29T23:54:10Z

"""Hybrid Koopman‑MinHash Surrogate System
========================================

Parents
-------
* **Parent A** – ``hybrid_hybrid_hybrid_ternar_koopman_operator_m632_s3.py``  
  Provides a discrete audit time‑series ``x_t`` and builds a quadratic
  observable lift ``ψ(x)``.  Dynamic Mode Decomposition (DMD) yields a finite‑
  dimensional Koopman matrix **K** that propagates lifted states linearly.

* **Parent B** – ``hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2500_s0.py``  
  Supplies a MinHash‑based similarity measure, a radial‑basis‐function (RBF)
  surrogate model and a pheromone signalling mechanism used for
  clustering/selection.

Mathematical Bridge
-------------------
The Koopman eigenvectors ``v_i`` (columns of the eigen‑basis of **K**) are
high‑dimensional feature directions in the lifted space.  By treating each
eigenvector as a binary‑ish signature (e.g. sign of components) we can
apply the MinHash similarity from Parent B to cluster eigenvectors that
behave similarly.  Inside each cluster we fit an RBF surrogate that maps a
lifted state ``ψ(x)`` to a *cluster‑specific correction factor*; pheromones
weight these corrections according to recent forecasting error.

The hybrid pipeline therefore consists of:

1. Build audit snapshots and lift them (Parent A).
2. Fit the Koopman matrix **K** with DMD (Parent A).
3. Decompose **K**, cluster eigenvectors via MinHash (Parent B).
4. For every cluster, learn an RBF surrogate using the lifted snapshots
   belonging to that cluster (Parent B).
5. Forecast by linear Koopman evolution, then apply the cluster‑wise
   surrogate corrections weighted by pheromones.

The code below implements this fused algorithm with three public
functions that demonstrate the core steps.
"""

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple, Sequence, Iterable, Dict

import numpy as np

# ----------------------------------------------------------------------
# Utility functions (shared between parents)
# ----------------------------------------------------------------------


def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def quadratic_lift(x: np.ndarray) -> np.ndarray:
    """
    Quadratic observable ψ(x) = [x, x⊙x, x_i·x_j (i<j)]ᵀ.
    Returns a 1‑D array.
    """
    n = x.size
    # linear part
    linear = x.ravel()
    # elementwise square
    square = np.multiply(x, x).ravel()
    # cross terms
    cross = []
    for i in range(n):
        for j in range(i + 1, n):
            cross.append(x[i] * x[j])
    return np.concatenate([linear, square, np.array(cross, dtype=x.dtype)])


def minhash_signature(a: Sequence[float], b: Sequence[float]) -> int:
    """
    Very simple MinHash‑like signature: count equal quantised components.
    The original parent used exact equality; we keep that for simplicity.
    """
    return sum(1 for x, y in zip(a, b) if x == y)


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def radial_basis_surrogate(
    input_vec: Sequence[float],
    centers: List[Sequence[float]],
    widths: List[float],
) -> float:
    """Sum of Gaussian RBFs centred at ``centers`` with given ``widths``."""
    return sum(
        gaussian(euclidean(input_vec, c), 1.0 / w)
        for c, w in zip(centers, widths)
    )


# ----------------------------------------------------------------------
# Core Hybrid Classes
# ----------------------------------------------------------------------


class PheromoneSystem:
    """
    Stores a scalar pheromone value per cluster.  The value decays
    exponentially with a half‑life; it is reinforced when the cluster’s
    surrogate reduces the forecast error.
    """

    def __init__(self, half_life_seconds: float = 300.0):
        self.half_life = half_life_seconds
        self.values: Dict[int, float] = {}
        self.last_update: Dict[int, float] = {}

    def _decay(self, cluster_id: int, now: float) -> None:
        if cluster_id not in self.values:
            return
        dt = now - self.last_update.get(cluster_id, now)
        decay_factor = 0.5 ** (dt / self.half_life)
        self.values[cluster_id] *= decay_factor
        self.last_update[cluster_id] = now

    def reinforce(self, cluster_id: int, amount: float) -> None:
        now = datetime.now(timezone.utc).timestamp()
        self._decay(cluster_id, now)
        self.values[cluster_id] = self.values.get(cluster_id, 0.0) + amount
        self.last_update[cluster_id] = now

    def get(self, cluster_id: int) -> float:
        now = datetime.now(timezone.utc).timestamp()
        self._decay(cluster_id, now)
        return self.values.get(cluster_id, 0.0)


# ----------------------------------------------------------------------
# 1. Prepare snapshots & lift (Parent A)
# ----------------------------------------------------------------------


def prepare_audit_snapshots(
    raw_series: Iterable[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a sequence of raw audit vectors into a lifted data matrix.

    Returns
    -------
    X_lift : np.ndarray, shape (d, m)
        Lifted snapshots ``ψ(x_t)`` for t = 0 … m‑1.
    X_raw  : np.ndarray, shape (n, m)
        Original (unlifted) snapshots for possible downstream use.
    """
    raw_list = [np.asarray(v, dtype=float) for v in raw_series]
    n = raw_list[0].size
    lifted = [quadratic_lift(v) for v in raw_list]
    X_lift = np.column_stack(lifted)          # d × m
    X_raw = np.column_stack(raw_list)        # n × m
    return X_lift, X_raw


# ----------------------------------------------------------------------
# 2. Fit Koopman matrix with DMD (Parent A)
# ----------------------------------------------------------------------


def fit_koopman_dmd(lifted: np.ndarray) -> np.ndarray:
    """
    Perform exact DMD: K = Y * X⁺, where
        X = ψ(x_0)…ψ(x_{m‑2})
        Y = ψ(x_1)…ψ(x_{m‑1})

    Parameters
    ----------
    lifted : np.ndarray, shape (d, m)
        Lifted snapshot matrix.

    Returns
    -------
    K : np.ndarray, shape (d, d)
        Finite‑dimensional approximation of the Koopman operator.
    """
    X = lifted[:, :-1]          # d × (m‑1)
    Y = lifted[:, 1:]           # d × (m‑1)
    X_pinv = np.linalg.pinv(X)  # (m‑1) × d
    K = Y @ X_pinv
    return K


# ----------------------------------------------------------------------
# 3. Cluster Koopman eigenvectors via MinHash & fit RBF surrogates
# ----------------------------------------------------------------------


def cluster_modes_and_fit_surrogates(
    K: np.ndarray,
    lifted_snapshots: np.ndarray,
    n_centers_per_cluster: int = 5,
) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[float]], Dict[int, np.ndarray]]:
    """
    1. Eigen‑decompose K.
    2. Cluster eigenvectors using a simple MinHash signature against a
       reference binary pattern (sign vector).
    3. For each cluster, select ``n_centers_per_cluster`` lifted snapshots
       (randomly) as RBF centres and compute widths from pairwise distances.
    4. Return structures needed for surrogate evaluation.

    Returns
    -------
    cluster_centers : dict[int, list[np.ndarray]]
        RBF centre vectors per cluster.
    cluster_widths  : dict[int, list[float]]
        Width (scale) for each centre.
    eigenvectors    : dict[int, np.ndarray]
        Mapping cluster_id → eigenvector (representative of the cluster).
    """
    # 1. Eigen‑decomposition
    eigvals, eigvecs = np.linalg.eig(K)          # eigvecs columns are eigenvectors
    # Ensure real part only for clustering (complex parts are ignored)
    eigvecs_real = np.real(eigvecs)

    # 2. Simple MinHash clustering
    # Reference pattern = sign of first eigenvector
    reference = np.sign(eigvecs_real[:, 0])
    clusters: Dict[int, List[int]] = {}
    for idx in range(eigvecs_real.shape[1]):
        signature = minhash_signature(reference, np.sign(eigvecs_real[:, idx]))
        clusters.setdefault(signature, []).append(idx)

    # 3. Fit surrogates per cluster
    cluster_centers: Dict[int, List[np.ndarray]] = {}
    cluster_widths: Dict[int, List[float]] = {}
    eigenvector_repr: Dict[int, np.ndarray] = {}

    rng = np.random.default_rng(42)

    for cid, idx_list in clusters.items():
        # Representative eigenvector (first in the list)
        eigenvector_repr[cid] = eigvecs_real[:, idx_list[0]]

        # Randomly pick lifted snapshots as centres
        all_indices = np.arange(lifted_snapshots.shape[1])
        chosen = rng.choice(all_indices, size=min(n_centers_per_cluster, all_indices.size), replace=False)
        centres = [lifted_snapshots[:, i] for i in chosen]
        cluster_centers[cid] = centres

        # Widths: average distance to nearest neighbour among centres
        widths = []
        for c in centres:
            dists = [euclidean(c, other) for other in centres if not np.allclose(c, other)]
            widths.append(np.mean(dists) if dists else 1.0)
        # Guard against zero width
        widths = [w if w > 1e-8 else 1.0 for w in widths]
        cluster_widths[cid] = widths

    return cluster_centers, cluster_widths, eigenvector_repr


# ----------------------------------------------------------------------
# 4. Hybrid forecast using Koopman evolution + surrogate + pheromones
# ----------------------------------------------------------------------


def hybrid_forecast(
    current_raw: Sequence[float],
    K: np.ndarray,
    cluster_centers: Dict[int, List[np.ndarray]],
    cluster_widths: Dict[int, List[float]],
    eigenvectors: Dict[int, np.ndarray],
    pheromone_system: PheromoneSystem,
    steps: int = 5,
) -> List[np.ndarray]:
    """
    Produce a multi‑step forecast.

    Procedure for each step:
    * Lift the current state.
    * Linear evolution via K.
    * Project the lifted state onto each eigenvector, obtain a cluster id
      via MinHash (same rule as clustering).
    * Evaluate the RBF surrogate of that cluster and scale the lifted
      contribution.
    * Weight the correction by the current pheromone level.
    * Inverse‑lift (approximate) by taking the first ``n`` components of the
      lifted vector (the original dimension).

    Returns a list of raw‑space forecasts (numpy arrays).
    """
    n_raw = len(current_raw)
    lifted = quadratic_lift(np.asarray(current_raw, dtype=float))
    forecasts_raw: List[np.ndarray] = []

    # Pre‑compute reference signature for clustering (same as during training)
    # Use the sign of the first eigenvector as reference.
    any_eig = next(iter(eigenvectors.values()))
    reference_sign = np.sign(any_eig)

    for _ in range(steps):
        # Linear Koopman step
        lifted = K @ lifted

        # Determine cluster for this lifted vector
        sign_vec = np.sign(lifted)
        cluster_id = minhash_signature(reference_sign, sign_vec)

        # Surrogate correction (scalar)
        centres = cluster_centers.get(cluster_id, [])
        widths = cluster_widths.get(cluster_id, [])
        if centres and widths:
            correction = radial_basis_surrogate(lifted, centres, widths)
        else:
            correction = 0.0

        # Pheromone weighting
        pher = pheromone_system.get(cluster_id)
        corrected = lifted + pher * correction * lifted

        # Approximate inverse lift: take first n_raw entries (the linear part)
        raw_est = corrected[:n_raw]
        forecasts_raw.append(raw_est.copy())

        # Update pheromones: if correction reduced magnitude, reward
        improvement = -abs(correction)  # simplistic proxy
        pheromone_system.reinforce(cluster_id, improvement)

        # Prepare for next iteration
        lifted = corrected

    return forecasts_raw


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Generate a synthetic audit series: decaying sinusoid with noise
    rng = np.random.default_rng(0)
    T = 20
    raw_series = []
    for t in range(T):
        base = np.array([math.sin(0.2 * t), math.cos(0.2 * t), math.exp(-0.05 * t)])
        noise = rng.normal(scale=0.02, size=base.shape)
        raw_series.append(base + noise)

    # 1. Prepare and lift
    lifted, raw = prepare_audit_snapshots(raw_series)

    # 2. Fit Koopman operator
    K = fit_koopman_dmd(lifted)

    # 3. Cluster eigenvectors and fit surrogates
    centers, widths, eig_repr = cluster_modes_and_fit_surrogates(K, lifted)

    # 4. Initialise pheromone system
    pheromones = PheromoneSystem(half_life_seconds=120.0)

    # 5. Forecast from the last observed raw state
    forecast_steps = 8
    forecasts = hybrid_forecast(
        current_raw=raw[:, -1],
        K=K,
        cluster_centers=centers,
        cluster_widths=widths,
        eigenvectors=eig_repr,
        pheromone_system=pheromones,
        steps=forecast_steps,
    )

    # Print results
    print("Hybrid forecast (raw space):")
    for i, vec in enumerate(forecasts, 1):
        print(f" step {i}: {vec}")

    # Simple sanity check: shapes match original dimension
    assert all(f.shape == raw[:, 0].shape for f in forecasts), "Shape mismatch in forecast"
    print("Smoke test completed successfully.")