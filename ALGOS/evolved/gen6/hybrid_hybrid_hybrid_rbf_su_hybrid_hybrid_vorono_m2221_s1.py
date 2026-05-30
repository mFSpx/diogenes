# DARWIN HAMMER — match 2221, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_rbf_surrogate_hybrid_hybrid_pherom_m411_s0.py (gen3)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_fold_c_m580_s0.py (gen5)
# born: 2026-05-29T23:41:23Z

"""Hybrid RBF‑Voronoi‑Bandit Algorithm
===================================

Parents
-------
* **Parent A** – `hybrid_hybrid_rbf_surrogate_hybrid_hybrid_pherom_m411_s0.py`  
  Provides a radial‑basis‑function (RBF) surrogate model and a pheromone
  mechanism based on entropy of surrogate scores.

* **Parent B** – `hybrid_hybrid_voronoi_parti_hybrid_hybrid_fold_c_m580_s0.py`  
  Supplies a Voronoi partition of the input space, a sheaf‑like restriction
  machinery and a bandit (UCB) policy that uses geometric proximity.

Mathematical Bridge
-------------------
The bridge is the **Voronoi partition** of the surrogate’s centre set.
Each Voronoi cell becomes a *node* of a sheaf‑style bandit system.
For a query point `x`

1. The nearest centre defines the active cell `c`.
2. The RBF surrogate produces a scalar prediction `ŷ(x)`.
3. A short history of predictions inside cell `c` yields an empirical
   distribution; its Shannon entropy `H_c` is turned into a pheromone
   intensity `τ_c = exp(-H_c)`.
4. A Upper‑Confidence‑Bound (UCB) bandit uses the pheromone `τ_c` as the
   reward estimate, biasing future sampling toward cells with low
   uncertainty (high pheromone).

Thus the surrogate’s algebraic kernel machinery, the geometric Voronoi
assignment, and the probabilistic bandit‑pheromone update are fused into a
single unified optimisation loop.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np

Vector = np.ndarray

# ----------------------------------------------------------------------
# Parent A – Radial‑Basis‑Function surrogate
# ----------------------------------------------------------------------
def _gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))

def _euclidean(a: Vector, b: Vector) -> float:
    return np.linalg.norm(a - b)

def _solve_linear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a·x = b with Gaussian elimination (no external libs)."""
    n = len(b)
    m = np.hstack((a.astype(float), b[:, None].astype(float)))
    for col in range(n):
        pivot = np.argmax(np.abs(m[col:, col])) + col
        if abs(m[pivot, col]) < 1e-12:
            raise ValueError("singular surrogate system")
        if pivot != col:
            m[[col, pivot]] = m[[pivot, col]]
        m[col] = m[col] / m[col, col]
        for row in range(n):
            if row == col:
                continue
            m[row] -= m[row, col] * m[col]
    return m[:, -1]

@dataclass(frozen=True)
class RBFSurrogate:
    """Thin wrapper around an RBF interpolant."""
    centers: np.ndarray          # shape (N, d)
    weights: np.ndarray          # shape (N,)
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Linear combination of Gaussian kernels."""
        dists = np.linalg.norm(self.centers - x, axis=1)
        kernels = np.vectorize(_gaussian)(dists, self.epsilon)
        return float(np.dot(self.weights, kernels))

def fit_rbf_surrogate(points: np.ndarray,
                     values: np.ndarray,
                     epsilon: float = 1.0,
                     ridge: float = 1e-9) -> RBFSurrogate:
    """Fit an RBF surrogate exactly (with optional ridge)."""
    if points.shape[0] != values.shape[0]:
        raise ValueError("points and values must have equal length")
    N = points.shape[0]
    K = np.empty((N, N), dtype=float)
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            K[i, j] = _gaussian(_euclidean(a, b), epsilon)
            if i == j:
                K[i, j] += ridge
    w = _solve_linear(K, values)
    return RBFSurrogate(points, w, epsilon)

# ----------------------------------------------------------------------
# Parent B – Voronoi partition + bandit policy
# ----------------------------------------------------------------------
def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def nearest_seed(point: np.ndarray, seeds: np.ndarray) -> int:
    """Return index of the nearest seed (ties broken by index)."""
    if seeds.size == 0:
        raise ValueError("seed set is empty")
    dists = np.linalg.norm(seeds - point, axis=1)
    return int(np.argmin(dists))

class VoronoiPartition:
    """Simple Voronoi partition based on a static seed set."""
    def __init__(self, seeds: np.ndarray):
        self.seeds = np.asarray(seeds, dtype=float)

    def region(self, x: np.ndarray) -> int:
        return nearest_seed(x, self.seeds)

# ----------------------------------------------------------------------
# Fusion – entropy‑based pheromones + UCB bandit
# ----------------------------------------------------------------------
def _shannon_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy, ignoring zero probabilities."""
    mask = probs > 0
    return -float(np.sum(probs[mask] * np.log(probs[mask])))

class HybridRBFVoronoiBandit:
    """
    Unified system:
    * RBF surrogate provides predictions.
    * Voronoi cells (centred on surrogate centres) are bandit arms.
    * Each arm keeps a short history of predictions; entropy of that
      history drives a pheromone value τ = exp(-H).
    * UCB uses τ as the estimated reward.
    """
    def __init__(self,
                 points: np.ndarray,
                 values: np.ndarray,
                 epsilon: float = 1.0,
                 ridge: float = 1e-9,
                 history_len: int = 20,
                 ucb_c: float = 1.0):
        # Core components
        self.surrogate = fit_rbf_surrogate(points, values, epsilon, ridge)
        self.partition = VoronoiPartition(self.surrogate.centers)

        # Bandit state per region
        self.history_len = max(1, int(history_len))
        self._histories = {i: [] for i in range(len(self.surrogate.centers))}
        self._counts = np.zeros(len(self.surrogate.centers), dtype=int)
        self._pheromones = np.ones(len(self.surrogate.centers), dtype=float)  # start neutral
        self.ucb_c = float(ucb_c)

    # ------------------------------------------------------------------
    # 1️⃣ Prediction + pheromone update
    # ------------------------------------------------------------------
    def predict(self, x: Vector) -> float:
        """Predict at x, update pheromone of the visited region."""
        region = self.partition.region(x)
        y_hat = self.surrogate.predict(x)

        # Update history (FIFO)
        hist = self._histories[region]
        hist.append(y_hat)
        if len(hist) > self.history_len:
            hist.pop(0)

        # Re‑compute pheromone from entropy
        probs = np.array(hist, dtype=float)
        if probs.size == 0:
            entropy = 0.0
        else:
            # Normalize to a probability distribution
            probs = probs - probs.min()  # shift to non‑negative
            total = probs.sum()
            probs = probs / total if total > 0 else np.full_like(probs, 1 / probs.size)
            entropy = _shannon_entropy(probs)
        self._pheromones[region] = math.exp(-entropy)

        # Bandit count increment
        self._counts[region] += 1
        return y_hat

    # ------------------------------------------------------------------
    # 2️⃣ Bandit arm selection (UCB)
    # ------------------------------------------------------------------
    def select_region(self) -> int:
        """Select a Voronoi region using Upper Confidence Bound."""
        total = self._counts.sum()
        if total == 0:
            # No pulls yet – pick uniformly at random
            return random.randrange(len(self._counts))
        ucb_values = self._pheromones + self.ucb_c * np.sqrt(
            2 * np.log(total) / np.maximum(self._counts, 1)
        )
        return int(np.argmax(ucb_values))

    # ------------------------------------------------------------------
    # 3️⃣ Action suggestion based on candidate points
    # ------------------------------------------------------------------
    def suggest_candidate(self, candidates: np.ndarray) -> Vector:
        """
        Given an (M, d) array of candidate points, choose the point that lies
        in the region with the highest UCB score. If several candidates belong
        to the same region, the one with the highest surrogate prediction is
        returned.
        """
        if candidates.ndim != 2:
            raise ValueError("candidates must be a 2‑D array")
        # Map each candidate to its region
        region_of = np.vectorize(self.partition.region, signature='(n)->()')
        cand_regions = region_of(candidates)

        # Compute UCB per region once
        ucb_per_region = np.empty(len(self._counts), dtype=float)
        total = self._counts.sum()
        for i in range(len(self._counts)):
            if self._counts[i] == 0:
                ucb_per_region[i] = float('inf')
            else:
                ucb_per_region[i] = (
                    self._pheromones[i] +
                    self.ucb_c * math.sqrt(2 * math.log(total) / self._counts[i])
                )
        # Choose best region (break ties by index)
        best_region = int(np.argmax(ucb_per_region))

        # Filter candidates belonging to that region
        mask = cand_regions == best_region
        if not np.any(mask):
            # Fallback: pick any candidate with highest UCB region anyway
            idx = random.randrange(len(candidates))
            return candidates[idx]

        # Within the region pick the point with highest surrogate prediction
        region_candidates = candidates[mask]
        preds = np.apply_along_axis(self.surrogate.predict, 1, region_candidates)
        best_idx = int(np.argmax(preds))
        return region_candidates[best_idx]

# ----------------------------------------------------------------------
# Public helper functions (required by the specification)
# ----------------------------------------------------------------------
def fit_surrogate(points: np.ndarray,
                  values: np.ndarray,
                  epsilon: float = 1.0,
                  ridge: float = 1e-9) -> RBFSurrogate:
    """Convenient wrapper around the RBF fitting routine."""
    return fit_rbf_surrogate(points, values, epsilon, ridge)

def update_pheromone(system: HybridRBFVoronoiBandit, x: Vector) -> float:
    """
    Perform a single prediction at ``x`` and return the updated pheromone
    value of the visited Voronoi cell.
    """
    _ = system.predict(x)               # side‑effects update internal state
    region = system.partition.region(x)
    return float(system._pheromones[region])

def select_action(system: HybridRBFVoronoiBandit,
                  candidates: np.ndarray) -> Vector:
    """
    Choose the next sampling point from ``candidates`` using the hybrid
    bandit‑pheromone policy.
    """
    return system.suggest_candidate(candidates)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a synthetic 2‑D regression problem
    rng = np.random.default_rng(42)
    X = rng.uniform(-5, 5, size=(50, 2))
    y = np.sin(X[:, 0]) + np.cos(X[:, 1])

    # Initialise the hybrid system
    hybrid = HybridRBFVoronoiBandit(
        points=X,
        values=y,
        epsilon=0.8,
        ridge=1e-6,
        history_len=10,
        ucb_c=1.5
    )

    # Perform a few predictions to populate pheromones
    for _ in range(20):
        test_pt = rng.uniform(-5, 5, size=2)
        pred = hybrid.predict(test_pt)
        _ = update_pheromone(hybrid, test_pt)  # just to exercise the helper

    # Propose a next sampling location from a random candidate pool
    candidate_pool = rng.uniform(-5, 5, size=(30, 2))
    next_pt = select_action(hybrid, candidate_pool)

    # Simple sanity checks (no exceptions should have been raised)
    assert isinstance(next_pt, np.ndarray) and next_pt.shape == (2,)
    print("Smoke test passed. Suggested point:", next_pt)