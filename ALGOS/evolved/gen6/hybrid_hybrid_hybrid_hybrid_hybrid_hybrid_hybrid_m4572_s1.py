# DARWIN HAMMER — match 4572, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_sparse_wta_hy_m656_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1219_s2.py (gen5)
# born: 2026-05-29T23:56:35Z

"""Hybrid Module: hybrid_hybrid_hybrid_ternar_hybrid_sparse_wta_hy_m656_s1 + hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1219_s2
This module mathematically fuses the two parent algorithms by using the **TTT‑Linear weight matrix (W)** as a common linear operator.
The bridge is:

1. **Linear transformation** – `W @ x` (from both parents).
2. **Similarity evaluation** – Structural Similarity Index (SSIM) between the original input `x` and its transformed version `W@x` (used in parent B).
3. **Pruning / selection** – The SSIM value modulates a **Bayesian tree‑cost** term, which is then fed to a **sparse winner‑take‑all (WTA)** mechanism that decides which model instance stays in the model pool (from parent A).

Thus the hybrid algorithm updates `W` with the TTT loss gradient, computes a Bayesian‑cost that incorporates SSIM, and finally performs a sparse WTA eviction based on the resulting scores."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Core linear‑transform utilities (from Parent A)
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize a TTT‑Linear weight matrix `W` of shape (d_out, d_in)."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> float:
    """
    Self‑supervised reconstruction loss.
    If ``target`` is None the identity mapping is used (i.e. target == x).
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """
    Gradient of ``ttt_loss`` w.r.t. ``W``.
    Closed‑form: 2 * (Wx - t) * xᵀ
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)


# ----------------------------------------------------------------------
# SSIM for 1‑D vectors (simplified version, from Parent B)
# ----------------------------------------------------------------------
def ssim_1d(x: np.ndarray, y: np.ndarray, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """
    Compute a 1‑D Structural Similarity Index between ``x`` and ``y``.
    The formula reduces to the classic SSIM when the signals are 1‑D.
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape for SSIM.")
    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))
    sigma_x2 = float(np.var(x))
    sigma_y2 = float(np.var(y))
    sigma_xy = float(np.mean((x - mu_x) * (y - mu_y)))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Bayesian tree‑cost integration that uses SSIM (Parent B concept)
# ----------------------------------------------------------------------
def bayesian_tree_cost(
    W: np.ndarray,
    x: np.ndarray,
    prior: float = 0.5,
    beta: float = 1.0,
) -> float:
    """
    Compute a Bayesian‑style cost for a split/gain decision.
    The cost is composed of a prior term and a similarity‑driven term:

        cost = -log(prior) + beta * (1 - SSIM(x, W@x))

    Lower cost indicates a more promising split.
    """
    if not (0.0 < prior < 1.0):
        raise ValueError("prior must be in (0,1).")
    transformed = W @ x
    sim = ssim_1d(x, transformed)
    return -math.log(prior) + beta * (1.0 - sim)


# ----------------------------------------------------------------------
# Sparse Winner‑Take‑All (WTA) mechanism (Parent A concept)
# ----------------------------------------------------------------------
def winner_take_all(scores: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Return the indices of the top‑k scores (sparse WTA).
    If ``k`` exceeds the number of scores, all indices are returned.
    """
    if k <= 0:
        return np.array([], dtype=int)
    k = min(k, scores.size)
    # argsort descending
    return np.argpartition(-scores, k - 1)[:k]


# ----------------------------------------------------------------------
# Model abstraction and pool management
# ----------------------------------------------------------------------
@dataclass
class Model:
    """Container for a single model instance."""
    W: np.ndarray
    score: float = field(default=-math.inf)  # higher is better


class ModelPool:
    """
    Holds a limited number of models. New models replace the worst ones
    according to a sparse WTA based on their ``score``.
    """

    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.models: List[Model] = []

    def evaluate_score(self, model: Model, x: np.ndarray) -> float:
        """
        Composite score = -ttt_loss + (1 - bayesian_tree_cost)
        Higher score => more likely to survive.
        """
        loss = ttt_loss(model.W, x)
        cost = bayesian_tree_cost(model.W, x)
        return -loss + (1.0 - cost)

    def add_or_update(self, candidate: Model, x: np.ndarray) -> None:
        """Insert ``candidate`` into the pool, evicting if necessary."""
        candidate.score = self.evaluate_score(candidate, x)

        if len(self.models) < self.max_size:
            self.models.append(candidate)
            return

        # Determine the index of the worst model (lowest score)
        scores = np.array([m.score for m in self.models])
        worst_idx = np.argmin(scores)

        if candidate.score > self.models[worst_idx].score:
            self.models[worst_idx] = candidate

    def top_models(self, k: int = 1) -> List[Model]:
        """Return the top‑k models according to their scores."""
        if not self.models:
            return []
        scores = np.array([m.score for m in self.models])
        top_idx = winner_take_all(scores, k)
        return [self.models[i] for i in top_idx]


# ----------------------------------------------------------------------
# Hybrid step that ties everything together
# ----------------------------------------------------------------------
def hybrid_step(
    model: Model,
    x: np.ndarray,
    learning_rate: float = 0.01,
    prior: float = 0.5,
    beta: float = 1.0,
) -> Tuple[float, float]:
    """
    Perform a single hybrid update:
      1. Compute TTT loss and its gradient.
      2. Update ``W`` with gradient descent.
      3. Compute SSIM‑based Bayesian cost.
      4. Return the new loss and the cost (both scalars).

    The updated ``model.W`` is stored in‑place.
    """
    # 1. loss & gradient
    loss = ttt_loss(model.W, x)
    grad = ttt_grad(model.W, x)

    # 2. gradient descent step
    model.W -= learning_rate * grad

    # 3. similarity‑driven cost
    cost = bayesian_tree_cost(model.W, x, prior=prior, beta=beta)

    # 4. refresh model score for potential pool insertion
    model.score = -loss + (1.0 - cost)

    return loss, cost


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Synthetic input vector
    d = 16
    x = np.random.randn(d)

    # Initialise a base model
    W0 = init_ttt(d_in=d, scale=0.05, seed=42)
    base_model = Model(W=W0)

    # Create a pool and insert the base model
    pool = ModelPool(max_size=3)
    pool.add_or_update(base_model, x)

    # Run a few hybrid steps, inserting new candidates each iteration
    for step in range(5):
        # Clone current best model and perturb it slightly to create a candidate
        best = pool.top_models(k=1)[0]
        candidate_W = best.W + 0.01 * np.random.randn(*best.W.shape)
        candidate = Model(W=candidate_W)

        loss, cost = hybrid_step(candidate, x, learning_rate=0.005, prior=0.6, beta=0.8)
        pool.add_or_update(candidate, x)

        print(
            f"Step {step+1:02d} | Loss: {loss:.4f} | Cost: {cost:.4f} | "
            f"Pool size: {len(pool.models)}"
        )

    # Display the scores of the final pool
    print("\nFinal pool scores:")
    for i, m in enumerate(pool.models):
        print(f" Model {i}: score = {m.score:.4f}")