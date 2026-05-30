# DARWIN HAMMER — match 886, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m28_s0.py (gen4)
# parent_b: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s4.py (gen3)
# born: 2026-05-29T23:31:27Z

"""Hybrid Algorithm Fusion of TTT‑Linear/Count‑Min Sketch (Parent A) and Resource‑Budget Selector (Parent B)

Both parent algorithms operate on *vectors* that encode an entity’s characteristics:

* **Parent A** builds a ternary‑router weight matrix **W** (the TTT‑Linear matrix) that
  linearly maps an input feature vector **x** to a latent representation **y = W·x**.
  A Count‑Min sketch stores hashed quasi‑identifier frequencies; the reconstruction‑risk
  ratio **ρ = ‖y‑x‖² / ‖x‖²** measures how well the router preserves the original signal.

* **Parent B** aggregates each entity into a 2‑dimensional *resource vector*
  **r = [L , P]** where **L** is a “load” (e.g. weighted textual cue or spatial distance)
  and **P** is a “privacy” penalty.  A greedy linear‑budget selector chooses a binary
  inclusion vector **x̂** that respects spatial and privacy budgets.

**Mathematical Bridge**

We reinterpret the *load* dimension **L** as a linear functional of the latent
representation **y** produced by Parent A:


L = w⁺·y – w⁻·y


with positive/negative weight vectors **w⁺**, **w⁻** (analogous to the cue‑weight
vectors of Parent B).  

The *privacy* dimension **P** is formed from two sources that both originate in
Parent A:


P = β · CM_estimate(id) + γ · ρ


where **CM_estimate(id)** is the Count‑Min sketch estimate for the entity’s
quasi‑identifier **id**, **ρ** is the reconstruction‑risk ratio, and **β,γ**
are scalar knobs.

Thus each entity yields a resource vector **r = [L , P]** that can be fed to
Parent B’s linear‑budget selector, completing a true hybrid system that couples
the TTT‑Linear router, differential‑privacy sketch, and budget‑constrained
selection in a single mathematical pipeline.
"""

import sys
import math
import random
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A components: TTT‑Linear matrix and Count‑Min sketch
# ----------------------------------------------------------------------


def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize the TTT‑Linear weight matrix **W**."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Gradient of the squared‑error loss ‖W·x – target‖²."""
    if target is None:
        target = x
    residual = W @ x - target
    return 2.0 * np.outer(residual, x)


def ttt_step(W: np.ndarray, x: np.ndarray, eta: float, target: np.ndarray | None = None) -> np.ndarray:
    """One gradient‑descent step on **W**."""
    grad = ttt_grad(W, x, target)
    return W - eta * grad


def reconstruction_risk(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> float:
    """Return the reconstruction‑risk ratio ρ = ‖W·x – target‖² / ‖x‖²."""
    if target is None:
        target = x
    num = np.sum((W @ x - target) ** 2)
    den = np.sum(x ** 2) + 1e-12  # avoid division by zero
    return float(num / den)


# Simple Count‑Min sketch ------------------------------------------------


class CountMinSketch:
    """Count‑Min sketch with deterministic hash functions."""

    def __init__(self, depth: int = 5, width: int = 2 ** 10, seed: int = 0):
        self.depth = depth
        self.width = width
        self.tables = np.zeros((depth, width), dtype=np.int64)
        self.seed = seed
        self._hash_seeds = [(seed + i * 0x9e3779b9) & 0xFFFFFFFF for i in range(depth)]

    def _hash(self, key: str, i: int) -> int:
        h = hashlib.blake2b(digest_size=8, person=bytes([i]))
        h.update(key.encode())
        # Mix in per‑row seed
        h.update(self._hash_seeds[i].to_bytes(4, "little"))
        return int.from_bytes(h.digest(), "little") % self.width

    def update(self, key: str, inc: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(key, i)
            self.tables[i, idx] += inc

    def estimate(self, key: str) -> int:
        """Return the minimum count over all rows (the CM estimate)."""
        mins = []
        for i in range(self.depth):
            idx = self._hash(key, i)
            mins.append(self.tables[i, idx])
        return int(min(mins))


# ----------------------------------------------------------------------
# Parent B component: resource‑budget selector
# ----------------------------------------------------------------------


def select_under_budget(
    resources: np.ndarray,
    spatial_budget: float,
    privacy_budget: float,
) -> Tuple[np.ndarray, List[int]]:
    """
    Greedy selector that returns a binary inclusion vector **x̂** and the list
    of selected indices.  Items are considered in ascending order of the
    combined metric (L + P); an item is taken if adding it does not exceed
    either budget.
    """
    n = resources.shape[0]
    order = np.argsort(resources[:, 0] + resources[:, 1])  # L+P ascending
    selected = np.zeros(n, dtype=int)
    chosen_idxs = []
    cur_spatial = 0.0
    cur_privacy = 0.0
    for idx in order:
        L, P = resources[idx]
        if cur_spatial + L <= spatial_budget and cur_privacy + P <= privacy_budget:
            selected[idx] = 1
            chosen_idxs.append(idx)
            cur_spatial += L
            cur_privacy += P
    return selected, chosen_idxs


# ----------------------------------------------------------------------
# Hybrid pipeline utilities
# ----------------------------------------------------------------------


def deterministic_vector_from_id(identifier: str, dim: int) -> np.ndarray:
    """
    Produce a deterministic pseudo‑random vector of length *dim* from a string id.
    The seed is derived from a SHA‑256 hash of the identifier.
    """
    h = hashlib.sha256(identifier.encode()).digest()
    seed = int.from_bytes(h[:4], "little")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim)


def compute_resource_vector(
    W: np.ndarray,
    x: np.ndarray,
    sketch: CountMinSketch,
    identifier: str,
    w_pos: np.ndarray,
    w_neg: np.ndarray,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute the hybrid resource vector **r = [L , P]** for a single entity.

    * **L** = w⁺·y – w⁻·y   where y = W·x
    * **P** = β·CM_estimate(id) + γ·reconstruction_risk(W, x)
    """
    y = W @ x
    load = float(np.dot(w_pos, y) - np.dot(w_neg, y))
    sketch.update(identifier)  # count this occurrence
    privacy = beta * sketch.estimate(identifier) + gamma * reconstruction_risk(W, x)
    return load, privacy


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------


def _smoke_test() -> None:
    # Settings
    d_in = 32
    d_out = 32
    n_items = 100
    spatial_budget = 150.0
    privacy_budget = 200.0
    beta = 0.5
    gamma = 2.0
    eta = 0.001  # learning rate for W updates

    # Initialise components
    W = init_ttt(d_in, d_out, scale=0.05, seed=42)
    sketch = CountMinSketch(depth=5, width=2048, seed=123)
    rng = np.random.default_rng(999)

    # Random positive/negative weight vectors for the load dimension
    w_pos = rng.standard_normal(d_out)
    w_neg = rng.standard_normal(d_out)

    # Generate synthetic identifiers and feature vectors
    identifiers = [f"user_{i:05d}" for i in range(n_items)]
    xs = [deterministic_vector_from_id(id_, d_in) for id_ in identifiers]

    # One gradient step per item (simulating online learning)
    for x, id_ in zip(xs, identifiers):
        W = ttt_step(W, x, eta)  # simple unsupervised update

    # Build the resource matrix R ∈ ℝⁿˣ²
    resources = np.empty((n_items, 2), dtype=float)
    for i, (x, id_) in enumerate(zip(xs, identifiers)):
        L, P = compute_resource_vector(
            W, x, sketch, id_, w_pos, w_neg, beta=beta, gamma=gamma
        )
        resources[i, 0] = max(L, 0.0)  # load must be non‑negative for budgeting
        resources[i, 1] = max(P, 0.0)  # privacy penalty also non‑negative

    # Perform budgeted selection
    selection_vec, selected_idxs = select_under_budget(
        resources, spatial_budget, privacy_budget
    )

    # Simple sanity checks
    total_load = float(np.dot(resources[:, 0], selection_vec))
    total_priv = float(np.dot(resources[:, 1], selection_vec))
    assert total_load <= spatial_budget + 1e-6
    assert total_priv <= privacy_budget + 1e-6

    # Print summary
    print(f"Total items: {n_items}")
    print(f"Selected items: {len(selected_idxs)}")
    print(f"Load used / budget: {total_load:.2f} / {spatial_budget}")
    print(f"Privacy used / budget: {total_priv:.2f} / {privacy_budget}")


if __name__ == "__main__":
    _smoke_test()