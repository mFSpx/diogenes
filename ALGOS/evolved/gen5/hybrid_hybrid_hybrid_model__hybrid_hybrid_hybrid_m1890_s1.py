# DARWIN HAMMER — match 1890, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_hybrid_regret_m804_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_sheaf_cohomol_m179_s1.py (gen3)
# born: 2026-05-29T23:39:30Z

"""Hybrid VRAM‑Scheduler + Workshare‑Sheaf Model

This module fuses the two parent algorithms:

* **Parent A** – Hybrid VRAM Scheduler with Test‑Time Training (TTT) and
  Regret‑Weighted Liquid‑Time‑Constant MinHash in a hyper‑dimensional space.
* **Parent B** – Work‑share allocation via a sinusoidal weekday rotation and a
  cellular sheaf whose restriction maps are linear transformations.

**Mathematical bridge**

Both parents manipulate high‑dimensional vectors/matrices and rely on linear
operators:

* In A the regret‑weighted action is a hyper‑dimensional vector `r ∈ ℝᴰ`.
  The VRAM scheduler updates a weight matrix `W` by gradient descent.
* In B a weekday‑dependent stochastic weight vector `a ∈ ℝⁿ` (row‑stochastic)
  rotates the allocation among *groups*.  The sheaf structure treats each
  group as a stalk and the rotation as a linear restriction map.

The hybrid model therefore

1. **Projects** the weekday allocation `a` onto the hyper‑dimensional space
   (by broadcasting / scaling) and **binds** it with the regret vector `r`
   using element‑wise multiplication (the hyper‑dimensional “bind” operation).
2. **Hashes** the bound vector with a MinHash‑style signature to obtain a
   similarity measure that drives the learning rate of the VRAM scheduler.
3. **Updates** the weight matrix `W` with a gradient step that is accepted only
   if the resulting memory footprint respects a configurable VRAM budget.

The three core functions below illustrate this unified pipeline.  The final
`HybridModel` class encapsulates state and provides a `step` method that can be
used in a test‑time‑training loop.  A small smoke test is executed when the
module is run as a script."""

import sys
import math
import random
import hashlib
from pathlib import Path
from typing import Tuple, Sequence, List

import numpy as np

# ----------------------------------------------------------------------
# Constants (shared)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
DIM_HYPER = 1024                     # Dimensionality of hyper‑dimensional vectors
NUM_MINHASH = 128                    # Number of random hyperplanes for MinHash
DEFAULT_BUDGET_MB = 4096             # VRAM budget for the weight matrix


# ----------------------------------------------------------------------
# Parent‑B utilities (weekday allocation & sheaf‑like linear maps)
# ----------------------------------------------------------------------
def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Row‑stochastic weight vector derived from a sinusoidal rotation.
    `dow` is the weekday index where 0 = Sunday … 6 = Saturday.
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


def sheaf_restriction_map(vec: np.ndarray, group_idx: int) -> np.ndarray:
    """
    Linear restriction map for a given group (stalk).
    For demonstration we rotate the vector by a small angle that depends on
    `group_idx`.  The map is orthogonal, preserving norms.
    """
    theta = (group_idx + 1) * (math.pi / 180.0)   # a few degrees per group
    c, s = math.cos(theta), math.sin(theta)
    # Simple 2‑D rotation applied block‑wise
    reshaped = vec.reshape(-1, 2)
    rot = np.array([[c, -s], [s, c]])
    rotated = reshaped @ rot.T
    return rotated.ravel()


# ----------------------------------------------------------------------
# Parent‑A utilities (hyper‑dimensional bind, MinHash, VRAM handling)
# ----------------------------------------------------------------------
def bind(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Hyper‑dimensional bind – element‑wise multiplication followed by L2‑normalisation.
    """
    if v1.shape != v2.shape:
        raise ValueError("bind: vectors must have identical shape")
    bound = v1 * v2
    norm = np.linalg.norm(bound)
    return bound / norm if norm > 0 else bound


def _random_hyperplanes(num_perm: int, dim: int) -> np.ndarray:
    """Generate a fixed set of random hyperplanes for MinHash."""
    rng = np.random.default_rng(42)   # deterministic for reproducibility
    return rng.standard_normal(size=(num_perm, dim))


_HYPERPLANES = _random_hyperplanes(NUM_MINHASH, DIM_HYPER)


def minhash_signature(vec: np.ndarray) -> np.ndarray:
    """
    Compute a binary MinHash (SimHash) signature using random hyperplanes.
    Returns a bool array of length `NUM_MINHASH`.
    """
    if vec.shape[0] != DIM_HYPER:
        raise ValueError("minhash_signature: vector dimensionality mismatch")
    proj = vec @ _HYPERPLANES.T   # shape (NUM_MINHASH,)
    return proj > 0


def minhash_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    Jaccard‑like similarity for binary signatures (fraction of equal bits).
    """
    if sig1.shape != sig2.shape:
        raise ValueError("signatures must share shape")
    return np.mean(sig1 == sig2)


def vram_usage_mb(matrix: np.ndarray) -> float:
    """Estimate VRAM usage of a float64 matrix in megabytes."""
    bytes_used = matrix.nbytes
    return bytes_used / (1024.0 ** 2)


def budget_constrained_update(W: np.ndarray, grad: np.ndarray,
                              lr: float, budget_mb: float) -> np.ndarray:
    """
    Apply a gradient step `W -= lr * grad` only if the resulting matrix stays
    within the VRAM budget.  If the budget would be exceeded, the step size is
    scaled down proportionally.
    """
    proposed = W - lr * grad
    usage = vram_usage_mb(proposed)
    if usage <= budget_mb:
        return proposed
    # Scale learning rate so that the new usage exactly matches the budget.
    scale = (budget_mb / usage) ** 0.5   # sqrt because usage ∝ lr² for linear grad
    return W - (lr * scale) * grad


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_regret_vector(token_vec: np.ndarray, dow: int) -> np.ndarray:
    """
    Produce a regret‑weighted hyper‑dimensional vector for a token.
    Steps:
    1. Obtain weekday allocation `a`.
    2. Expand `a` to the hyper‑dimensional space by repeating / scaling.
    3. Apply a sheaf restriction map per group and bind the result with the token.
    """
    if token_vec.shape != (DIM_HYPER,):
        raise ValueError("token_vec must be of shape (DIM_HYPER,)")

    alloc = weekday_weight_vector(GROUPS, dow)          # shape (4,)
    # Expand allocation to match DIM_HYPER by tiling
    repeats = DIM_HYPER // len(alloc)
    expanded = np.tile(alloc, repeats)
    if expanded.shape[0] < DIM_HYPER:
        # Pad the remainder with the first element
        pad_len = DIM_HYPER - expanded.shape[0]
        expanded = np.concatenate([expanded, np.full(pad_len, alloc[0])])
    expanded = expanded.astype(np.float64)

    # Apply a simple sheaf restriction per group (rotate each slice)
    slice_len = DIM_HYPER // len(GROUPS)
    transformed = np.empty_like(token_vec)
    for i, group in enumerate(GROUPS):
        start = i * slice_len
        end = start + slice_len
        transformed[start:end] = sheaf_restriction_map(token_vec[start:end], i)

    # Bind the transformed token with the expanded allocation vector
    bound = bind(transformed, expanded)
    # Add a small random regret perturbation (simulating expected value, cost, risk)
    rng = np.random.default_rng()
    regret_noise = rng.normal(loc=0.0, scale=0.01, size=DIM_HYPER)
    return bound + regret_noise


def hybrid_gradient(W: np.ndarray, token_vec: np.ndarray,
                    regret_vec: np.ndarray) -> np.ndarray:
    """
    Simple gradient derived from the outer product of the token and regret vectors.
    This mimics a linear model where `W` maps token embeddings to regret predictions.
    """
    return np.outer(regret_vec, token_vec) / (np.linalg.norm(token_vec) + 1e-12)


def hybrid_step(W: np.ndarray, token_vec: np.ndarray, dow: int,
                lr: float, budget_mb: float) -> Tuple[np.ndarray, float]:
    """
    Perform one hybrid update step:
    * Compute the regret vector (weekday‑aware hyper‑dimensional binding).
    * Generate MinHash signatures before and after the update.
    * Derive a similarity‑scaled learning rate.
    * Apply a budget‑constrained gradient descent update.
    Returns the new weight matrix and the MinHash similarity that was used.
    """
    # Regret vector based on current token and weekday
    r_vec = hybrid_regret_vector(token_vec, dow)

    # Signature before update
    sig_before = minhash_signature(r_vec)

    # Gradient w.r.t. current W
    grad = hybrid_gradient(W, token_vec, r_vec)

    # Simulate a “future” regret vector after an imagined update (for similarity)
    W_proposed = W - lr * grad
    future_r = hybrid_regret_vector(token_vec, dow)  # same token, same dow
    sig_after = minhash_signature(future_r)

    sim = minhash_similarity(sig_before, sig_after)

    # Modulate learning rate with similarity (more similar -> smaller step)
    effective_lr = lr * (1.0 - sim)

    # Apply budget‑constrained update
    W_new = budget_constrained_update(W, grad, effective_lr, budget_mb)
    return W_new, sim


# ----------------------------------------------------------------------
# Model class encapsulating state
# ----------------------------------------------------------------------
class HybridModel:
    """
    Encapsulates the weight matrix, VRAM budget and provides a test‑time‑training
    interface that merges the parent algorithms.
    """

    def __init__(self, dim_input: int = DIM_HYPER,
                 dim_output: int = DIM_HYPER,
                 budget_mb: float = DEFAULT_BUDGET_MB,
                 base_lr: float = 0.01):
        self.W = np.zeros((dim_output, dim_input), dtype=np.float64)
        self.budget_mb = budget_mb
        self.base_lr = base_lr
        self.rng = np.random.default_rng(1234)   # token embedding generator

    def token_embedding(self) -> np.ndarray:
        """Generate a random hyper‑dimensional token embedding."""
        return self.rng.normal(loc=0.0, scale=1.0, size=DIM_HYPER).astype(np.float64)

    def step(self, dow: int) -> Tuple[np.ndarray, float]:
        """
        Execute a single hybrid step for the current weekday.
        Returns the updated weight matrix and the MinHash similarity.
        """
        token = self.token_embedding()
        self.W, sim = hybrid_step(self.W, token, dow,
                                  self.base_lr, self.budget_mb)
        return self.W, sim


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model = HybridModel()
    for day in range(7):  # iterate over a week
        W, sim = model.step(dow=day)
        usage = vram_usage_mb(W)
        print(f"Day {day}: similarity={sim:.4f}, VRAM usage={usage:.2f} MiB")
    print("Hybrid model ran without errors.")