# DARWIN HAMMER — match 5606, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_omni_chaotic__hybrid_hybrid_hybrid_m926_s0.py (gen5)
# parent_b: hybrid_hybrid_sketches_rlct_hybrid_hybrid_bayes__m98_s0.py (gen4)
# born: 2026-05-30T00:03:25Z

"""Hybrid Sketch-Pheromone JEPA Engine
===================================

This module fuses the core topologies of:

* **Parent A** – *hybrid_hybrid_omni_chaotic__hybrid_hybrid_hybrid_m926_s0.py*  
  (LUCIDOTA chaotic omni‑front synthesis with JEPA energy‑based latent variable
  prediction and pheromone‑driven Fisher‑information routing).

* **Parent B** – *hybrid_hybrid_sketches_rlct_hybrid_hybrid_bayes__m98_s0.py*  
  (Count‑Min sketch for fast frequency estimation, Bayesian updates and
  RLCT‑style action selection).

**Mathematical bridge**  
The pheromone probabilities of Parent A are represented by a *Count‑Min sketch* –
a compact, probabilistic histogram of surface‑usage events.  
From this sketch we derive a *Fisher information matrix* that quantifies the
sensitivity of the pheromone distribution with respect to its underlying
parameters.  The Fisher matrix is then used as the *inverse covariance* of the
Gaussian prior in the JEPA energy function of Parent A:

\[
E(\mathbf{z}) = \tfrac12(\mathbf{z}-\boldsymbol\mu)^{\!\top}\,\mathbf{F}\,
(\mathbf{z}-\boldsymbol\mu),
\]

where \(\boldsymbol\mu\) is the normalized sketch (the empirical mean
pheromone probability) and \(\mathbf{F}\) is the Fisher information matrix.
Gradient descent on this energy updates the latent vector \(\mathbf{z}\) while
the sketch itself is continuously refined by Bayesian counting of new
pheromone events.

The three functions below demonstrate this hybrid operation:
`count_min_sketch`, `fisher_information_from_sketch`, and `jepa_energy_update`.
"""

import sys
import math
import random
import hashlib
import pathlib
from collections import defaultdict
from typing import Iterable, List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent B – Count‑Min Sketch utilities
# ----------------------------------------------------------------------
def count_min_sketch(
    items: Iterable[str],
    width: int = 64,
    depth: int = 4,
) -> List[List[int]]:
    """Return a Count‑Min sketch table for the given iterable of hashable strings.

    The table has shape (depth, width) and stores integer frequency estimates.
    """
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        # deterministic hash for reproducibility
        h = hashlib.md5(item.encode("utf-8")).digest()
        base = int.from_bytes(h, byteorder="big")
        for d in range(depth):
            # simple linear probing of the hash space per depth
            index = (base + d * 0x9e3779b9) % width
            table[d][index] += 1
    return table


def sketch_to_distribution(
    sketch: List[List[int]],
    eps: float = 1e-12,
) -> np.ndarray:
    """Convert a Count‑Min sketch to a normalized probability vector.

    The vector length equals `depth * width`.  Values are flattened and
    L1‑normalized.  Very small entries are clipped to `eps` to avoid division by zero.
    """
    flat = np.array(sketch, dtype=np.float64).flatten()
    total = flat.sum()
    if total < eps:
        return np.full_like(flat, 1.0 / flat.size)
    prob = flat / total
    prob = np.maximum(prob, eps)
    return prob / prob.sum()


# ----------------------------------------------------------------------
# Parent A – Fisher information and JEPA energy utilities
# ----------------------------------------------------------------------
def fisher_information_from_sketch(
    prob_vec: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Estimate a diagonal Fisher information matrix from a probability vector.

    For a categorical distribution the Fisher information for parameter p_i is
    `1 / p_i`.  We return a diagonal matrix `F = diag(1 / p_i)`.
    """
    inv = 1.0 / np.maximum(prob_vec, eps)
    return np.diag(inv)


def jepa_energy(
    latent: np.ndarray,
    mu: np.ndarray,
    fisher: np.ndarray,
) -> float:
    """Compute the JEPA energy `E(z) = 0.5 * (z-mu)^T * F * (z-mu)`."""
    diff = latent - mu
    return 0.5 * diff.T @ fisher @ diff


def jepa_energy_gradient(
    latent: np.ndarray,
    mu: np.ndarray,
    fisher: np.ndarray,
) -> np.ndarray:
    """Gradient of the JEPA energy with respect to the latent vector."""
    return fisher @ (latent - mu)


# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_update(
    latent: np.ndarray,
    sketch: List[List[int]],
    new_items: Iterable[str],
    lr: float = 0.1,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Perform one hybrid iteration:

    1. Update the Count‑Min sketch with `new_items` (Bayesian counting).
    2. Convert the updated sketch to a probability vector `mu`.
    3. Build the Fisher information matrix `F` from `mu`.
    4. Take a gradient‑descent step on the JEPA energy.

    Returns the updated latent vector and the updated sketch.
    """
    # 1. Bayesian sketch update
    updated_sketch = count_min_sketch(new_items, width=len(sketch[0]), depth=len(sketch))
    # merge with previous counts (simple additive merge)
    for d in range(len(sketch)):
        for w in range(len(sketch[0])):
            sketch[d][w] += updated_sketch[d][w]

    # 2. Distribution and prior mean
    mu = sketch_to_distribution(sketch)

    # 3. Fisher information (diagonal)
    F = fisher_information_from_sketch(mu)

    # 4. Gradient descent on JEPA energy
    grad = jepa_energy_gradient(latent, mu, F)
    latent_new = latent - lr * grad

    return latent_new, sketch


def latent_initialization(dim: int = 768, seed: int = 42) -> np.ndarray:
    """Create a deterministic latent vector (e.g., Gaussian init)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim)


def run_hybrid_cycle(
    initial_items: List[str],
    steps: int = 5,
    latent_dim: int = 768,
) -> np.ndarray:
    """Run a full hybrid cycle starting from `initial_items`.

    Returns the final latent vector after `steps` updates.
    """
    # initialise sketch and latent vector
    sketch = count_min_sketch(initial_items)
    latent = latent_initialization(dim=latent_dim)

    for step in range(steps):
        # generate synthetic new pheromone events
        new_items = [f"event_{random.randint(0, 1000)}" for _ in range(20)]
        latent, sketch = hybrid_update(latent, sketch, new_items, lr=0.05)

    return latent


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic demo
    demo_items = [f"init_{i}" for i in range(30)]
    final_latent = run_hybrid_cycle(demo_items, steps=3, latent_dim=128)

    # Print a concise summary
    print("Hybrid JEPA latent vector (first 8 components):")
    print(final_latent[:8])
    print("Norm of latent vector:", np.linalg.norm(final_latent))
    sys.exit(0)