# DARWIN HAMMER — match 3312, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m1861_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_percep_m1895_s2.py (gen5)
# born: 2026-05-29T23:49:10Z

"""
Hybrid Algorithm: Fusion of Simulated‑Annealing Leader Election / Physarum Network (Parent A)
with Bayesian VRAM‑Allocation Hygiene & Perceptual RBF (Parent B).

Mathematical Bridge
-------------------
* The broadcast probability `p_broadcast` (temperature‑driven prior of Parent A) is
  interpreted as a *prior* `p` for the Bayesian fusion of Parent B.
* The hygiene score `h` computed from the tree‑metric of Parent B is multiplied
  with `p_broadcast` to obtain a combined prior `π = p_broadcast * h`.
* The Physarum conductance `g` and edge length `ℓ` define a *flux* `φ = g·|ΔP|/ℓ`
  (ΔP = pressure difference).  Flux scales the width of the Gaussian kernel
  used in the RBF similarity `ℓ_RBF = σ·(1+φ)`.  Thus the network dynamics of
  Parent A modulate the *likelihood* `ℓike` supplied by Parent B.
* A Bayesian update
      posterior = (ℓike * π) / (ℓike * π + (1‑ℓike)*(1‑π))
  fuses the two topologies into a single probability that drives decision‑making
  and weight‑matrix adaptation.
* The TTT‑Linear weight matrix `W` from Parent A is updated by a gradient step
  proportional to the posterior error, closing the feedback loop.

The module implements three core hybrid operations:
1. `combined_prior` – merges broadcast probability with tree‑hygiene.
2. `scaled_rbf_likelihood` – computes an RBF similarity whose bandwidth is
   stretched by Physarum flux.
3. `bayesian_fusion_update` – performs the Bayesian fusion and updates `W`.

All functions are pure NumPy / std‑lib and can be exercised by the smoke test
at the bottom of the file.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set, Hashable, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Types shared by both parents
# ----------------------------------------------------------------------
Node = Hashable
Graph = Dict[Node, Set[Node]]
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Parent A – Simulated Annealing / Physarum utilities
# ----------------------------------------------------------------------
def broadcast_probability(phases: int, phase: int) -> float:
    """p = 1 / 2^{phases‑phase}, clamped to [0,1]."""
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Geometric cooling schedule."""
    return t0 * (alpha ** k)


def physarum_flux(pressure_a: float, pressure_b: float,
                  conductance: float, edge_length: float) -> float:
    """Flux φ = g * |ΔP| / ℓ."""
    if edge_length <= 0:
        raise ValueError("edge_length must be positive")
    return conductance * abs(pressure_a - pressure_b) / edge_length


def update_conductance(conductance: float, flux: float,
                       eta: float = 0.1) -> float:
    """Simple conductance adaptation: g_{new} = g + η·(φ - g)."""
    return conductance + eta * (flux - conductance)


# ----------------------------------------------------------------------
# Parent B – Tree hygiene & Perceptual RBF utilities
# ----------------------------------------------------------------------
def euclidean_length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """2‑D Euclidean distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_hygiene_score(
    nodes: Dict[Hashable, Tuple[float, float]],
    edges: List[Tuple[Hashable, Hashable]],
    root: Hashable,
) -> float:
    """
    Hygiene score = 1 / (1 + average root‑to‑leaf distance).
    The distance metric is the Euclidean length of edges.
    """
    # Build adjacency
    adj: Dict[Hashable, List[Hashable]] = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # BFS to compute distances from root
    visited: Set[Hashable] = {root}
    queue: List[Tuple[Hashable, float]] = [(root, 0.0)]
    leaf_distances: List[float] = []

    while queue:
        cur, dist = queue.pop(0)
        children = [nbr for nbr in adj[cur] if nbr not in visited]
        if not children:  # leaf
            leaf_distances.append(dist)
        for nxt in children:
            visited.add(nxt)
            edge_len = euclidean_length(nodes[cur], nodes[nxt])
            queue.append((nxt, dist + edge_len))

    if not leaf_distances:
        return 1.0
    avg_dist = sum(leaf_distances) / len(leaf_distances)
    return 1.0 / (1.0 + avg_dist)


def rbf_similarity(
    x: Vector,
    y: Vector,
    sigma: float = 1.0,
) -> float:
    """Gaussian RBF: exp(-||x‑y||² / (2σ²))."""
    diff = np.asarray(x) - np.asarray(y)
    sq_norm = np.dot(diff, diff)
    return math.exp(-sq_norm / (2.0 * sigma * sigma))


# ----------------------------------------------------------------------
# Hybrid Core Functions (the mathematical fusion)
# ----------------------------------------------------------------------
def combined_prior(
    phases: int,
    phase: int,
    hygiene: float,
) -> float:
    """
    Fuse Parent A's broadcast probability with Parent B's hygiene score.

    π = p_broadcast * hygiene
    """
    p = broadcast_probability(phases, phase)
    return max(0.0, min(1.0, p * hygiene))


def scaled_rbf_likelihood(
    feature_vec: Vector,
    reference_vec: Vector,
    base_sigma: float,
    flux: float,
) -> float:
    """
    Compute RBF likelihood where the kernel width is stretched by Physarum flux.

    σ_eff = base_sigma * (1 + φ)
    ℓike = rbf_similarity(feature_vec, reference_vec, σ_eff)
    """
    sigma_eff = base_sigma * (1.0 + flux)
    return rbf_similarity(feature_vec, reference_vec, sigma_eff)


def bayesian_fusion_update(
    prior: float,
    likelihood: float,
    W: np.ndarray,
    learning_rate: float = 0.05,
) -> Tuple[float, np.ndarray]:
    """
    Perform Bayesian fusion and update the TTT‑Linear weight matrix.

    posterior = (ℓike * π) / (ℓike * π + (1‑ℓike)*(1‑π))

    Gradient step on W:
        ∂L/∂W = (posterior - target) * feature_outer
    where `target` is a dummy 1.0 (we aim for high posterior) and
    `feature_outer` is an outer product of a synthetic feature vector.
    """
    # Clamp to avoid division by zero
    pi = max(1e-9, min(1.0 - 1e-9, prior))
    li = max(1e-9, min(1.0 - 1e-9, likelihood))

    posterior = (li * pi) / (li * pi + (1.0 - li) * (1.0 - pi))

    # Dummy feature vector for gradient (could be any problem‑specific vector)
    feature = np.ones(W.shape[0])
    grad = (posterior - 1.0) * np.outer(feature, feature)

    W_new = W - learning_rate * grad
    return posterior, W_new


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ----- Parent‑A side setup -----
    phases, phase = 5, 3
    pressure_a, pressure_b = 1.2, 0.4
    conductance = 0.8
    edge_length = 2.0

    flux = physarum_flux(pressure_a, pressure_b, conductance, edge_length)
    conductance = update_conductance(conductance, flux)

    # ----- Parent‑B side setup -----
    nodes = {
        "root": (0.0, 0.0),
        "n1": (1.0, 0.0),
        "n2": (0.0, 1.0),
        "leaf": (1.0, 1.0),
    }
    edges = [("root", "n1"), ("root", "n2"), ("n1", "leaf")]
    hygiene = tree_hygiene_score(nodes, edges, "root")

    feature_vec = [0.2, 0.8, 0.5]
    reference_vec = [0.0, 1.0, 0.0]
    base_sigma = 0.5

    # ----- Hybrid computation -----
    prior = combined_prior(phases, phase, hygiene)
    likelihood = scaled_rbf_likelihood(feature_vec, reference_vec, base_sigma, flux)

    # Initialise a small TTT‑Linear weight matrix
    W = np.eye(len(feature_vec))

    posterior, W_updated = bayesian_fusion_update(prior, likelihood, W)

    # Simple sanity prints (not required by spec but harmless)
    print(f"Flux: {flux:.4f}")
    print(f"Hygiene: {hygiene:.4f}")
    print(f"Prior (π): {prior:.4f}")
    print(f"Likelihood (ℓ): {likelihood:.4f}")
    print(f"Posterior: {posterior:.4f}")
    print(f"Updated W:\n{W_updated}")

    # Ensure no NaNs propagate
    assert not np.isnan(posterior), "Posterior is NaN"
    assert not np.isnan(W_updated).any(), "Weight matrix contains NaN"

    sys.exit(0)