# DARWIN HAMMER — match 5229, survivor 2
# gen: 6
# parent_a: hybrid_physarum_network_hybrid_hybrid_hdc_hy_m683_s5.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1396_s4.py (gen4)
# born: 2026-05-30T00:00:45Z

"""Hybrid Physarum‑Bayesian Semantic Network
------------------------------------------------
Parent A: physarum_network‑style flux, conductance dynamics, high‑dimensional
vector binding/bundling (hyperdimensional computing).
Parent B: semantic neighbor Bayesian update, label scoring, and dynamic
resource allocation.

Mathematical bridge
~~~~~~~~~~~~~~~~~~~
- Each graph node is represented by a bipolar hyperdimensional vector
  `v_i = symbol_vector(id_i)`.  
- Node “pressure” is defined as the normalized dot‑product of the node
  vector with a global reference vector `r = bundle(all_vectors)`.  
- Edge length `ℓ_ij` is the Euclidean distance between the two node vectors.
- A semantic likelihood `L_ij` is obtained from the isotropic Gaussian
  kernel applied to `ℓ_ij` (parent A) – this plays the role of the
  likelihood in a Bayesian update (parent B).
- The prior probability for an edge is derived from its current conductance
  `g_ij` (scaled to `[0,1]`).  Using `bayes_marginal` and `bayes_update`
  we obtain a posterior weight `w_ij` that modulates the Physarum flux.
- Conductance update combines the Physarum flux with the Bayesian posterior,
  thus fusing both dynamical systems into a single unified rule.

The module implements three core hybrid functions demonstrating this
integration and provides a tiny smoke‑test at the end.
"""

import hashlib
import math
import random
import sys
import pathlib
from typing import List, Sequence, Tuple, Dict, Callable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Hyperdimensional utilities (Parent A)
# ----------------------------------------------------------------------
def random_vector(dim: int = 10_000, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=dim) * 2 - 1  # {-1, +1}


def symbol_vector(symbol: str, dim: int = 10_000) -> np.ndarray:
    h = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(h, "big")
    return random_vector(dim, seed)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return a * b


def bundle(vectors: Sequence[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("at least one vector is required")
    stacked = np.stack(vectors)
    sums = stacked.sum(axis=0)
    return np.where(sums >= 0, 1, -1)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return float(np.linalg.norm(a - b))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


# ----------------------------------------------------------------------
# Bayesian utilities (Parent B)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_node_pressure(node_vec: np.ndarray, reference_vec: np.ndarray) -> float:
    """Pressure = normalized dot product between node vector and reference."""
    dot = float(np.dot(node_vec, reference_vec))
    norm = float(np.linalg.norm(node_vec) * np.linalg.norm(reference_vec))
    return dot / norm if norm != 0 else 0.0


def semantic_likelihood(dist: float, epsilon: float = 1.0) -> float:
    """Treat Gaussian of Euclidean distance as a likelihood."""
    return gaussian(dist, epsilon)


def hybrid_edge_update(
    node_a_id: str,
    node_b_id: str,
    vectors: Dict[str, np.ndarray],
    conductance: float,
    prior: float,
    false_positive: float = 0.01,
    epsilon: float = 1.0,
) -> Tuple[float, float]:
    """
    Perform a single hybrid update for an edge (a,b).

    Returns
    -------
    new_conductance : float
        Updated conductance after flux and Bayesian modulation.
    posterior : float
        Posterior probability used as a scaling factor.
    """
    # 1. Pressures from hyperdimensional vectors
    ref_vec = bundle(list(vectors.values()))
    p_a = compute_node_pressure(vectors[node_a_id], ref_vec)
    p_b = compute_node_pressure(vectors[node_b_id], ref_vec)

    # 2. Edge length from Euclidean distance of vectors
    length_ij = euclidean(vectors[node_a_id], vectors[node_b_id]) + 1e-12

    # 3. Physarum‑style flux
    flux_ij = conductance / length_ij * (p_a - p_b)

    # 4. Semantic likelihood (Gaussian of distance)
    likelihood = semantic_likelihood(length_ij, epsilon)

    # 5. Bayesian posterior that modulates the flux
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)

    # 6. Hybrid conductance update: flux scaled by posterior + small decay
    decay = 0.01
    new_conductance = max(conductance + flux_ij * posterior - decay * conductance, 1e-6)

    return new_conductance, posterior


def hybrid_network_step(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    vectors: Dict[str, np.ndarray],
    conductances: Dict[Tuple[str, str], float],
    priors: Dict[Tuple[str, str], float],
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """
    Execute one synchronous update over the whole network.

    Returns
    -------
    new_conductances : dict
        Updated conductance values per edge.
    posteriors : dict
        Posterior probabilities per edge (useful for downstream resource allocation).
    """
    new_conductances: Dict[Tuple[str, str], float] = {}
    posteriors: Dict[Tuple[str, str], float] = {}

    for (a, b) in edges:
        key = (a, b) if (a, b) in conductances else (b, a)
        g = conductances.get(key, 1.0)
        prior = priors.get(key, 0.5)
        new_g, post = hybrid_edge_update(a, b, vectors, g, prior)
        new_conductances[key] = new_g
        posteriors[key] = post

    return new_conductances, posteriors


def label_score(text: str, label: str) -> float:
    """Very simple literal count based score (kept from Parent B)."""
    return float(text.count(label))


def allocate_resources(
    posteriors: Dict[Tuple[str, str], float],
    total_budget: float = 1.0,
) -> Dict[Tuple[str, str], float]:
    """
    Distribute a total resource budget across edges proportionally to their
    posterior probabilities (dynamic allocation inspired by Parent B).
    """
    total = sum(posteriors.values()) + 1e-12
    allocation = {e: (p / total) * total_budget for e, p in posteriors.items()}
    return allocation


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph
    node_ids = ["A", "B", "C"]
    vectors = {nid: symbol_vector(nid, dim=1024) for nid in node_ids}

    # Undirected edges (order will be normalized internally)
    edges = [("A", "B"), ("B", "C"), ("C", "A")]

    # Initialise conductances and priors
    conductances = {tuple(sorted(e)): 1.0 for e in edges}
    priors = {tuple(sorted(e)): 0.5 for e in edges}

    # Run a few hybrid steps
    for step in range(5):
        conductances, post = hybrid_network_step(
            nodes=node_ids,
            edges=edges,
            vectors=vectors,
            conductances=conductances,
            priors=priors,
        )
        # Update priors with the latest posteriors for the next iteration
        priors = post
        allocation = allocate_resources(post, total_budget=1.0)

        print(f"Step {step + 1}")
        for e in conductances:
            print(
                f"  Edge {e}: conductance={conductances[e]:.4f}, "
                f"posterior={post[e]:.4f}, allocation={allocation[e]:.4f}"
            )
    sys.exit(0)