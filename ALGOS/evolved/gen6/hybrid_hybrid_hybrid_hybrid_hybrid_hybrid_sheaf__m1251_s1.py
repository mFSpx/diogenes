# DARWIN HAMMER — match 1251, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m767_s0.py (gen5)
# parent_b: hybrid_hybrid_sheaf_cohomol_hybrid_shannon_entro_m6_s0.py (gen2)
# born: 2026-05-29T23:36:08Z

"""Hybrid Endpoint‑Sheaf‑Entropy‑Hoeffding Algorithm

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – Hybrid Endpoint‑SSM‑Hoeffding‑Tropical module.  
  It computes a *health score* for each service endpoint and decides,
  via a Hoeffding bound, whether to switch to a better endpoint.

* **Parent B** – Hybrid Sheaf‑Cohomology‑Shannon‑Entropy‑RSA cipher.  
  It builds a sheaf (a collection of vector‑valued sections over a graph)
  and measures the Shannon entropy of the information carried by the sections.

**Mathematical bridge** – The health scores of the endpoints are encoded as
high‑dimensional *hypervectors* (the hyperdimensional primitive of Parent B).
These hypervectors are stored as sections of a sheaf.  The Shannon entropy
of the distribution of section values quantifies the uncertainty of the
current endpoint configuration, while the Hoeffding bound provides a
statistical guarantee on the health‑score estimates.  By combining the
entropy (information‑theoretic) and Hoeffding (concentration‑inequality)
criteria we obtain a more robust decision rule for endpoint switching.

The public API offers three representative hybrid operations:
    1. `hybrid_compute_health_scores` – compute health scores for a list of endpoints.
    2. `hybrid_encode_and_update_sheaf` – encode each health score into a hypervector
       and store it as a sheaf section.
    3. `hybrid_maybe_switch` – decide, using Hoeffding bound, Fisher‑like variance
       of hypervectors and sheaf entropy, whether to switch to a better endpoint.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (adapted from the parents)
# ----------------------------------------------------------------------

@dataclass
class Endpoint:
    """Endpoint state used by Parent A."""
    id: int
    failure_rate: float          # probability of failure per unit time
    recovery_priority: float    # morphological recovery priority (higher = better)
    health_score: float = 0.0   # will be filled by `hybrid_compute_health_scores`


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a random variable bounded in [0, r].

    Parameters
    ----------
    r : float
        Range of the random variable (max – min). Must be > 0.
    delta : float
        Desired failure probability, 0 < delta < 1.
    n : int
        Number of independent observations (must be > 0).

    Returns
    -------
    float
        Upper confidence width ε such that
        P(|mean - true| > ε) ≤ delta.
    """
    if r <= 0:
        raise ValueError("Range r must be positive.")
    if not (0 < delta < 1):
        raise ValueError("Delta must be in (0,1).")
    if n <= 0:
        raise ValueError("Number of observations n must be positive.")
    return math.sqrt((r ** 2 * math.log(2 / delta)) / (2 * n))


# ----------------------------------------------------------------------
# Sheaf implementation (simplified from Parent B)
# ----------------------------------------------------------------------

class Sheaf:
    """A minimal sheaf where each node holds a vector (section)."""

    def __init__(self, node_dims: Dict[int, int], edges: List[Tuple[int, int]]):
        """
        Parameters
        ----------
        node_dims : dict[int, int]
            Mapping from node identifier to dimension of its section vector.
        edges : list[tuple[int, int]]
            List of undirected edges; only needed for completeness.
        """
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
        self._sections: Dict[int, np.ndarray] = {}

    def set_section(self, node: int, value: np.ndarray) -> None:
        """Store a section vector for a node."""
        dim_expected = self.node_dims.get(node)
        if dim_expected is None:
            raise KeyError(f"Node {node} not defined in sheaf.")
        if value.shape != (dim_expected,):
            raise ValueError(f"Section for node {node} must have shape ({dim_expected},).")
        self._sections[node] = value.astype(float)

    def get_section(self, node: int) -> np.ndarray:
        """Retrieve a stored section vector."""
        return self._sections[node]

    def all_sections(self) -> List[np.ndarray]:
        """Return a list of all stored section vectors."""
        return list(self._sections.values())


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------

def hybrid_compute_health_scores(endpoints: List[Endpoint]) -> None:
    """Populate the `health_score` field of each endpoint.

    The health score combines failure‑rate and recovery priority:
        h = (1 - failure_rate) * recovery_priority

    The function mutates the endpoint objects in‑place.
    """
    for ep in endpoints:
        ep.health_score = (1.0 - ep.failure_rate) * ep.recovery_priority


def hybrid_encode_hypervector(score: float, dim: int = 1024) -> np.ndarray:
    """Encode a scalar health score into a high‑dimensional hypervector.

    The encoding follows a simple random‑projection scheme:
        v_i = sign( g_i + α·score )
    where g_i ~ N(0,1) and α scales the influence of the score.

    The resulting vector contains values in {+1, -1}.
    """
    if dim <= 0:
        raise ValueError("Dimension must be positive.")
    alpha = 5.0  # scaling factor that makes the score visible in the random noise
    g = np.random.randn(dim)
    v = np.sign(g + alpha * score)
    # Replace zeros (unlikely) with +1 for a clean binary hypervector
    v[v == 0] = 1
    return v


def hybrid_encode_and_update_sheaf(
    endpoints: List[Endpoint],
    sheaf: Sheaf,
    dim: int = 1024,
) -> None:
    """Encode each endpoint's health score into a hypervector and store it
    as a sheaf section keyed by the endpoint id.

    Parameters
    ----------
    endpoints : list[Endpoint]
        List of endpoints with already computed `health_score`.
    sheaf : Sheaf
        Sheaf whose node identifiers correspond to endpoint ids.
    dim : int, optional
        Dimensionality of the hypervectors (default 1024).
    """
    for ep in endpoints:
        hv = hybrid_encode_hypervector(ep.health_score, dim=dim)
        sheaf.set_section(ep.id, hv)


def sheaf_shannon_entropy(sheaf: Sheaf) -> float:
    """Compute the Shannon entropy of the absolute values of all sheaf sections.

    The sections are concatenated, normalized to a probability distribution,
    and the standard entropy formula H = - Σ p_i log₂ p_i is applied.
    """
    sections = sheaf.all_sections()
    if not sections:
        return 0.0
    concatenated = np.concatenate([np.abs(sec) for sec in sections])
    total = concatenated.sum()
    if total == 0:
        return 0.0
    probs = concatenated / total
    # Avoid log2(0) by masking zero entries
    mask = probs > 0
    entropy = -np.sum(probs[mask] * np.log2(probs[mask]))
    return float(entropy)


def hybrid_fisher_information(hypervectors: List[np.ndarray]) -> float:
    """A lightweight Fisher‑information‑like scalar derived from hypervectors.

    For a set of binary hypervectors we use the empirical variance of each
    coordinate across the population; the sum of variances serves as a proxy
    for Fisher information (higher variance ⇒ more information about the
    underlying parameter).
    """
    if not hypervectors:
        return 0.0
    stacked = np.stack(hypervectors)  # shape (N, dim)
    variances = np.var(stacked, axis=0, ddof=1)  # unbiased variance per coordinate
    return float(variances.sum())


def hybrid_maybe_switch(
    endpoints: List[Endpoint],
    sheaf: Sheaf,
    delta: float = 0.05,
    min_observations: int = 30,
) -> Tuple[bool, int]:
    """Decide whether to switch to a better endpoint.

    The decision combines three ingredients:
        1. Hoeffding confidence width on the health scores.
        2. Fisher‑information proxy from the hypervectors stored in the sheaf.
        3. Shannon entropy of the sheaf sections.

    A switch is triggered if the best observed health score exceeds the
    current best by more than the Hoeffding width *and* the entropy is
    above a modest threshold (indicating sufficient uncertainty to explore).

    Returns
    -------
    switched : bool
        True if a switch should occur.
    new_best_id : int
        Identifier of the endpoint that would become the new active one
        (or the current best if no switch).
    """
    if not endpoints:
        raise ValueError("Endpoint list cannot be empty.")

    # 1. Hoeffding bound on health scores
    scores = np.array([ep.health_score for ep in endpoints])
    r = scores.max() - scores.min()
    eps = hoeffding_bound(r=r, delta=delta, n=min_observations)

    # Identify current best (the one with max score)
    current_best_idx = int(np.argmax(scores))
    current_best_score = scores[current_best_idx]

    # Potential better candidate (strictly greater than current best + eps)
    better_candidates = [
        (i, s) for i, s in enumerate(scores) if s > current_best_score + eps
    ]

    if not better_candidates:
        return False, endpoints[current_best_idx].id

    # 2. Fisher information from hypervectors
    hypervectors = sheaf.all_sections()
    fisher = hybrid_fisher_information(hypervectors)

    # 3. Entropy of the sheaf
    entropy = sheaf_shannon_entropy(sheaf)

    # Heuristic thresholds (tuned empirically)
    fisher_thresh = 0.1 * len(hypervectors) * hypervectors[0].size  # ~10% of max possible variance
    entropy_thresh = 0.5 * math.log2(hypervectors[0].size)          # half of max entropy per vector

    # Decide
    if fisher > fisher_thresh and entropy > entropy_thresh:
        # Choose the candidate with the highest health score
        new_idx, _ = max(better_candidates, key=lambda pair: pair[1])
        return True, endpoints[new_idx].id
    else:
        return False, endpoints[current_best_idx].id


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a synthetic set of endpoints
    random.seed(42)
    np.random.seed(42)

    endpoints = [
        Endpoint(id=i,
                 failure_rate=random.uniform(0.0, 0.3),
                 recovery_priority=random.uniform(0.5, 1.5))
        for i in range(5)
    ]

    # Step 1: compute health scores
    hybrid_compute_health_scores(endpoints)

    # Build a sheaf: one node per endpoint, dimension = 1024
    node_dims = {ep.id: 1024 for ep in endpoints}
    edges = []  # no edges needed for the demo
    sheaf = Sheaf(node_dims=node_dims, edges=edges)

    # Step 2: encode health scores and store them in the sheaf
    hybrid_encode_and_update_sheaf(endpoints, sheaf, dim=1024)

    # Step 3: compute entropy (just for demonstration)
    ent = sheaf_shannon_entropy(sheaf)
    print(f"Sheaf Shannon entropy: {ent:.4f} bits")

    # Step 4: decide whether to switch
    switched, new_id = hybrid_maybe_switch(endpoints, sheaf, delta=0.05, min_observations=30)
    if switched:
        print(f"Switching to endpoint {new_id} based on hybrid decision.")
    else:
        print(f"No switch needed; staying with current best endpoint {new_id}.")

    # Verify that all sections are correctly sized
    for ep in endpoints:
        sec = sheaf.get_section(ep.id)
        assert sec.shape == (1024,)
    print("Smoke test completed successfully.")