# DARWIN HAMMER — match 2673, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_bayes_claim_k_m455_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m900_s2.py (gen4)
# born: 2026-05-29T23:43:35Z

"""
Hybrid Algorithm: hybrid_doomsday_pheromone_entropy_tree.py

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – `hybrid_hybrid_doomsday_cale_hybrid_bayes_claim_k_m455_s0.py`  
  Provides the Doomsday calendar generation, Gini‑coefficient inequality measure,
  Euclidean distance between points and a minimum‑cost tree evaluation.

* **Parent B** – `hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m900_s2.py`  
  Supplies pheromone‑style signal recording, Shannon entropy computation and a
  mapping `signal_value = exp(-H)` that turns uncertainty into a quantitative
  weight.

**Mathematical Bridge**  
The bridge is built on the observation that both parents manipulate *probability‑like*
quantities over a discrete set (weekdays in A, categorical pheromone signals in B).
We therefore:

1. Generate a weekday frequency distribution with the Doomsday calendar (A).  
2. Treat the normalized frequencies as a **prior** distribution over the seven weekdays.  
3. Record synthetic “pheromone” evidence for each weekday, compute its Shannon entropy,
   and map the entropy to a likelihood via `L_i = exp(-H_i) * e_i` (B).  
4. Perform a Bayesian update `posterior ∝ prior × likelihood`.  
5. Combine the Gini coefficient of the original frequencies with the normalized
   entropy of the evidence to obtain a single *dispersion metric* that quantifies
   both inequality and uncertainty.  
6. Use the posterior probabilities as node weights in a Euclidean minimum‑cost
   spanning‑tree (MST) calculation, yielding a final expected routing cost.

The resulting system simultaneously evaluates inequality (Gini), uncertainty
(Shannon entropy), Bayesian evidence integration, and graph‑theoretic cost.
"""

from __future__ import annotations

import math
import random
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Parent A – Doomsday & Gini utilities
# ----------------------------------------------------------------------
def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index (0=Monday … 6=Sunday) using Doomsday algorithm."""
    return (date(year, month, day).weekday() + 1) % 7


def gini_coefficient(values: Iterable[float]) -> float:
    """Standard Gini coefficient for a non‑negative vector."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def euclidean_length(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Parent B – Entropy & Pheromone utilities
# ----------------------------------------------------------------------
def shannon_entropy(probs: Iterable[float]) -> float:
    """Compute Shannon entropy (base e) of a probability distribution."""
    probs = np.array(list(probs), dtype=float)
    probs = probs[probs > 0]  # ignore zero probabilities
    return -float(np.sum(probs * np.log(probs)))


def pheromone_likelihood(evidence: Iterable[float]) -> np.ndarray:
    """
    Transform raw pheromone evidence into a likelihood vector.
    The bridge formula from Parent B is:
        L_i = exp(-H) * e_i
    where H is the entropy of the normalized evidence.
    """
    ev = np.array(list(evidence), dtype=float)
    if ev.size == 0:
        return np.ones_like(ev)
    # Normalise to obtain a probability‑like vector for entropy
    prob = ev / ev.sum() if ev.sum() != 0 else np.full_like(ev, 1 / ev.size)
    H = shannon_entropy(prob)
    likelihood = np.exp(-H) * ev
    # Re‑normalise so that likelihoods sum to 1 (required for Bayesian update)
    total = likelihood.sum()
    return likelihood / total if total != 0 else np.full_like(likelihood, 1 / likelihood.size)


# ----------------------------------------------------------------------
# Hybrid Core Functions (demonstrating the fusion)
# ----------------------------------------------------------------------
def weekday_frequencies(start_year: int, end_year: int) -> np.ndarray:
    """
    Generate a frequency count of weekdays for every day in the inclusive
    interval [start_year, end_year] using the Doomsday calendar.
    Returns a length‑7 ndarray ordered Monday→Sunday.
    """
    counts = np.zeros(7, dtype=int)
    for yr in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Determine days in month (simple Gregorian calendar, ignoring
            # exotic calendar reforms – sufficient for the demo)
            if month == 2:
                is_leap = (yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0))
                days = 29 if is_leap else 28
            elif month in (4, 6, 9, 11):
                days = 30
            else:
                days = 31
            for day in range(1, days + 1):
                wd = doomsday(yr, month, day)
                counts[wd] += 1
    return counts


def hybrid_bayesian_update(prior: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    """
    Perform a Bayesian update where `prior` is the weekday prior (from frequencies)
    and `evidence` are raw pheromone signals.  The evidence is first turned into a
    likelihood via the entropy‑based mapping (Parent B) and then combined with the
    prior (Parent A).
    """
    if prior.shape != evidence.shape:
        raise ValueError("Prior and evidence must have the same shape.")
    # Normalise prior to a probability distribution
    prior_prob = prior / prior.sum() if prior.sum() != 0 else np.full_like(prior, 1 / prior.size)
    # Convert raw evidence into a likelihood vector
    likelihood = pheromone_likelihood(evidence)
    # Bayesian rule (unnormalised posterior)
    unnorm_posterior = prior_prob * likelihood
    # Normalise to obtain posterior probabilities
    posterior = unnorm_posterior / unnorm_posterior.sum() if unnorm_posterior.sum() != 0 else np.full_like(unnorm_posterior, 1 / unnorm_posterior.size)
    return posterior


def dispersion_metric(freqs: np.ndarray, evidence: np.ndarray, alpha: float = 0.5) -> float:
    """
    Compute a unified dispersion metric that blends:
      * Gini coefficient of the raw weekday frequencies (inequality)
      * Normalised Shannon entropy of the pheromone evidence (uncertainty)

    The blending factor `alpha` weights Gini vs entropy (0 ≤ α ≤ 1).
    """
    gini = gini_coefficient(freqs)
    # Normalised entropy: divide by log(k) where k is number of categories
    prob_evidence = evidence / evidence.sum() if evidence.sum() != 0 else np.full_like(evidence, 1 / evidence.size)
    entropy = shannon_entropy(prob_evidence)
    max_entropy = math.log(evidence.size) if evidence.size > 1 else 1.0
    norm_entropy = entropy / max_entropy
    return alpha * gini + (1 - alpha) * norm_entropy


def minimum_cost_spanning_tree(
    nodes: Dict[str, Point],
    root: str,
    node_weights: np.ndarray,
) -> float:
    """
    Compute the total cost of a Minimum‑Cost Spanning Tree (MST) where each edge
    cost is the Euclidean length multiplied by the average of the two incident
    node weights.  The algorithm uses Prim's method.

    Parameters
    ----------
    nodes : dict mapping node identifier → (x, y) coordinate
    root : identifier of the starting node (must exist in `nodes`)
    node_weights : ndarray of length len(nodes); order follows `list(nodes.keys())`

    Returns
    -------
    total_cost : float
    """
    if root not in nodes:
        raise ValueError("Root node not found in nodes dictionary.")
    if len(nodes) != len(node_weights):
        raise ValueError("Length of node_weights must match number of nodes.")

    node_ids = list(nodes.keys())
    idx_map = {nid: i for i, nid in enumerate(node_ids)}
    visited = set()
    visited.add(root)
    total_cost = 0.0

    # Pre‑compute distance matrix
    n = len(node_ids)
    dist = np.zeros((n, n), dtype=float)
    for i, a in enumerate(node_ids):
        for j, b in enumerate(node_ids):
            if i == j:
                continue
            dist[i, j] = euclidean_length(nodes[a], nodes[b])

    # Prim's algorithm
    while len(visited) < n:
        best_edge = None
        best_cost = math.inf
        for u in visited:
            ui = idx_map[u]
            for v in node_ids:
                if v in visited:
                    continue
                vi = idx_map[v]
                edge_len = dist[ui, vi]
                weight_factor = (node_weights[ui] + node_weights[vi]) / 2.0
                edge_cost = edge_len * weight_factor
                if edge_cost < best_cost:
                    best_cost = edge_cost
                    best_edge = v
        if best_edge is None:
            break  # disconnected graph (should not happen with complete graph)
        total_cost += best_cost
        visited.add(best_edge)

    return total_cost


# ----------------------------------------------------------------------
# Helper utilities (borrowed from Parent B)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Compact binary fingerprint of a list of floats (max 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer bit‑patterns."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Generate weekday frequencies for a modest interval
    freqs = weekday_frequencies(2024, 2024)          # one year
    print("Weekday frequencies:", freqs)

    # 2. Synthesize pheromone evidence (e.g., surface usage counts) for each weekday
    random.seed(42)
    evidence = np.array([random.randint(1, 100) for _ in range(7)], dtype=float)
    print("Raw pheromone evidence:", evidence)

    # 3. Hybrid Bayesian update
    posterior = hybrid_bayesian_update(freqs, evidence)
    print("Posterior probabilities:", posterior)

    # 4. Unified dispersion metric
    metric = dispersion_metric(freqs, evidence, alpha=0.6)
    print("Dispersion metric (blend Gini/entropy):", metric)

    # 5. Build a small geometric graph (coordinates placed on a circle)
    theta = np.linspace(0, 2 * math.pi, 7, endpoint=False)
    nodes = {f"W{d}": (math.cos(t), math.sin(t)) for d, t in enumerate(theta)}
    root_node = "W0"

    # 6. Compute MST cost weighted by posterior probabilities
    mst_cost = minimum_cost_spanning_tree(nodes, root_node, posterior)
    print("Weighted MST cost:", mst_cost)

    # 7. Demonstrate auxiliary functions
    phash = compute_phash(evidence.tolist())
    print("Phash of evidence:", phash)
    # Hamming distance between phash and its bit‑reversed version
    rev = int('{:064b}'.format(phash)[::-1], 2)
    print("Hamming distance to reversed phash:", hamming_distance(phash, rev))

    sys.exit(0)