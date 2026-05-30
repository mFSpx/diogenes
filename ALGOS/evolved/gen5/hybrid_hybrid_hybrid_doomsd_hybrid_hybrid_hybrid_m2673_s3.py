# DARWIN HAMMER — match 2673, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_bayes_claim_k_m455_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m900_s2.py (gen4)
# born: 2026-05-29T23:43:35Z

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
        raise ValueError(f"Root node '{root}' not found in nodes")

    num_nodes = len(nodes)
    if len(node_weights) != num_nodes:
        raise ValueError(f"Node weights must have length {num_nodes}")

    # Create a list of edges
    edges = []
    for i, node_i in enumerate(nodes):
        for node_j in nodes:
            if node_i != node_j:
                j = list(nodes.keys()).index(node_j)
                weight_i = node_weights[i]
                weight_j = node_weights[j]
                edge_cost = euclidean_length(nodes[node_i], nodes[node_j]) * (weight_i + weight_j) / 2
                edges.append((node_i, node_j, edge_cost))

    # Sort edges by cost
    edges.sort(key=lambda x: x[2])

    # Initialize the MST and its cost
    mst = []
    total_cost = 0.0
    visited = set()

    # Start with the root node
    visited.add(root)

    # Iterate over the sorted edges
    for edge in edges:
        node_i, node_j, edge_cost = edge
        if node_i in visited and node_j not in visited:
            mst.append(edge)
            total_cost += edge_cost
            visited.add(node_j)
        elif node_j in visited and node_i not in visited:
            mst.append(edge)
            total_cost += edge_cost
            visited.add(node_i)

    return total_cost


def improved_hybrid_algorithm(start_year: int, end_year: int, nodes: Dict[str, Point], root: str) -> float:
    """
    Improved hybrid algorithm that integrates the mathematical systems more deeply.

    Parameters
    ----------
    start_year : int
    end_year : int
    nodes : dict mapping node identifier → (x, y) coordinate
    root : identifier of the starting node (must exist in `nodes`)

    Returns
    -------
    total_cost : float
    """
    # Generate weekday frequencies
    weekday_freqs = weekday_frequencies(start_year, end_year)

    # Generate synthetic pheromone evidence
    evidence = np.random.rand(7)

    # Perform Bayesian update
    posterior = hybrid_bayesian_update(weekday_freqs, evidence)

    # Compute dispersion metric
    dispersion = dispersion_metric(weekday_freqs, evidence)

    # Compute the total cost of the MST
    total_cost = minimum_cost_spanning_tree(nodes, root, posterior)

    return total_cost