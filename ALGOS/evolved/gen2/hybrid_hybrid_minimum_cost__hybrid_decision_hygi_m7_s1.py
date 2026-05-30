# DARWIN HAMMER — match 7, survivor 1
# gen: 2
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# parent_b: hybrid_decision_hygiene_shannon_entropy_m12_s1.py (gen1)
# born: 2026-05-29T23:22:20Z

"""
This module integrates the hybrid_minimum_cost_tree_bayes_update and hybrid_decision_hygiene_shannon_entropy algorithms into a single hybrid system.
The bridge between the two structures is the concept of expected cost and Shannon entropy, which can be applied to the decision hygiene scoring system.
By calculating the expected cost of a decision tree and the Shannon entropy of the decision hygiene feature counts, we can gain insights into the complexity and uncertainty of the decision-making process.
The mathematical bridge is formed by using the expected cost as a weight for the decision hygiene scores, and then calculating the Shannon entropy of the weighted scores.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import random

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Algorithm A – deterministic tree utilities
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Edge, float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS/DFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                # identify the edge direction used for length lookup
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)

    return adj, edge_len, dist


# ----------------------------------------------------------------------
# Algorithm B – Bayesian primitives (vectorised)
# ----------------------------------------------------------------------
def bayes_marginal(prior: np.ndarray, likelihood: np.ndarray, false_positive: float) -> np.ndarray:
    """
    Vectorised marginal:  P(E) = L·p + FP·(1‑p)
    """
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: np.ndarray, likelihood: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    """
    Vectorised posterior:  P(H|E) = p·L / P(E)
    """
    if np.any(marginal <= 0):
        raise ValueError("All marginal probabilities must be > 0")
    return prior * likelihood / marginal


def bayes_edge_posteriors(
    prior_dict: Dict[Edge, float],
    likelihood_dict: Dict[Edge, float],
    false_positive: float,
) -> Dict[Edge, float]:
    """
    Compute posterior probability for each edge using Bayesian update (Eq. 2).

    Parameters
    ----------
    prior_dict, likelihood_dict : dict mapping Edge → probability in [0,1]
    false_positive : scalar false‑positive rate (FP) in [0,1]

    Returns
    -------
    posterior_dict : dict mapping Edge → posterior probability
    """
    edges = list(prior_dict.keys())
    priors = np.array([prior_dict[e] for e in edges], dtype=float)
    likes = np.array([likelihood_dict[e] for e in edges], dtype=float)

    marginal = bayes_marginal(priors, likes, false_positive)
    post = bayes_update(priors, likes, marginal)

    return {e: float(p) for e, p in zip(edges, post)}


# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    path_weight: float,
    edge_priors: Dict[Edge, float],
    edge_likelihoods: Dict[Edge, float],
    false_positive: float,
) -> float:
    """
    Compute the hybrid expected cost.

    Steps
    -----
    1. Obtain deterministic geometry via `tree_metrics`.
    2. Compute posterior edge probabilities via `bayes_edge_posteriors`.
    3. Derive a node belief *q_v* as the complement of the probability that
       **none** of its incident edges are present:
           q_v = 1 - ∏_{e∈inc(v)} (1 - p_e)
    4. Assemble the expected material and path components.

    Returns
    -------
    Expected hybrid cost (float).
    """
    # 1. Geometry
    adj, edge_len, dist = tree_metrics(nodes, edges, root)

    # 2. Bayesian edge posteriors
    posteriors = bayes_edge_posteriors(edge_priors, edge_likelihoods, false_positive)

    # 3. Node beliefs
    node_beliefs = {}
    for node in nodes:
        incident_edges = [(node, neighbor) for neighbor in adj[node]] + [(neighbor, node) for neighbor in adj[node]]
        prob_none_present = 1
        for edge in incident_edges:
            if edge in posteriors:
                prob_none_present *= (1 - posteriors[edge])
        node_beliefs[node] = 1 - prob_none_present

    # 4. Expected cost
    expected_cost = 0
    for edge in edges:
        if edge in posteriors:
            expected_cost += posteriors[edge] * edge_len[edge]
    for node in nodes:
        expected_cost += path_weight * node_beliefs[node] * dist[node]

    return expected_cost


def shannon_entropy(observations: List[int | float], is_distribution: bool = False) -> float:
    """
    Compute the Shannon entropy of a list of observations.

    Parameters
    ----------
    observations : list of int or float
    is_distribution : bool, optional (default=False)

    Returns
    -------
    Shannon entropy (float)
    """
    xs = list(observations)
    if not xs:
        return 0.0
    if is_distribution:
        probs = [float(x) for x in xs]
        if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError("distribution must sum to 1")
    else:
        c = {x: observations.count(x) for x in observations}
        total = sum(c.values())
        probs = [v / total for v in c.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def hybrid_score(text: str, nodes: Dict[str, Point], edges: List[Edge], root: str, path_weight: float, edge_priors: Dict[Edge, float], edge_likelihoods: Dict[Edge, float], false_positive: float) -> Tuple[float, float]:
    """
    Compute the hybrid score, which is the product of the expected cost and the Shannon entropy.

    Parameters
    ----------
    text : str
    nodes : dict mapping node → point
    edges : list of edges
    root : str
    path_weight : float
    edge_priors : dict mapping edge → prior probability
    edge_likelihoods : dict mapping edge → likelihood
    false_positive : float

    Returns
    -------
    Hybrid score (tuple of float)
    """
    expected_cost = hybrid_tree_cost(nodes, edges, root, path_weight, edge_priors, edge_likelihoods, false_positive)
    feature_counts = {
        "evidence_count": text.count("evidence"),
        "planning_count": text.count("planning"),
        "delay_count": text.count("delay"),
        "support_count": text.count("support"),
        "boundary_count": text.count("boundary"),
        "outcome_count": text.count("outcome"),
        "impulsive_count": text.count("impulsive"),
        "scarcity_count": text.count("scarcity"),
        "risk_count": text.count("risk"),
    }
    shannon_entropy_score = shannon_entropy(list(feature_counts.values()))
    return expected_cost * shannon_entropy_score, shannon_entropy_score


if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 0), "C": (1, 1)}
    edges = [("A", "B"), ("B", "C")]
    root = "A"
    path_weight = 1.0
    edge_priors = {("A", "B"): 0.5, ("B", "C"): 0.5}
    edge_likelihoods = {("A", "B"): 0.5, ("B", "C"): 0.5}
    false_positive = 0.1
    text = "This is a test text with some evidence and planning."
    score, entropy = hybrid_score(text, nodes, edges, root, path_weight, edge_priors, edge_likelihoods, false_positive)
    print("Hybrid score:", score)
    print("Shannon entropy:", entropy)