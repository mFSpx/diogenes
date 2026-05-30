# DARWIN HAMMER — match 4747, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m482_s0.py (gen5)
# parent_b: hybrid_hybrid_ternary_route_hybrid_hybrid_ternar_m439_s0.py (gen4)
# born: 2026-05-29T23:58:06Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Helper mathematics (parents)
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# 1. Health‑aware Gini impurity
# ----------------------------------------------------------------------
def health_weighted_gini(labels: List[int], morphologies: List[Morphology]) -> float:
    """
    Compute Gini impurity where each sample i is weighted by a health factor:
        w_i = mass_i / (length_i * width_i * height_i)
    The final impurity is Σ w_i * G_i / Σ w_i.
    """
    if len(labels) != len(morphologies):
        raise ValueError("labels and morphologies must have same length")
    # health weights, protect against zero volume
    weights = []
    for m in morphologies:
        vol = max(m.length * m.width * m.height, 1e-12)
        weights.append(m.mass / vol)
    total_w = sum(weights)
    if total_w == 0:
        return 0.0

    # class counts weighted
    class_weights: Dict[int, float] = {}
    for lbl, w in zip(labels, weights):
        class_weights[lbl] = class_weights.get(lbl, 0.0) + w

    # Gini = 1 - Σ (p_k)^2 where p_k = weighted proportion of class k
    gini = 1.0
    for w in class_weights.values():
        pk = w / total_w
        gini -= pk * pk
    return gini

# ----------------------------------------------------------------------
# 2. Hoeffding bound based split decision
# ----------------------------------------------------------------------
def hoeffding_split_decision(
    parent_labels: List[int],
    left_labels: List[int],
    right_labels: List[int],
    parent_morph: List[Morphology],
    left_morph: List[Morphology],
    right_morph: List[Morphology],
    delta: float = 1e-7,
    R: float = 1.0,
) -> Tuple[bool, float]:
    """
    Decide whether to split a node using the Hoeffding bound applied to the
    health‑weighted Gini impurity.

    Returns (split, epsilon) where split is True if the observed gain exceeds ε.
    """
    n = len(parent_labels)
    if n == 0:
        return False, 0.0

    g_parent = health_weighted_gini(parent_labels, parent_morph)
    g_left = health_weighted_gini(left_labels, left_morph)
    g_right = health_weighted_gini(right_labels, right_morph)

    # weighted gain (parent impurity minus weighted child impurity)
    n_left = len(left_labels)
    n_right = len(right_labels)
    gain = g_parent - (n_left / n) * g_left - (n_right / n) * g_right

    epsilon = math.sqrt((R * R * math.log(1.0 / delta)) / (2.0 * n))
    return gain > epsilon, epsilon

# ----------------------------------------------------------------------
# 3. Bayesian ternary routing with health‑aware Gini penalty
# ----------------------------------------------------------------------
def bayesian_ternary_route_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    edge_priors: Dict[Edge, float],
    evidence: Dict[Edge, Tuple[float, float]],
    path_weight: float = 0.2,
    leaf_partitions: List[Tuple[List[int], List[Morphology]]] = None,
) -> float:
    """
    Compute the cost of a ternary routing tree where edge costs are expectations
    under Bayesian‑updated priors.  An optional health‑aware Gini penalty is added
    based on leaf partitions (labels + morphologies).
    """
    # 1. Update priors with evidence (likelihood, false_positive)
    posteriors: Dict[Edge, float] = {}
    for e in edges:
        prior = edge_priors.get(e, 0.5)  # uniform default
        lik, fp = evidence.get(e, (0.5, 0.5))
        marginal = bayes_marginal(prior, lik, fp)
        posterior = bayes_update(prior, lik, marginal)
        posteriors[e] = posterior

    # 2. Expected edge length = length * posterior probability
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len = length(nodes[a], nodes[b])
        material += edge_len * posteriors[(a, b)]

    # 3. Compute distance from root to every node (expected path length)
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if nb not in dist:
                edge = (cur, nb) if (cur, nb) in posteriors else (nb, cur)
                exp_len = length(nodes[cur], nodes[nb]) * posteriors[edge]
                dist[nb] = dist[cur] + exp_len
                stack.append(nb)

    expected_path_cost = path_weight * sum(dist.values())

    # 4. Optional Gini penalty from leaf partitions
    gini_penalty = 0.0
    if leaf_partitions:
        penalties = []
        for labels, morphs in leaf_partitions:
            penalties.append(health_weighted_gini(labels, morphs))
        if penalties:
            gini_penalty = material * (sum(penalties) / len(penalties))

    return material + expected_path_cost + gini_penalty

def improved_hybrid_split_and_route(
    data: np.ndarray,
    labels: List[int],
    morphologies: List[Morphology],
    nodes: Dict[str, Point],
    edges: List[Edge],
    edge_priors: Dict[Edge, float],
    delta: float = 1e-7,
) -> Tuple[bool, float, float]:
    """
    Perform a single binary split on the data using health‑weighted Gini and
    Hoeffding bound, then evaluate the ternary routing cost of the resulting
    tree (with Bayesian edge priors).
    """
    n = len(labels)
    if n == 0:
        return False, 0.0, 0.0

    # Generate candidate splits
    candidate_splits = []
    for i in range(n - 1):
        left_labels = labels[:i + 1]
        right_labels = labels[i + 1:]
        left_morph = morphologies[:i + 1]
        right_morph = morphologies[i + 1:]

        # Compute weighted Gini gain
        split_decision, epsilon = hoeffding_split_decision(
            labels, left_labels, right_labels, morphologies, left_morph, right_morph, delta
        )
        if split_decision:
            candidate_splits.append((i, epsilon))

    # Evaluate ternary routing cost for each candidate split
    best_split = None
    best_cost = float('inf')
    for split_idx, epsilon in candidate_splits:
        # Create leaf partitions
        left_labels = labels[:split_idx + 1]
        right_labels = labels[split_idx + 1:]
        left_morph = morphologies[:split_idx + 1]
        right_morph = morphologies[split_idx + 1:]
        leaf_partitions = [(left_labels, left_morph), (right_labels, right_morph)]

        # Evaluate ternary routing cost
        cost = bayesian_ternary_route_cost(
            nodes, edges, list(nodes.keys())[0], edge_priors, {}, path_weight=0.2, leaf_partitions=leaf_partitions
        )
        if cost < best_cost:
            best_cost = cost
            best_split = split_idx

    if best_split is not None:
        # Compute weighted Gini gain for best split
        best_split_decision, best_epsilon = hoeffding_split_decision(
            labels, labels[:best_split + 1], labels[best_split + 1:], morphologies, morphologies[:best_split + 1], morphologies[best_split + 1:]
        )
        return best_split_decision, best_epsilon, best_cost
    else:
        return False, 0.0, 0.0