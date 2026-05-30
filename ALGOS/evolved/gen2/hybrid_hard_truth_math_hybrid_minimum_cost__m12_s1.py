# DARWIN HAMMER — match 12, survivor 1
# gen: 2
# parent_a: hard_truth_math.py (gen0)
# parent_b: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# born: 2026-05-29T23:25:17Z

#!/usr/bin/env python3
"""Hybrid module combining Hard-truth Math (Algorithm A) and Hybrid Minimum Cost Tree Bayes Update (Algorithm B).

Mathematical bridge: 
The governing equations of both parents are integrated by using the expected value of the edge lengths from Algorithm B to weight the stylometry features in Algorithm A.
This allows for a probabilistic transformation of the stylometry features, enabling the hybrid to adapt to different writing styles and contexts.

The hybrid replaces the deterministic stylometry features in Algorithm A with their expected values under the posterior edge belief obtained from Algorithm B.
Similarly, the node distances are weighted by a node belief derived from incident edge posteriors.
The resulting hybrid cost is a combination of the expected stylometry features and the weighted node distances.

The module implements:
* `hybrid_lsm_vector` – computes the expected stylometry features using the posterior edge beliefs.
* `hybrid_lsm_score` – evaluates the similarity between two texts using the expected stylometry features.
* `hybrid_tree_cost` – computes the hybrid cost using the expected stylometry features and weighted node distances.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import random
from collections import Counter
import re

# Types
Point = Tuple[float, float]
Edge = Tuple[str, str]

# Algorithm A – stylometry features
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def lsm_vector(text: str) -> dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {cat: sum(cnt[w] for w in vocab) / total for cat, vocab in FUNCTION_CATS.items()}

# Algorithm B – Minimum Cost Tree and Bayes Update
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
    edge_len : dict mapping edge (ordered a
    """
    adj = {node: [] for node in nodes}
    edge_len = {}
    node_dist = {node: 0.0 for node in nodes}

    for edge in edges:
        adj[edge[0]].append(edge[1])
        adj[edge[1]].append(edge[0])
        edge_len[edge] = length(nodes[edge[0]], nodes[edge[1]])

    # Compute root-to-node distances using BFS
    queue = [root]
    visited = {root}
    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if neighbor not in visited:
                node_dist[neighbor] = node_dist[node] + edge_len[(node, neighbor)]
                queue.append(neighbor)
                visited.add(neighbor)

    return adj, edge_len, node_dist

def bayes_edge_posteriors(
    edge_len: Dict[Edge, float],
    prior: float,
    likelihood: float,
    false_positive_rate: float,
) -> Dict[Edge, float]:
    """
    Compute posterior edge beliefs using Bayes' rule.

    Returns
    -------
    edge_posteriors : dict mapping edge → posterior belief
    """
    edge_posteriors = {}
    for edge, length in edge_len.items():
        posterior = (prior * likelihood) / (likelihood * prior + false_positive_rate * (1 - prior))
        edge_posteriors[edge] = posterior
    return edge_posteriors

def hybrid_lsm_vector(
    text: str,
    edge_posteriors: Dict[Edge, float],
) -> dict[str, float]:
    """
    Compute expected stylometry features using posterior edge beliefs.

    Returns
    -------
    expected_features : dict mapping feature → expected value
    """
    features = lsm_vector(text)
    expected_features = {}
    for feature, value in features.items():
        expected_value = sum(edge_posteriors[edge] * value for edge in edge_posteriors)
        expected_features[feature] = expected_value
    return expected_features

def hybrid_lsm_score(
    text_a: str,
    text_b: str,
    edge_posteriors: Dict[Edge, float],
) -> tuple[float, dict[str, float]]:
    """
    Compute similarity between two texts using expected stylometry features.

    Returns
    -------
    similarity : float
    feature_similarity : dict mapping feature → similarity
    """
    features_a = hybrid_lsm_vector(text_a, edge_posteriors)
    features_b = hybrid_lsm_vector(text_b, edge_posteriors)
    similarity = 0.0
    feature_similarity = {}
    for feature in features_a:
        value_a = features_a[feature]
        value_b = features_b[feature]
        score = 1.0 - (abs(value_a - value_b) / (value_a + value_b + 1e-6))
        score = max(0.0, min(1.0, score))
        similarity += score
        feature_similarity[feature] = score
    return similarity, feature_similarity

def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    prior: float,
    likelihood: float,
    false_positive_rate: float,
) -> float:
    """
    Compute hybrid cost using expected stylometry features and weighted node distances.

    Returns
    -------
    hybrid_cost : float
    """
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    edge_posteriors = bayes_edge_posteriors(edge_len, prior, likelihood, false_positive_rate)
    hybrid_cost = 0.0
    for edge, posterior in edge_posteriors.items():
        hybrid_cost += posterior * edge_len[edge]
    for node, distance in node_dist.items():
        hybrid_cost += distance * sum(edge_posteriors[edge] for edge in edge_posteriors if node in edge)
    return hybrid_cost

if __name__ == "__main__":
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 1.0),
        "C": (2.0, 2.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    text_a = "This is a test text."
    text_b = "This is another test text."
    prior = 0.5
    likelihood = 0.8
    false_positive_rate = 0.1

    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    edge_posteriors = bayes_edge_posteriors(edge_len, prior, likelihood, false_positive_rate)
    expected_features_a = hybrid_lsm_vector(text_a, edge_posteriors)
    expected_features_b = hybrid_lsm_vector(text_b, edge_posteriors)
    similarity, feature_similarity = hybrid_lsm_score(text_a, text_b, edge_posteriors)
    hybrid_cost = hybrid_tree_cost(nodes, edges, root, prior, likelihood, false_positive_rate)

    print("Expected features A:", expected_features_a)
    print("Expected features B:", expected_features_b)
    print("Similarity:", similarity)
    print("Feature similarity:", feature_similarity)
    print("Hybrid cost:", hybrid_cost)