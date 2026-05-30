# DARWIN HAMMER — match 4747, survivor 6
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

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def health_weighted_gini(labels: List[int], morphologies: List[Morphology]) -> float:
    if len(labels) != len(morphologies):
        raise ValueError("labels and morphologies must have same length")
    weights = []
    for m in morphologies:
        vol = max(m.length * m.width * m.height, 1e-12)
        weights.append(m.mass / vol)
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    class_weights: Dict[int, float] = {}
    for lbl, w in zip(labels, weights):
        class_weights[lbl] = class_weights.get(lbl, 0.0) + w
    gini = 1.0
    for w in class_weights.values():
        pk = w / total_w
        gini -= pk * pk
    return gini

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
    n = len(parent_labels)
    if n == 0:
        return False, 0.0
    g_parent = health_weighted_gini(parent_labels, parent_morph)
    g_left = health_weighted_gini(left_labels, left_morph)
    g_right = health_weighted_gini(right_labels, right_morph)
    n_left = len(left_labels)
    n_right = len(right_labels)
    gain = g_parent - (n_left / n) * g_left - (n_right / n) * g_right
    epsilon = math.sqrt((R * R * math.log(1.0 / delta)) / (2.0 * n))
    return gain > epsilon, epsilon

def bayesian_ternary_route_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    edge_priors: Dict[Edge, float],
    evidence: Dict[Edge, Tuple[float, float]],
    path_weight: float = 0.2,
    leaf_partitions: List[Tuple[List[int], List[Morphology]]] = None,
) -> float:
    posteriors: Dict[Edge, float] = {}
    for e in edges:
        prior = edge_priors.get(e, 0.5)  
        lik, fp = evidence.get(e, (0.5, 0.5))
        marginal = bayes_marginal(prior, lik, fp)
        posterior = bayes_update(prior, lik, marginal)
        posteriors[e] = posterior
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len = length(nodes[a], nodes[b])
        material += edge_len * posteriors[(a, b)]
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
    gini_penalty = 0.0
    if leaf_partitions:
        penalties = []
        for labels, morphs in leaf_partitions:
            penalties.append(health_weighted_gini(labels, morphs))
        if penalties:
            gini_penalty = material * (sum(penalties) / len(penalties))
    return material + expected_path_cost + gini_penalty

def hybrid_split_and_route(
    data: np.ndarray,
    labels: List[int],
    morphologies: List[Morphology],
    nodes: Dict[str, Point],
    edges: List[Edge],
    edge_priors: Dict[Edge, float],
    delta: float = 1e-7,
) -> Tuple[bool, float, float]:
    n = len(labels)
    if n == 0:
        return False, 0.0, 0.0
    split, epsilon = hoeffding_split_decision(labels, [], [], morphologies, [], [], delta=delta)
    if not split:
        return False, 0.0, 0.0
    cost = bayesian_ternary_route_cost(nodes, edges, 'root', edge_priors, {}, path_weight=0.2)
    return True, epsilon, cost