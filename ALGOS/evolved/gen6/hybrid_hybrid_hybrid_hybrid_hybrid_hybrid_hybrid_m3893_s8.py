# DARWIN HAMMER — match 3893, survivor 8
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m2411_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1830_s0.py (gen4)
# born: 2026-05-29T23:52:28Z

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from dataclasses import dataclass, field

def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

class Edge:
    def __init__(self, node1: str, node2: str):
        self.node1 = node1
        self.node2 = node2

    def __eq__(self, other):
        return (self.node1 == other.node1 and self.node2 == other.node2) or (
            self.node1 == other.node2 and self.node2 == other.node1
        )

    def __hash__(self):
        return hash((self.node1, self.node2))

    def __repr__(self):
        return f"Edge({self.node1!r}, {self.node2!r})"

def edge_feature(edge: Edge) -> List[float]:
    return [ord(c) for c in edge.node1 + edge.node2]

class RBFSurrogate:
    def __init__(self, centers: List[List[float]], weights: List[float], epsilon: float = 1.0):
        self.centers = centers
        self.weights = weights
        self.epsilon = epsilon

    def predict(self, x: List[float]) -> float:
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon) for w, c in zip(self.weights, self.centers)
        )

def fit_rbf(points: List[List[float]], values: List[float], epsilon: float = 1.0, ridge: float = 1e-9) -> RBFSurrogate:
    A = np.array(
        [[gaussian(euclidean(p1, p2), epsilon) for p2 in points] for p1 in points],
        dtype=float,
    )
    A += ridge * np.eye(len(points))
    weights = np.linalg.solve(A, np.array(values, dtype=float))
    return RBFSurrogate(points, weights.tolist(), epsilon)

def extract_node_priors(motifs: List[List[Edge]], epsilon: float = 1.0) -> Dict[Edge, float]:
    node_priors: Dict[Edge, float] = {}
    for motif in motifs:
        for edge in motif:
            score = 0.0
            for other_edge in motif:
                if edge != other_edge:
                    score += gaussian(
                        euclidean(
                            edge_feature(edge),
                            edge_feature(other_edge),
                        ),
                        epsilon,
                    )
            node_priors[edge] = score / len(motif) if motif else 0.0
    return node_priors

def normalize_distribution(values: List[float]) -> List[float]:
    total = sum(values)
    return [v / total for v in values] if total > 0 else [0.0 for _ in values]

def gini_coefficient(probs: List[float]) -> float:
    if not probs:
        return 0.0
    sorted_p = sorted(probs)
    n = len(probs)
    cumulative = sum((i + 1) * p for i, p in enumerate(sorted_p))
    return 1.0 - 2.0 * cumulative / (n * sum(sorted_p))

def compute_regret_weighted_distribution(
    priors: Dict[Edge, float],
    costs: Dict[Edge, float],
) -> Dict[Edge, float]:
    raw = {e: max(p - costs.get(e, 0.0), 0.0) for e, p in priors.items()}
    total = sum(raw.values())
    if total == 0:
        uniform = 1.0 / len(raw) if raw else 0.0
        return {e: uniform for e in raw}
    return {e: v / total for e, v in raw.items()}

def koopman_forecast(feature: List[float], K: np.ndarray) -> np.ndarray:
    vec = np.array(feature, dtype=float)
    return K @ vec

def hybrid_fuse(
    motifs: List[List[Edge]],
    epsilon: float = 1.0,
    eta: float = 0.5,
    K: np.ndarray = None,
) -> Dict[Edge, float]:
    node_priors = extract_node_priors(motifs, epsilon)
    costs = {e: random.random() for e in node_priors}
    regret_dist = compute_regret_weighted_distribution(node_priors, costs)
    G = gini_coefficient(list(regret_dist.values()))

    edges = list(node_priors.keys())
    raw_features = [edge_feature(e) for e in edges]

    dim = len(raw_features[0])
    if K is None:
        K = np.eye(dim)
    koopman_features = [koopman_forecast(f, K).tolist() for f in raw_features]

    scaled_eps = epsilon * (1.0 + G)
    rbf_model = fit_rbf(koopman_features, [node_priors[e] for e in edges], epsilon=scaled_eps)

    fused_scores: Dict[Edge, float] = {}
    for edge in edges:
        koopman_feature = koopman_forecast(edge_feature(edge), K)
        prior_score = node_priors.get(edge, 0.0)
        rbf_score = rbf_model.predict(koopman_feature.tolist())
        exploration_score = eta * np.linalg.norm(koopman_feature) * G
        fused_scores[edge] = prior_score * rbf_score + exploration_score

    return fused_scores

def robust_hybrid_fuse(
    motifs: List[List[Edge]],
    epsilon: float = 1.0,
    eta: float = 0.5,
    K: np.ndarray = None,
    alpha: float = 0.1,
) -> Dict[Edge, float]:
    node_priors = extract_node_priors(motifs, epsilon)
    costs = {e: random.random() for e in node_priors}
    regret_dist = compute_regret_weighted_distribution(node_priors, costs)
    G = gini_coefficient(list(regret_dist.values()))

    edges = list(node_priors.keys())
    raw_features = [edge_feature(e) for e in edges]

    dim = len(raw_features[0])
    if K is None:
        K = np.eye(dim)
    koopman_features = [koopman_forecast(f, K).tolist() for f in raw_features]

    scaled_eps = epsilon * (1.0 + G)
    rbf_model = fit_rbf(koopman_features, [node_priors[e] for e in edges], epsilon=scaled_eps)

    fused_scores: Dict[Edge, float] = {}
    for edge in edges:
        koopman_feature = koopman_forecast(edge_feature(edge), K)
        prior_score = node_priors.get(edge, 0.0)
        rbf_score = rbf_model.predict(koopman_feature.tolist())
        exploration_score = eta * np.linalg.norm(koopman_feature) * G
        robust_score = alpha * prior_score + (1 - alpha) * rbf_score
        fused_scores[edge] = robust_score + exploration_score

    return fused_scores