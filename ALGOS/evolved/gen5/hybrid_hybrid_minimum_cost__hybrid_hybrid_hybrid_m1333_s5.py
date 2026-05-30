# DARWIN HAMMER — match 1333, survivor 5
# gen: 5
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_indy_learning_m816_s1.py (gen4)
# born: 2026-05-29T23:35:27Z

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

Point = Tuple[float, float]
Edge = Tuple[str, str]          
Classification = str

def euclidean_length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, false_positive: float) -> float:
    marginal = bayes_marginal(prior, likelihood, false_positive)
    if marginal == 0.0:
        raise ValueError("Marginal probability is zero; cannot divide.")
    return prior * likelihood / marginal

def update_edge_priors(
    edge_priors: Dict[Edge, float],
    evidence: Dict[Edge, Tuple[float, float]],
) -> Dict[Edge, float]:
    updated = {}
    for e, prior in edge_priors.items():
        if e not in evidence:
            updated[e] = prior
            continue
        likelihood, false_positive = evidence[e]
        updated[e] = bayes_update(prior, likelihood, false_positive)
    return updated

CLASSIFICATION_MAP = {
    "usable_now": 1,
    "research_only": 0,
    "needs_conversion": -1,
    "unsafe_for_fastpath": -1,
    "unsupported": -1,
}

def ternary_vector(classif: Classification) -> Tuple[int, int, int]:
    val = CLASSIFICATION_MAP.get(classif, -1)
    return (val, val, val)

def build_audit_matrix(node_classes: Dict[str, Classification]) -> np.ndarray:
    rows = [ternary_vector(cls) for cls in node_classes.values()]
    return np.array(rows, dtype=int)

def ontology_frequency_vector(texts: List[str], vocabulary: List[str]) -> np.ndarray:
    vocab_index = {term: i for i, term in enumerate(vocabulary)}
    vec = np.zeros(len(vocabulary), dtype=float)
    for txt in texts:
        for token in txt.split():
            if token in vocab_index:
                vec[vocab_index[token]] += 1.0
    return vec

def normalised_weight_vector(freq: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(freq)
    return freq / norm if norm > 0 else freq

def path_signature(audit_matrix: np.ndarray) -> np.ndarray:
    return np.prod(audit_matrix, axis=0)

def node_score(
    audit_vec: np.ndarray,
    signature: np.ndarray,
    ontology_vec: np.ndarray,
    term_weights: np.ndarray,
) -> float:
    audit_part = float(np.dot(audit_vec, signature))
    ontology_part = float(np.dot(ontology_vec, term_weights))
    return audit_part * ontology_part

def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    node_classes: Dict[str, Classification],
    node_texts: Dict[str, List[str]],
    edge_priors: Dict[Edge, float],
    edge_evidence: Dict[Edge, Tuple[float, float]],
    vocabulary: List[str],
    path_lambda: float = 0.2,
) -> float:
    posteriors = update_edge_priors(edge_priors, edge_evidence)

    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        key = (a, b) if (a, b) in posteriors else (b, a)
        weight = posteriors.get(key, 1.0)   
        length = euclidean_length(nodes[a], nodes[b])
        material += length * weight
        adj[a].append(b)
        adj[b].append(a)

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + euclidean_length(nodes[cur], nodes[nb])
                stack.append(nb)

    audit_matrix = build_audit_matrix(node_classes)
    signature = path_signature(audit_matrix)

    ontology_vectors = {node: ontology_frequency_vector(node_texts[node], vocabulary) for node in node_texts}
    term_weights = normalised_weight_vector(np.mean(list(ontology_vectors.values()), axis=0))

    hybrid_cost = material
    for node, node_class in node_classes.items():
        audit_vec = audit_matrix[node_classes.keys().__len__() - list(node_classes.keys()).index(node) - 1]
        ontology_vec = ontology_vectors[node]
        node_score_val = node_score(audit_vec, signature, ontology_vec, term_weights)
        hybrid_cost += path_lambda * dist[node] * node_score_val

    return hybrid_cost