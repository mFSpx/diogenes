# DARWIN HAMMER — match 2422, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_semantic_neig_hybrid_temporal_moti_m47_s3.py (gen3)
# parent_b: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s3.py (gen2)
# born: 2026-05-29T23:42:11Z

"""
Hybrid module combining Hybrid Semantic‑Temporal Morphology Fusion 
and Hybrid Hard Truth Math, Minimum-Cost Tree scoring and Bayesian evidence update.

Mathematical bridge:
The core topologies of both parents are fused: the stylometry features/classifier helpers 
from Hybrid Hard Truth Math supply the probabilistic weights, while the 
spatio-temporal motif support and semantic similarity from Hybrid Semantic‑Temporal 
Morphology Fusion supply the geometric quantities.

The recovery priority  R(m)  is used as a continuous weight for the temporal support  s(p) 
and the edge posterior belief *p_e*. The cosine similarity between the entity’s feature vector 
and its *semantic neighbours* supplies a spatial proximity factor  σ(i,j)  that replaces 
the binary matrix D(i,j) used by the possum filter and the deterministic edge contribution 
ℓ(e) in the stylometry features/classifier helpers.

The resulting hybrid cost is

    C_h = Σ_e (p_e · ℓ(e)) / Σ_e |p_e| + λ Σ_v (q_v · d(v)) / Σ_v |q_v| + γ Σ_m (R(m) · σ)

where λ is a path-weight, γ is a morphology-weight, d(v) is the root-to-node distance, 
and σ is the spatial proximity factor.
"""

import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import numpy as np

# ---------- Data structures -------------------------------------------------

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int

@dataclass(frozen=True)
class HybridMotif:
    """Entity representing a spatio‑temporal motif with morphology."""
    pattern: Tuple[str, ...]
    support: int
    centroid_lat: float
    centroid_lon: float
    morphology: Morphology
    vector: Tuple[float, ...]          # semantic feature vector
    score: float                       # unified hybrid score

@dataclass(frozen=True)
class Edge:
    node1: str
    node2: str
    posterior_belief: float

# ---------- Parent A – morphology & semantic utilities -----------------------

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (2 * (length + width)) / (length * width)

def cosine_similarity(vector1: Tuple[float, ...], vector2: Tuple[float, ...]) -> float:
    dot_product = sum(x * y for x, y in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(x ** 2 for x in vector1))
    magnitude2 = math.sqrt(sum(x ** 2 for x in vector2))
    return dot_product / (magnitude1 * magnitude2)

# ---------- Parent B – tree metrics and Bayesian update -----------------------

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    adjacency_list = {node: [] for node in nodes}
    edge_lengths = {}
    root_distances = {node: 0 for node in nodes}
    
    for edge in edges:
        adjacency_list[edge.node1].append(edge.node2)
        adjacency_list[edge.node2].append(edge.node1)
        edge_lengths[(edge.node1, edge.node2)] = length(nodes[edge.node1], nodes[edge.node2])
        edge_lengths[(edge.node2, edge.node1)] = length(nodes[edge.node1], nodes[edge.node2])
    
    # calculate root-to-node distances
    queue = [root]
    while queue:
        node = queue.pop(0)
        for neighbor in adjacency_list[node]:
            if neighbor not in root_distances or root_distances[node] + edge_lengths[(node, neighbor)] < root_distances[neighbor]:
                root_distances[neighbor] = root_distances[node] + edge_lengths[(node, neighbor)]
                queue.append(neighbor)
    
    return adjacency_list, edge_lengths, root_distances

def bayes_edge_posteriors(edges: List[Edge]) -> Dict[Tuple[str, str], float]:
    posterior_beliefs = {}
    for edge in edges:
        posterior_beliefs[(edge.node1, edge.node2)] = edge.posterior_belief
        posterior_beliefs[(edge.node2, edge.node1)] = edge.posterior_belief
    return posterior_beliefs

# ---------- Hybrid module -------------------------------------------------

def hybrid_stylometry(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Edge],
    motifs: List[HybridMotif],
) -> float:
    adjacency_list, edge_lengths, root_distances = tree_metrics(nodes, edges, list(nodes.keys())[0])
    posterior_beliefs = bayes_edge_posteriors(edges)
    
    stylometry_features = 0
    for edge in edges:
        stylometry_features += posterior_beliefs[(edge.node1, edge.node2)] * edge_lengths[(edge.node1, edge.node2)]
    
    morphology_weight = 0
    for motif in motifs:
        morphology_weight += motif.morphology.mass * cosine_similarity(motif.vector, (1, 1, 1))
    
    return stylometry_features + morphology_weight

def hybrid_tree_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Edge],
    motifs: List[HybridMotif],
) -> float:
    adjacency_list, edge_lengths, root_distances = tree_metrics(nodes, edges, list(nodes.keys())[0])
    posterior_beliefs = bayes_edge_posteriors(edges)
    
    tree_cost = 0
    for edge in edges:
        tree_cost += posterior_beliefs[(edge.node1, edge.node2)] * edge_lengths[(edge.node1, edge.node2)]
    
    morphology_weight = 0
    for motif in motifs:
        morphology_weight += motif.morphology.mass * cosine_similarity(motif.vector, (1, 1, 1))
    
    return tree_cost + morphology_weight

def hybrid_motif_support(
    motifs: List[HybridMotif],
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Edge],
) -> float:
    motif_support = 0
    for motif in motifs:
        motif_support += motif.support * cosine_similarity(motif.vector, (1, 1, 1))
    
    return motif_support

if __name__ == "__main__":
    nodes = {
        "A": (0, 0),
        "B": (1, 1),
        "C": (2, 2),
    }
    edges = [
        Edge("A", "B", 0.5),
        Edge("B", "C", 0.3),
        Edge("C", "A", 0.2),
    ]
    motifs = [
        HybridMotif(
            ("A", "B", "C"),
            10,
            1.0,
            1.0,
            Morphology(1.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            1.0,
        )
    ]
    
    print(hybrid_stylometry(nodes, edges, motifs))
    print(hybrid_tree_cost(nodes, edges, motifs))
    print(hybrid_motif_support(motifs, nodes, edges))