# DARWIN HAMMER — match 4456, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_nlms_omni_cha_m1_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_minimu_temporal_motifs_m280_s0.py (gen3)
# born: 2026-05-29T23:55:50Z

"""
This module represents a hybrid algorithm, combining the principles of 
minimum-cost tree scoring and NLMS prediction from 
hybrid_hybrid_hybrid_minimu_hybrid_nlms_omni_cha_m1_s0.py, 
and Bayesian updates to temporal motif mining from 
hybrid_hybrid_hybrid_minimu_temporal_motifs_m280_s0.py.

The mathematical bridge between these two systems lies in the application 
of Bayesian updates to the edge weights of the minimum-cost tree, 
effectively allowing the tree to adapt and re-weight its edges based on 
both physical distances and temporal motif topology.

The core idea is to use Bayesian updates to update the edge weights in the 
tree scoring function, thus creating a dynamic system where the tree 
structure, temporal motifs, and Bayesian updates inform each other.

The mathematical interface is established by treating the edge weights as 
a vector of coefficients that are applied to the features of the nodes 
in the graph, producing a prediction that is used to update the edge 
weights in the minimum-cost tree.

This fusion requires the creation of a new data structure that combines 
the features of the nodes in the graph with the weights of the Bayesian 
updates, and the development of new functions that operate on this 
data structure to produce the hybrid output.
"""

import math
import numpy as np
from collections import Counter
from dataclasses import dataclass

@dataclass(frozen=True)
class Node: id: str; features: list[float]
@dataclass(frozen=True)
class Edge: node1: str; node2: str; weight: float
@dataclass(frozen=True)
class BurstSignal: key: str; count: int; z_score: float; prior: float; likelihood: float; false_positive: float
@dataclass(frozen=True)
class TemporalMotif: pattern: tuple[str,...]; support: int; prior: float; likelihood: float; false_positive: float

def length(a: list[float], b: list[float]) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(sum((a[i] - b[i]) ** 2 for i in range(len(a)))) ** 0.5

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def update_edge_weights(edges: list[Edge], nodes: list[Node], prior: float, likelihood: float, false_positive: float) -> list[Edge]:
    updated_edges = []
    for edge in edges:
        node1 = next((n for n in nodes if n.id == edge.node1), None)
        node2 = next((n for n in nodes if n.id == edge.node2), None)
        if node1 and node2:
            distance = length(node1.features, node2.features)
            marginal = bayes_marginal(prior, likelihood, false_positive)
            updated_weight = bayes_update(edge.weight, likelihood, marginal) * distance
            updated_edges.append(Edge(edge.node1, edge.node2, updated_weight))
    return updated_edges

def detect_bursts(events: list[dict], key: str='type', prior: float = 0.5, likelihood: float = 0.8, false_positive: float = 0.1) -> list[BurstSignal]:
    c=Counter(str(e.get(key,'')) for e in events)
    if not c: return []
    mean=sum(c.values())/len(c); sd=math.sqrt(sum((v-mean)**2 for v in c.values())/len(c)) or 1.0
    burst_signals = []
    for k,v in c.items():
        marginal = bayes_marginal(prior, likelihood, false_positive)
        z_score = (v - mean) / sd
        burst_signals.append(BurstSignal(k, v, z_score, prior, likelihood, false_positive))
    return burst_signals

def mine_temporal_motifs(events: list[dict], prior: float, likelihood: float, false_positive: float) -> list[TemporalMotif]:
    sessions = []
    cur = []
    last = None
    for e in sorted(events, key=lambda x: x.get('t', 0)):
        t = float(e.get('t', 0))
        if cur and last is not None and t - last > 1800.0:
            sessions.append(cur)
            cur = []
        cur.append(e)
        last = t
    if cur:
        sessions.append(cur)
    motifs = []
    for session in sessions:
        patterns = Counter(tuple(e.get('type', '')) for e in session)
        for pattern, support in patterns.items():
            marginal = bayes_marginal(prior, likelihood, false_positive)
            motifs.append(TemporalMotif(pattern, support, prior, likelihood, false_positive))
    return motifs

if __name__ == "__main__":
    nodes = [Node("A", [1.0, 2.0]), Node("B", [3.0, 4.0])]
    edges = [Edge("A", "B", 1.0)]
    events = [{"t": 1.0, "type": "A"}, {"t": 2.0, "type": "B"}, {"t": 3.0, "type": "A"}]

    updated_edges = update_edge_weights(edges, nodes, 0.5, 0.8, 0.1)
    burst_signals = detect_bursts(events)
    temporal_motifs = mine_temporal_motifs(events, 0.5, 0.8, 0.1)

    print(updated_edges)
    print(burst_signals)
    print(temporal_motifs)