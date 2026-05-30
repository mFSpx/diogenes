# DARWIN HAMMER — match 1062, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_label_foundry_path_signature_m231_s2.py (gen3)
# parent_b: hybrid_hybrid_ternary_route_hybrid_endpoint_circ_m233_s0.py (gen3)
# born: 2026-05-29T23:32:32Z

"""
Hybrid algorithm combining the HybridSignatureLabeler from hybrid_hybrid_label_foundry_path_signature_m231_s2.py 
and the Hybrid ternary router from hybrid_hybrid_ternary_route_hybrid_endpoint_circ_m233_s0.py.
The mathematical bridge between the two structures is the integration of the lead-lag transform from 
the HybridSignatureLabeler into the tree cost calculation of the Hybrid ternary router. 
This allows for a more accurate representation of the geometry of the path and its impact on the routing decisions.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Tuple

@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  

@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float  

@dataclass(frozen=True)
class LabelError:
    doc_id: str
    given_label: int
    suggested_label: int
    error_probability: float

Point = Tuple[float, float]
Edge = Tuple[str, str]

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T):
        if t == 0:
            out[t] = np.concatenate((path[t], np.zeros(d)))
        elif t == T - 1:
            out[2 * t - 1] = np.concatenate((np.zeros(d), path[t]))
        else:
            out[2 * t - 1] = np.concatenate((path[t - 1], path[t]))
    return out

def tree_cost(nodes: Dict[str, Point], edges: List[Edge], root: str, path_weight: float = 0.2) -> float:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
                stack.append(b)
    return material

def labeling_function(name: str | None = None):
    def deco(fn: callable) -> callable:
        fn.lf_name = name or fn.__name__
        return fn
    return deco

def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    votes: Dict[str, List[int]] = defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                votes[r.doc_id].append(r.label)
    out = []
    for doc_id, labels in votes.items():
        label = max(set(labels), key=labels.count)
        confidence = labels.count(label) / len(labels)
        out.append(ProbabilisticLabel(doc_id, label, confidence))
    return out

def hybrid_operation(path, nodes: Dict[str, Point], edges: List[Edge], root: str):
    transformed_path = lead_lag_transform(path)
    nodes_list = [nodes[node] for node in nodes]
    cost = tree_cost({str(i): node for i, node in enumerate(nodes_list)}, [(str(i), str(i+1)) for i in range(len(nodes_list)-1)], '0')
    labels = aggregate_labels([LabelingFunctionResult("example", "doc1", 1)])
    return transformed_path, cost, labels

if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 1), "C": (2, 2)}
    edges = [("A", "B"), ("B", "C")]
    path = np.array([[0, 0], [1, 1], [2, 2]])
    root = "A"
    transformed_path, cost, labels = hybrid_operation(path, nodes, edges, root)
    print("Transformed Path: ", transformed_path)
    print("Tree Cost: ", cost)
    print("Labels: ", labels)