# DARWIN HAMMER — match 1944, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_perceptual_de_hybrid_label_foundry_m1030_s2.py (gen4)
# parent_b: hybrid_hybrid_physarum_netw_hybrid_hybrid_hybrid_m520_s3.py (gen5)
# born: 2026-05-29T23:39:59Z

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, List, Tuple, Dict
from collections import Counter, defaultdict

Vector = Sequence[float]

@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  # binary 0/1

@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float  # in [0,1]

@dataclass(frozen=True)
class LabelError:
    doc_id: str
    given_label: int
    suggested_label: int
    error_probability: float

@dataclass
class Node:
    id: str
    conductance: float
    pressure: float

def compute_dhash(values: List[float]) -> int:
    """Difference hash: 1 bit per adjacent pair, 1 if decreasing."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: List[float]) -> int:
    """Average hash limited to first 64 values."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers."""
    return bin(a ^ b).count('1')

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on a single edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / edge_length * (pressure_a - pressure_b)

def update_conductance(conductance: float, dt: float, gain: float, decay: float, flux: float) -> float:
    """Update conductance using the physarum dynamics."""
    return max(0, conductance + dt * (gain * abs(flux) - decay * conductance))

def labeling_function(name: str | None = None):
    """Decorator that annotates a labeling function with a name."""
    def deco(fn: Callable[[dict], int]) -> Callable[[dict], int]:
        def wrapper(doc: dict) -> int:
            return fn(doc)
        wrapper.__name__ = name
        return wrapper
    return deco

def hybrid_step(nodes: List[Node], edges: List[Tuple[str, str]], dt: float, gain: float, decay: float) -> List[Node]:
    """Perform a hybrid step, updating the conductance and pressures of the nodes."""
    new_nodes = []
    edge_dict = {(edge[0], edge[1]): True for edge in edges}
    for node in nodes:
        flux_sum = 0
        for edge in edges:
            if edge[0] == node.id and edge[1] in [n.id for n in nodes]:
                other_node = next(n for n in nodes if n.id == edge[1])
                flux_sum += flux(node.conductance, 1.0, node.pressure, other_node.pressure)
            elif edge[1] == node.id and edge[0] in [n.id for n in nodes]:
                other_node = next(n for n in nodes if n.id == edge[0])
                flux_sum -= flux(node.conductance, 1.0, other_node.pressure, node.pressure)
        new_conductance = update_conductance(node.conductance, dt, gain, decay, flux_sum)
        new_pressure = node.pressure + dt * flux_sum
        new_nodes.append(Node(node.id, new_conductance, new_pressure))
    return new_nodes

def aggregate_labels(labeling_results: List[LabelingFunctionResult]) -> List[ProbabilisticLabel]:
    """Aggregate labeling function results into probabilistic labels."""
    label_counts = defaultdict(Counter)
    for result in labeling_results:
        label_counts[result.doc_id][result.label] += 1
    probabilistic_labels = []
    for doc_id, label_counts in label_counts.items():
        total_votes = sum(label_counts.values())
        if total_votes == 0:
            probabilistic_labels.append(ProbabilisticLabel(doc_id, -1, 0.0))
        else:
            label = max(label_counts, key=label_counts.get)
            confidence = label_counts[label] / total_votes
            probabilistic_labels.append(ProbabilisticLabel(doc_id, label, confidence))
    return probabilistic_labels

def integrate_phash_and_rbf(nodes: List[Node], labeling_results: List[LabelingFunctionResult]):
    phashes = [compute_phash([node.pressure]) for node in nodes]
    labels = aggregate_labels(labeling_results)
    label_dict = {label.doc_id: label.label for label in labels}
    node_labels = []
    for node in nodes:
        node_label = label_dict.get(node.id, -1)
        node_labels.append(node_label)
    rbf_values = np.array([node.pressure for node in nodes])
    return phashes, node_labels, rbf_values

if __name__ == "__main__":
    nodes = [Node("A", 1.0, 0.5), Node("B", 1.0, 0.5)]
    edges = [("A", "B"), ("B", "A")]
    new_nodes = hybrid_step(nodes, edges, 0.1, 1.0, 0.1)
    print(new_nodes)
    labeling_results = [LabelingFunctionResult("LF1", "A", 1), LabelingFunctionResult("LF2", "B", 0)]
    probabilistic_labels = aggregate_labels(labeling_results)
    print(probabilistic_labels)
    phashes, node_labels, rbf_values = integrate_phash_and_rbf(new_nodes, labeling_results)
    print(phashes)
    print(node_labels)
    print(rbf_values)