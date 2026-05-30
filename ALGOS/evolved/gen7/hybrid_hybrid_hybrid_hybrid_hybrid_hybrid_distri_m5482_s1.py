# DARWIN HAMMER — match 5482, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1213_s7.py (gen6)
# parent_b: hybrid_hybrid_distributed_l_hybrid_hybrid_model__m99_s1.py (gen3)
# born: 2026-05-30T00:02:24Z

import math
import random
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Set

Node = int
Graph = Dict[Node, Set[Node]]

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def compute_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def compute_dhash(values: List[float]) -> int:
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def broadcast_probability(phase: int, step: int) -> float:
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))

def build_graph(elements: List[Morphology]) -> Graph:
    graph: Graph = {}
    hashes: Dict[str, int] = {}
    for i, element in enumerate(elements):
        hashes[str(i)] = compute_phash([element.length, element.width, element.height, element.mass])
    for i in range(len(elements)):
        graph[i] = set()
        for j in range(i + 1, len(elements)):
            if hamming_distance(hashes[str(i)], hashes[str(j)]) <= 4:
                graph[i].add(j)
                graph[j] = graph.get(j, set())
                graph[j].add(i)
    return graph

def compute_sphericity_index(cluster: List[Morphology]) -> float:
    lengths = [morphology.length for morphology in cluster]
    widths = [morphology.width for morphology in cluster]
    heights = [morphology.height for morphology in cluster]
    masses = [morphology.mass for morphology in cluster]
    return np.mean([length / width for length, width in zip(lengths, widths)])

def compute_bic(cluster: List[Morphology]) -> float:
    lengths = [morphology.length for morphology in cluster]
    widths = [morphology.width for morphology in cluster]
    heights = [morphology.height for morphology in cluster]
    masses = [morphology.mass for morphology in cluster]
    n = len(cluster)
    k = 4  
    return n * np.log(np.mean([length / width for length, width in zip(lengths, widths)])) + k * np.log(n)

def compute_rict(cluster: List[Morphology]) -> float:
    lengths = [morphology.length for morphology in cluster]
    widths = [morphology.width for morphology in cluster]
    heights = [morphology.height for morphology in cluster]
    masses = [morphology.mass for morphology in cluster]
    d = 4  
    return (d / 2) * np.log(len(cluster)) - np.log(np.mean([length / width for length, width in zip(lengths, widths)]))

def hybrid_operation(elements: List[Morphology]) -> None:
    graph = build_graph(elements)
    clusters = []
    visited = set()
    for node in graph:
        if node not in visited:
            cluster = []
            stack = [node]
            while stack:
                current_node = stack.pop()
                if current_node not in visited:
                    visited.add(current_node)
                    cluster.append(elements[current_node])
                    stack.extend(graph[current_node] - visited)
            clusters.append(cluster)
    for i, cluster in enumerate(clusters):
        sphericity_index_value = compute_sphericity_index(cluster)
        bic_value = compute_bic(cluster)
        rict_value = compute_rict(cluster)
        print(f"Cluster {i}: Sphericity Index = {sphericity_index_value}, BIC = {bic_value}, RLCT = {rict_value}")

if __name__ == "__main__":
    elements = [
        Morphology(length=10.0, width=20.0, height=30.0, mass=40.0),
        Morphology(length=15.0, width=25.0, height=35.0, mass=45.0),
        Morphology(length=20.0, width=30.0, height=40.0, mass=50.0),
        Morphology(length=25.0, width=35.0, height=45.0, mass=55.0),
        Morphology(length=30.0, width=40.0, height=50.0, mass=60.0)
    ]
    hybrid_operation(elements)