# DARWIN HAMMER — match 70, survivor 1
# gen: 4
# parent_a: shap_attribution.py (gen0)
# parent_b: hybrid_hybrid_pheromone_hyb_hybrid_infotaxis_min_m81_s0.py (gen3)
# born: 2026-05-29T23:28:13Z

import numpy as np
import random
import math
from pathlib import Path
from itertools import combinations
from typing import Any, Callable, Iterable

Node = int
Graph = dict[int, set[int]]
Model = dict[int, float]

def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    return math.factorial(subset_size) * math.factorial(feature_count - subset_size - 1) / math.factorial(feature_count)

def compute_dhash(values: list[float]) -> int:
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: list[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def broadcast_probability(phase: int, step: int) -> float:
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))

def shap_value(feature_index: int, feature_count: int, value_fn: Callable[[frozenset[int]], float]) -> float:
    total = 0.0
    for k in range(feature_count + 1):
        for subset in combinations(range(feature_count), k):
            s = frozenset(subset)
            total += shapley_kernel_weight(k, feature_count) * (value_fn(s | {feature_index}) - value_fn(s))
    return total

def leader_election(graph: Graph, values: list[float], seed: int | str | None = None) -> set[Node]:
    rng = random.Random(seed)
    undecided = set(graph)
    leaders: set[Node] = set()
    blocked: set[Node] = set()
    for phase in range(1, 8 + 1):
        if not undecided:
            break
        p = broadcast_probability(8, phase)
        broadcasts = {n for n in undecided if rng.random() < p}
        new_leaders = {n for n in broadcasts if not (graph.get(n, set()) & broadcasts)}
        leaders |= new_leaders
        blocked.update(undecided - new_leaders)
        undecided -= new_leaders
        subgraph_values = {n: values[list(graph.keys()).index(n)] for n in undecided}
        phash = compute_phash(subgraph_values.values())
        dhash = compute_dhash(subgraph_values.values())
        for n in broadcasts:
            if graph.get(n, set()) & broadcasts:
                continue
            for m in graph[n]:
                if m in broadcasts:
                    continue
                if hamming_distance(phash, compute_phash([subgraph_values[n], subgraph_values[m]])) <= 2:
                    graph[n].add(m)
    return leaders

def minhash_clustering(graph: Graph, seed: int | str | None = None) -> dict[Node, set[Node]]:
    rng = random.Random(seed)
    clusters: dict[Node, set[Node]] = {}
    for n in graph:
        clusters[n] = set()
    for n in graph:
        if n in clusters[n]:
            continue
        cluster = leader_election({n: graph[n]}, [1.0]*len(graph[n]), seed)
        for m in graph:
            if m in clusters and cluster & clusters[m]:
                clusters[m] |= cluster
                break
        clusters[n] = cluster
    return clusters

def hybrid_feature_clustering(model: Model, graph: Graph, seed: int | str | None = None) -> dict[Node, set[Node]]:
    def value_fn(s: frozenset[int]) -> float:
        return sum(model[i] for i in s)

    shap_values = {i: shap_value(i, len(model), value_fn) for i in model}
    clusters = minhash_clustering(graph, seed)
    for n in model:
        cluster_values = [sum(model[i] for i in cluster) for cluster in clusters.values()]
        phash = compute_phash([model[n]])
        for cluster_phash in [compute_phash(cluster_values)]:
            if hamming_distance(phash, cluster_phash) <= 2:
                clusters[n] = list(clusters.values())[list(cluster_values).index(cluster_values[clusters[n]])]
                break
    return clusters

if __name__ == "__main__":
    model = {0: 0.5, 1: 0.3, 2: 0.2}
    graph = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
    clusters = hybrid_feature_clustering(model, graph)
    print(clusters)