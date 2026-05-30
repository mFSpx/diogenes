# DARWIN HAMMER — match 5496, survivor 0
# gen: 7
# parent_a: hybrid_distributed_leader_e_perceptual_dedupe_m16_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_ssim_m1265_s1.py (gen6)
# born: 2026-05-30T00:02:25Z

#!/usr/bin/env python3
"""Hybrid algorithm fusing distributed_leader_election and 
hybrid_hybrid_hybrid_hybrid_ssim_m1265_s1.py, leveraging the similarity 
between graph node values using SSIM and the resulting hashes to inform the 
leader election process, ensuring that leaders are chosen from clusters of 
similar nodes, thus creating a more meaningful and efficient clustering of the graph."""

import numpy as np
from collections.abc import Mapping, Hashable
import random
import math
import sys
from pathlib import Path

Node = Hashable
Graph = Mapping[Node, set[Node]]

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

def ssim_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Calculate the structural similarity between two vectors."""
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    k1 = 0.01
    k2 = 0.03
    lum = 0.5
    k = (2 * lum - 1)
    mean1 = np.mean(vector1)
    mean2 = np.mean(vector2)
    cov = np.mean((vector1 - mean1) * (vector2 - mean2))
    var1 = np.mean((vector1 - mean1) ** 2) + C1
    var2 = np.mean((vector2 - mean2) ** 2) + C1
    ssim = ((2 * mean1 * mean2 + k1 * mean1 + k2 * mean2) * cov +
            C2 * var1 + C2 * var2) / (var1 + var2 + C2 * var1 + C2 * var2)
    return ssim

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def broadcast_probability(phase: int, step: int) -> float:
    """Return p=1/2^(phase-step), clamped to [0, 1]."""
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))

def maximal_independent_set(graph: Graph, phases: int = 8, seed: int | str | None = None) -> set[Node]:
    """Approximate a maximal independent set using local broadcast rounds."""
    rng = random.Random(seed)
    undecided = set(graph)
    leaders: set[Node] = set()
    blocked: set[Node] = set()
    for phase in range(1, phases + 1):
        if not undecided:
            break
        p = broadcast_probability(phases, phase)
        broadcasts = {n for n in undecided if rng.random() < p}
        new_leaders = {n for n in broadcasts if not (graph.get(n, set()) & broadcasts)}
        leaders |= new_leaders
        blocked |= set().union(*(graph.get(n, set()) for n in new_leaders), new_leaders) if new_leaders else set()
        undecided -= blocked
    for n in sorted(undecided, key=str):
        if not (graph.get(n, set()) & leaders):
            leaders.add(n)
    return leaders

def cluster_by_ssim(graph: Graph, similarity_threshold: float = 0.9, seed: int | str | None = None) -> set[Node]:
    """Cluster nodes in the graph based on their SSIM similarity."""
    rng = random.Random(seed)
    clusters: dict[Node, set[Node]] = {n: {n} for n in graph}
    for n1 in graph:
        for n2 in graph:
            if n1 != n2:
                values1 = list(graph[n1])
                values2 = list(graph[n2])
                similarity = ssim_similarity(np.array(values1), np.array(values2))
                if similarity > similarity_threshold:
                    clusters[n1] |= {n2}
                    clusters[n2] |= {n1}
    for n in clusters:
        clusters[n] = {n for n in clusters[n] if n == max(clusters[n], key=lambda x: len(graph[x]))}
    return set().union(*clusters.values())

def fusion_election(graph: Graph, phases: int = 8, seed: int | str | None = None) -> set[Node]:
    """Fusion of distributed leader election and SSIM clustering."""
    leaders = maximal_independent_set(graph, phases, seed)
    clusters = cluster_by_ssim(graph, seed=seed)
    return set().union(*(set(c) for c in clusters if c & leaders))

def test_fusion():
    graph = {1: {1, 2, 3}, 2: {2, 3, 4}, 3: {3, 4, 5}, 4: {4, 5, 6}, 5: {5, 6, 7}, 6: {6, 7, 8}, 7: {7, 8}}
    leaders = fusion_election(graph, seed=42)
    assert len(leaders) > 0

if __name__ == "__main__":
    test_fusion()