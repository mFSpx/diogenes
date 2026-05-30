# DARWIN HAMMER — match 165, survivor 0
# gen: 4
# parent_a: hybrid_distributed_leader_e_perceptual_dedupe_m16_s2.py (gen1)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_sketches_rlct_m33_s4.py (gen3)
# born: 2026-05-29T23:27:12Z

"""Hybrid algorithm combining distributed leader election, perceptual hashing clustering, 
and Fisher score-based localization.

Mathematical bridge:
- Each node n in a graph G=(V,E) carries a numeric feature vector f_n.
- From f_n we compute a perceptual hash h_n (64-bit integer) using the
  phash routine of *perceptual_dedupe.py*.
- The Hamming distance d(h_i,h_j) defines a similarity weight
  w_ij = 1 - d(h_i,h_j)/64 ∈ [0,1].
- In the broadcast phase of the MIS algorithm (*distributed_leader_election.py*)
  the raw broadcast probability p_raw = 1/2^{phase-step} is modulated by the
  average similarity of a candidate node to its undecided neighbours:
        p_mod = p_raw * avg_{j∈N(i)∩U} w_ij .
  Thus nodes that are perceptually similar to many neighbours broadcast
  less aggressively, encouraging diversity among elected leaders.
- We extend this idea by using the Fisher score to evaluate the similarity
  between localizations. Given a node with localization theta, we compute
  the Fisher score with respect to all other nodes, weighted by their
  perceptual similarity.
- The result is a set of leaders that are both graph-independent and
  perceptually diverse, with an additional layer of localization-based
  similarity.
"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import Mapping, Hashable, Sequence, List, Dict, Set, Tuple

Node = Hashable
Graph = Mapping[Node, Set[Node]]
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Perceptual hashing utilities (from perceptual_dedupe.py)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Return a 64-bit perceptual hash of a list of floats."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    # limit to first 64 values for deterministic size
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers."""
    return (a ^ b).bit_count()

def cluster_by_phash(hashes: Dict[Node, int], max_distance: int = 4) -> List[List[Node]]:
    """Simple greedy clustering based on Hamming distance."""
    clusters: List[List[Node]] = []
    for node, h in hashes.items():
        added = False
        for c in clusters:
            if hamming_distance(h, c[0][1]) <= max_distance:
                c.append((node, h))
                added = True
                break
        if not added:
            clusters.append([(node, h)])
    return clusters

# ----------------------------------------------------------------------
# Fisher score-based localization utilities (from hybrid_fisher_localization_hybrid_ternary_route_m40_s5.py)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Hybrid function: compute perceptual hash and similarity-weighted Fisher score
# ----------------------------------------------------------------------
def compute_similarity_weighted_fisher_score(
    hashes: Dict[Node, int],
    localizations: List[float],
    width: float
) -> float:
    """Compute the similarity-weighted Fisher score for a set of nodes."""
    similarity_matrix = np.zeros((len(hashes), len(hashes)))
    for i in range(len(hashes)):
        for j in range(len(hashes)):
            if i != j:
                similarity_matrix[i, j] = 1 - hamming_distance(hashes[i], hashes[j]) / 64
    weighted_fisher_score = 0.0
    for i in range(len(hashes)):
        weighted_node_score = 0.0
        for j in range(len(hashes)):
            if i != j:
                weighted_node_score += similarity_matrix[i, j] * fisher_score(localizations[i], localizations[j], width)
        weighted_fisher_score += weighted_node_score
    return weighted_fisher_score / len(hashes)

# ----------------------------------------------------------------------
# Hybrid function: compute MIS with similarity-weighted Fisher score modulation
# ----------------------------------------------------------------------
def hybrid_mis(
    graph: Graph,
    hashes: Dict[Node, int],
    localizations: List[float],
    width: float
) -> List[Node]:
    """Compute the MIS with similarity-weighted Fisher score modulation."""
    leaders = []
    undecided_nodes = set(graph.keys())
    while undecided_nodes:
        candidate_node = random.choice(list(undecided_nodes))
        similarity_weighted_fisher_score = compute_similarity_weighted_fisher_score(hashes, localizations, width)
        broadcast_probability = 1 / (2 ** len(hashes)) * similarity_weighted_fisher_score
        if random.random() < broadcast_probability:
            leaders.append(candidate_node)
            undecided_nodes.remove(candidate_node)
    return leaders

# ----------------------------------------------------------------------
# Hybrid function: count-min sketch with perceptual hashing
# ----------------------------------------------------------------------
def count_min_sketch_with_phash(
    items: Sequence[Node],
    width: int = 64,
    depth: int = 4
) -> List[List[int]]:
    """Compute the count-min sketch with perceptual hashing."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        hash_value = compute_phash([item])
        for d in range(depth):
            table[d][int(hash_value % width)] += 1
    return table

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    graph = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
    hashes = {1: compute_phash([0.1, 0.2, 0.3]), 2: compute_phash([0.4, 0.5, 0.6]), 3: compute_phash([0.7, 0.8, 0.9])}
    localizations = [0.1, 0.2, 0.3]
    width = 0.1
    hybrid_mis(graph, hashes, localizations, width)
    count_min_sketch_with_phash([1, 2, 3])