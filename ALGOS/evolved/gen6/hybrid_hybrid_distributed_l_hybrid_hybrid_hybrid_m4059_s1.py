# DARWIN HAMMER — match 4059, survivor 1
# gen: 6
# parent_a: hybrid_distributed_leader_e_perceptual_dedupe_m16_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1219_s0.py (gen5)
# born: 2026-05-29T23:53:16Z

"""
Hybrid Algorithm Combining Distributed Leader Election and Stylometry-Based Feature Extraction

This module fuses the core mathematics of hybrid_distributed_leader_election.py and hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1219_s0.py.
The mathematical bridge between the two structures is the use of the perceptual hash h_n as a feature vector for each node n in a graph G=(V,E),
and the stylometry feature extraction lsm_vector() as a basis for the feature weighting in the distributed leader election algorithm.

Mathematical bridge:
- Each node n in a graph G=(V,E) carries a numeric feature vector f_n = (h_n, lsm_vector(text_n))
- From f_n we compute a perceptual hash h_n (64-bit integer) using the phash routine of *perceptual_dedupe.py*
- The Hamming distance d(h_i,h_j) defines a similarity weight w_ij = 1 - d(h_i,h_j)/64 ∈ [0,1]
- In the broadcast phase of the MIS algorithm (*distributed_leader_election.py*)
  the raw broadcast probability p_raw = 1/2^{phase-step} is modulated by the average similarity of a candidate node to its undecided neighbours:
        p_mod = p_raw * avg_{j∈N(i)∩U} w_ij .
  Thus nodes that are perceptually similar to many neighbours and have similar stylometry features broadcast less aggressively,
  encouraging diversity among elected leaders and reducing the impact of outliers.

The hybrid functions below implement:
1. hashing of node attributes,
2. construction of a similarity matrix from Hamming distances,
3. a MIS procedure that uses the similarity-modulated broadcast probability and stylometry feature weighting.

The result is a set of leaders that are both graph-independent and perceptually diverse, with a reduced impact of outliers.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Mapping, Hashable, Sequence, List, Dict, Set, Tuple
import math
import random

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
        for c in clusters:
            if any(hamming_distance(h, h_c) <= max_distance for h_c in [hashes[node_c] for node_c in c]):
                c.append(node)
                break
        else:
            clusters.append([node])
    return clusters

# ----------------------------------------------------------------------
# Stylometry utilities (from hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1219_s0.py)
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}

PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

def words(text: str) -> List[str]:
    return [word for word in (text or "").lower().split() if word.isalpha()]

def lsm_vector(text: str) -> dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[word] for word in cat) / total for cat in FUNCTION_CATS.values()
    }

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def compute_similarity_matrix(graph: Graph, hashes: Dict[Node, int]) -> np.ndarray:
    """Compute a similarity matrix from Hamming distances."""
    num_nodes = len(graph)
    similarity_matrix = np.ones((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if i in graph[j]:
                similarity_matrix[i, j] = 1 - hamming_distance(hashes[i], hashes[j]) / 64
                similarity_matrix[j, i] = similarity_matrix[i, j]
    return similarity_matrix

def stylometry_weighted_mis(graph: Graph, hashes: Dict[Node, int], similarity_matrix: np.ndarray) -> List[Node]:
    """Run MIS algorithm with stylometry feature weighting."""
    num_nodes = len(graph)
    weights = np.array([1] * num_nodes)
    for i in range(num_nodes):
        weights[i] *= similarity_matrix[i, :].mean()
    return distributed_leader_election(graph, weights)

def distributed_leader_election(graph: Graph, weights: np.ndarray) -> List[Node]:
    """Run distributed leader election algorithm."""
    num_nodes = len(graph)
    leaders = []
    undecided = set(graph.keys())
    for _ in range(num_nodes):
        candidate = random.choice(list(undecided))
        raw_broadcast_prob = 1 / (2 ** (len(leaders) + 1))
        modulated_broadcast_prob = raw_broadcast_prob * np.mean(weights[list(graph[candidate]) & undecided])
        if random.random() < modulated_broadcast_prob:
            leaders.append(candidate)
        undecided -= {candidate}
    return leaders

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    graph = {
        'A': {'B', 'C'},
        'B': {'A', 'D'},
        'C': {'A', 'F'},
        'D': {'B'},
        'E': {'F'},
        'F': {'C', 'E'}
    }
    hashes = {
        'A': compute_phash([1.0, 2.0, 3.0]),
        'B': compute_phash([4.0, 5.0, 6.0]),
        'C': compute_phash([7.0, 8.0, 9.0]),
        'D': compute_phash([10.0, 11.0, 12.0]),
        'E': compute_phash([13.0, 14.0, 15.0]),
        'F': compute_phash([16.0, 17.0, 18.0])
    }
    similarity_matrix = compute_similarity_matrix(graph, hashes)
    leaders = stylometry_weighted_mis(graph, hashes, similarity_matrix)
    print(leaders)