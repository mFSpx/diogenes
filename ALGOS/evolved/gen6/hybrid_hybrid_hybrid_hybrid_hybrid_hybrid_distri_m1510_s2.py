# DARWIN HAMMER — match 1510, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m1019_s1.py (gen5)
# parent_b: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s3.py (gen2)
# born: 2026-05-29T23:37:06Z

"""
Hybrid Algorithm: Stylometry‑Weighted Ollivier‑Ricci Curvature → Epistemic Certainty → Leader Election

This module fuses the core topologies of:
- `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m1019_s1.py` – 
    stylometry → graph edge weights → Ollivier‑Ricci curvature → epistemic certainty
- `hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s3.py` – 
    tropical max‑plus algebra for aggregating piecewise‑linear functions → 
    leader election via Hoeffding bound and simulated annealing

Mathematical bridge: 
The Ollivier-Ricci curvature of the graph, derived from stylometry, is used as the weight 
in the tropical max-plus algebra. This allows for the aggregation of the curvature 
values across the graph, providing a measure of the overall certainty of the graph. 
The leader election algorithm then uses this aggregated certainty value to make 
decisions about the election of leaders.

"""

import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np

FUNCTION_CATS: Dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers "
        "herself it its itself they them their theirs themselves what which who whom "
        "this that these those am is are was were be been being have has had do does did "
        "shall should will would may might must can could ought i'm you're he's she's "
        "it's we're they're i've you've he's she's it's we've they've i'm you're he's "
        "she's it's we're they're i'd you'd he'd she'd it'd we'd they'd i'll you'll "
        "he'll she'll it'll we'll they'll".split()
    )
}

@dataclass
class CertaintyFlag:
    label: str
    confidence: int

def stylometry_features(text: str) -> Dict[str, int]:
    """Returns a dict of category counts."""
    words = re.findall(r'\b\w+\b', text.lower())
    feature_counts = Counter(words)
    return {cat: sum(feature_counts[word] for word in func_cats) 
            for cat, func_cats in FUNCTION_CATS.items()}

def build_weighted_graph(features_list: List[Dict[str, int]]) -> Tuple[np.ndarray, List[float]]:
    """Builds the adjacency matrix **W** from a list of feature dicts using cosine similarity 
    and also returns node strengths."""
    num_docs = len(features_list)
    W = np.zeros((num_docs, num_docs))
    node_strengths = []
    
    for i, features_i in enumerate(features_list):
        node_strength_i = sum(features_i.values())
        node_strengths.append(node_strength_i)
        
        for j, features_j in enumerate(features_list):
            if i == j:
                continue
            dot_product = sum(features_i.get(word, 0) * features_j.get(word, 0) 
                                for word in set(features_i) | set(features_j))
            magnitude_i = math.sqrt(sum(val ** 2 for val in features_i.values()))
            magnitude_j = math.sqrt(sum(val ** 2 for val in features_j.values()))
            cosine_similarity = dot_product / (magnitude_i * magnitude_j) if magnitude_i * magnitude_j != 0 else 0
            W[i, j] = cosine_similarity
    
    return W, node_strengths

def curvature_to_certainty(W: np.ndarray, node_strengths: List[float]) -> Dict[Tuple[int, int], CertaintyFlag]:
    """Computes κ per edge, maps to confidence, and yields a dict of `CertaintyFlag` objects keyed by edge tuples."""
    certainty_flags = {}
    
    for i in range(W.shape[0]):
        for j in range(i + 1, W.shape[1]):
            if W[i, j] == 0:
                continue
            curvature = 1 - abs(node_strengths[i] - node_strengths[j]) / (node_strengths[i] + node_strengths[j] + 1e-6)
            confidence_bps = int((curvature + 1) / 2 * 10000)
            label = "high" if confidence_bps > 5000 else "low"
            certainty_flags[(i, j)] = CertaintyFlag(label, confidence_bps)
    
    return certainty_flags

def tropical_broadcast(W: np.ndarray, node_strengths: List[float], num_iterations: int = 10) -> np.ndarray:
    """Computes a broadcast strength vector `b` by repeatedly applying tropical matrix multiplication."""
    b = np.array(node_strengths)
    
    for _ in range(num_iterations):
        b_new = np.max(W + b[:, np.newaxis], axis=0)
        b = b_new
    
    return b

def hoeffding_split_test(b: np.ndarray, threshold: float = 0.5) -> List[bool]:
    """Treats `b` as observed gains and applies the Hoeffding bound to decide which nodes have enough statistical evidence to become *candidate leaders*."""
    num_nodes = len(b)
    numcandidate_leaders = int(threshold * num_nodes)
    return [True if b[i] >= b[np.argsort(b)[::-1][numcandidate_leaders - 1]] else False for i in range(num_nodes)]

def simulated_annealing_acceptance(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance used in simulated annealing."""
    if delta_e < 0:
        return 1.0
    else:
        return math.exp(-delta_e / temperature)

def leader_election(b: np.ndarray, W: np.ndarray, node_strengths: List[float], temperature: float = 1.0) -> List[bool]:
    """Leader election algorithm using the Hoeffding bound and simulated annealing."""
    hoeffding_test = hoeffding_split_test(b)
    leader_nodes = [i for i, test in enumerate(hoeffding_test) if test]
    new_leader_nodes = []
    
    for node in leader_nodes:
        delta_e = b[node] - np.mean([b[i] for i in range(len(b)) if i not in leader_nodes])
        acceptance_prob = simulated_annealing_acceptance(delta_e, temperature)
        if random.random() < acceptance_prob:
            new_leader_nodes.append(node)
    
    return [True if i in new_leader_nodes else False for i in range(len(b))]

if __name__ == "__main__":
    text1 = "This is a sample text."
    text2 = "This is another sample text."
    features1 = stylometry_features(text1)
    features2 = stylometry_features(text2)
    W, node_strengths = build_weighted_graph([features1, features2])
    certainty_flags = curvature_to_certainty(W, node_strengths)
    b = tropical_broadcast(W, node_strengths)
    leader_nodes = leader_election(b, W, node_strengths)
    print(leader_nodes)