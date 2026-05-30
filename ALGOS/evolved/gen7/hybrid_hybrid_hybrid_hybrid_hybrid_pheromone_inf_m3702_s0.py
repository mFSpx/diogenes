# DARWIN HAMMER — match 3702, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_fractional_hd_m2661_s1.py (gen6)
# parent_b: hybrid_pheromone_infotaxis_m3_s4.py (gen1)
# born: 2026-05-29T23:51:13Z

"""
This module fuses the governing equations of hybrid_hybrid_hybrid_rbf_su_hybrid_fractional_hd_m2661_s1.py (PARENT ALGORITHM A)
and hybrid_pheromone_infotaxis_m3_s4.py (PARENT ALGORITHM B). The mathematical bridge between the two parents lies in the 
similarity matrix computation and pheromone signal processing. We integrate the Radial Basis Function (RBF) kernel from PARENT 
A with the pheromone decay mechanism from PARENT B to create a novel hybrid algorithm.

The key interface is the use of a Gaussian RBF kernel to compute the similarity between feature vectors, and then applying 
pheromone-like decay to the similarity values to simulate a dynamic environment.

Imports: numpy, standard library, math, random, sys, pathlib
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from typing import Hashable, Sequence, List, Dict, Set, Tuple
from dataclasses import dataclass
import hashlib

Node = Hashable
Graph = Dict[Node, Set[Node]]
FeatureVec = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def similarity_matrix(features: Dict[Node, FeatureVec]) -> Tuple[np.ndarray, List[Node]]:
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = compute_phash(list(features[ni]))
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
            else:
                hj = compute_phash(list(features[nj]))
                d = hamming_distance(hi, hj)
                S[i, j] = 1.0 - d / 64.0
    return S, nodes

@dataclass(frozen=True)
class PheromoneEntry:
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: int
    created_at: float
    last_decay: float

    def age_seconds(self) -> float:
        return (self.last_decay - self.created_at)

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> 'PheromoneEntry':
        factor = self.decay_factor()
        return PheromoneEntry(
            self.surface_key,
            self.signal_kind,
            self.signal_value * factor,
            self.half_life_seconds,
            self.created_at,
            self.last_decay + 1.0
        )

def hybrid_similarity_matrix(features: Dict[Node, FeatureVec], 
                             half_life_seconds: int, 
                             time_step: float) -> Tuple[np.ndarray, List[Node]]:
    S, nodes = similarity_matrix(features)
    pheromone_entries = [PheromoneEntry(str(i), "similarity", S[i, i], half_life_seconds, 0.0, 0.0) for i in range(len(nodes))]
    
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            pheromone_entries[i] = pheromone_entries[i].apply_decay()
            S[i, j] *= pheromone_entries[i].signal_value
    
    return S, nodes

def generate_random_features(n: int, dim: int) -> Dict[int, FeatureVec]:
    features = {}
    for i in range(n):
        features[i] = [random.random() for _ in range(dim)]
    return features

def smoke_test():
    features = generate_random_features(10, 5)
    S, nodes = hybrid_similarity_matrix(features, half_life_seconds=10, time_step=1.0)
    print(S)

if __name__ == "__main__":
    smoke_test()