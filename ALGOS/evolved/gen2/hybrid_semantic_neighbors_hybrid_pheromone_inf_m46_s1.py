# DARWIN HAMMER — match 46, survivor 1
# gen: 2
# parent_a: semantic_neighbors.py (gen0)
# parent_b: hybrid_pheromone_infotaxis_m3_s0.py (gen1)
# born: 2026-05-29T23:23:42Z

#!/usr/bin/env python3
"""Hybrid of semantic_neighbors.py and hybrid_pheromone_infotaxis_m3_s0.py: This module integrates the semantic neighborhood 
search from semantic_neighbors.py with the pheromone-based surface usage tracking and entropy-based action selection from 
hybrid_pheromone_infotaxis_m3_s0.py. The mathematical bridge between the two lies in the idea of using pheromone signals 
as probabilities to inform the semantic neighborhood search, ultimately guiding the selection of neighbors based on surface 
usage patterns."""

import math
import numpy as np
import random
import sys
from pathlib import Path

def _cos(a, b):
    den = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
    return 0.0 if den == 0 else sum(x*y for x, y in zip(a, b)) / den

def pheromone_probabilities(pheromones):
    total = sum(pheromones)
    return [p / total for p in pheromones]

def entropy(probabilities, eps=1e-12):
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p / total) * math.log(max(p / total, eps)) for p in probabilities if p > 0)

def expected_entropy(p_hit, hit_state, miss_state):
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)

class HybridEnclave:
    def __init__(self):
        self.enclave = {}
        self.pheromones = {}

    def clear_enclave(self):
        self.enclave.clear()
        self.pheromones.clear()

    def register_document(self, doc_id, vector, pheromones):
        self.enclave[doc_id] = vector
        self.pheromones[doc_id] = pheromones

    def semantic_neighbors(self, doc_id, k=5):
        v = self.enclave[doc_id]
        pheromones = self.pheromones[doc_id]
        probabilities = pheromone_probabilities(pheromones)
        entropy_values = []
        for d, w in self.enclave.items():
            if d != doc_id:
                similarity = _cos(v, w)
                pheromone_weight = probabilities[self.enclave.keys().index(d)]
                entropy_values.append((d, similarity * pheromone_weight))
        return sorted(entropy_values, key=lambda x: (-x[1], x[0]))[:k]

    def best_action(self, actions, doc_id, k=5):
        neighbors = self.semantic_neighbors(doc_id, k)
        best_action = min(actions, key=lambda a: (expected_entropy(a[0], a[1], a[2]), a))
        return best_action

if __name__ == "__main__":
    enclave = HybridEnclave()
    enclave.register_document("doc1", [0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    enclave.register_document("doc2", [0.7, 0.8, 0.9], [0.1, 0.2, 0.3])
    actions = {
        'action1': (0.5, [0.2, 0.3, 0.5], [0.1, 0.4, 0.5]),
        'action2': (0.7, [0.6, 0.2, 0.2], [0.3, 0.6, 0.1]),
    }
    neighbors = enclave.semantic_neighbors("doc1")
    best = enclave.best_action(actions, "doc1")
    print("Neighbors:", neighbors)
    print("Best action:", best)