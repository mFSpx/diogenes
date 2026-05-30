# DARWIN HAMMER — match 46, survivor 0
# gen: 2
# parent_a: semantic_neighbors.py (gen0)
# parent_b: hybrid_pheromone_infotaxis_m3_s0.py (gen1)
# born: 2026-05-29T23:23:42Z

"""
This module integrates the semantic neighbor search from semantic_neighbors.py with the pheromone-based surface usage tracking 
from hybrid_pheromone_infotaxis_m3_s0.py. The mathematical bridge between the two lies in the idea of using pheromone signals 
as probabilities to inform the cosine similarity calculation, ultimately guiding the selection of semantic neighbors based on 
surface usage patterns.
"""

import json
import math
import numpy as np
import random
import sys
from pathlib import Path

class HybridEnclave:
    def __init__(self):
        self._enclave = {}
        self._pheromone_probabilities = {}

    def clear_enclave(self) -> None:
        self._enclave.clear()
        self._pheromone_probabilities.clear()

    def register_document(self, doc_id: str, vector: np.ndarray) -> None:
        self._enclave[doc_id] = vector

    def calculate_pheromone_probabilities(self, surface_key: str, limit: int, db_url: str):
        # Simulate pheromone database query
        pheromones = np.random.rand(limit)
        total = np.sum(pheromones)
        self._pheromone_probabilities[surface_key] = pheromones / total

    def _cos(self, a: np.ndarray, b: np.ndarray):
        den = np.linalg.norm(a) * np.linalg.norm(b)
        return 0.0 if den == 0 else np.dot(a, b) / den

    def semantic_neighbors(self, doc_id: str, k: int = 5) -> list[tuple[str, float]]:
        v = self._enclave[doc_id]
        return sorted(((d, self._cos(v, w)) for d, w in self._enclave.items() if d != doc_id), key=lambda x: (-x[1], x[0]))[:k]

    def pheromone_informed_neighbors(self, doc_id: str, surface_key: str, k: int = 5) -> list[tuple[str, float]]:
        v = self._enclave[doc_id]
        pheromone_probabilities = self._pheromone_probabilities.get(surface_key, np.ones(len(self._enclave)))
        return sorted(((d, self._cos(v, w) * pheromone_probabilities[i]) for i, (d, w) in enumerate(self._enclave.items()) if d != doc_id), key=lambda x: (-x[1], x[0]))[:k]

    def hybrid_search(self, doc_id: str, surface_key: str, k: int = 5) -> list[tuple[str, float]]:
        v = self._enclave[doc_id]
        pheromone_probabilities = self._pheromone_probabilities.get(surface_key, np.ones(len(self._enclave)))
        scores = [(d, self._cos(v, w) * pheromone_probabilities[i]) for i, (d, w) in enumerate(self._enclave.items()) if d != doc_id]
        return sorted(scores, key=lambda x: (-x[1], x[0]))[:k]

if __name__ == "__main__":
    enclave = HybridEnclave()
    enclave.register_document("doc1", np.array([1.0, 2.0, 3.0]))
    enclave.register_document("doc2", np.array([4.0, 5.0, 6.0]))
    enclave.calculate_pheromone_probabilities("surface_key", 2, "db_url")
    print(enclave.semantic_neighbors("doc1"))
    print(enclave.pheromone_informed_neighbors("doc1", "surface_key"))
    print(enclave.hybrid_search("doc1", "surface_key"))