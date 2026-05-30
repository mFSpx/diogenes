# DARWIN HAMMER — match 5255, survivor 0
# gen: 6
# parent_a: hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m2611_s0.py (gen5)
# born: 2026-05-30T00:01:00Z

"""
Hybrid Algorithm: Semantic-Pheromone Infotaxis meets Hybrid Workshare Allocation
Parents:
- hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s2.py (semantic neighbors + pheromone infotaxis)
- hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m2611_s0.py (hybrid workshare allocation)

The mathematical bridge between the two algorithms is formed by interpreting the 
cosine similarities between document vectors as raw pheromone signals, which are 
then used to modulate the labeling functions in the workshare allocation process. 
The resulting hybrid algorithm integrates the semantic search and pheromone-based 
decision-making of the first parent with the labeling and allocation mechanisms 
of the second parent.

The core interface is formed by the `_cosine` function from the semantic 
neighbors module and the `compute_dhash` function from the workshare allocation 
module. The cosine similarities are used to compute a weighted Hamming distance 
between the perceptual hashes of the workshare lanes, which informs the allocation 
process.

"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
from collections import defaultdict

_ENCLAVE: dict[str, np.ndarray] = {}

def clear_enclave() -> None:
    """Remove all registered documents."""
    _ENCLAVE.clear()

def register_document(doc_id: str, vector: list[float]) -> None:
    """Store a document vector as a NumPy array for fast linear algebra."""
    _ENCLAVE[doc_id] = np.array(vector, dtype=float)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0.0:
        return 0.0
    return float(np.dot(a, b) / den)

def semantic_neighbors(doc_id: str, k: int = 5) -> list[tuple[str, float]]:
    """Return the k most similar documents (excluding the query)."""
    v = _ENCLAVE[doc_id]
    sims = [(d, _cosine(v, w)) for d, w in _ENCLAVE.items() if d != doc_id]
    sims.sort(key=lambda x: (-x[1], x[0]))
    return sims[:k]

@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int

@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float

def compute_dhash(values: list[float]) -> int:
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: list[float]) -> int:
    if not values:
        return 0
    dhash = compute_dhash(values)
    return dhash

def hybrid_allocation(doc_id: str, k: int = 5) -> List[ProbabilisticLabel]:
    neighbors = semantic_neighbors(doc_id, k)
    labels = []
    for neighbor, similarity in neighbors:
        vector = _ENCLAVE[neighbor]
        dhash = compute_phash(vector.tolist())
        label = dhash % 2  # Simple labeling function
        confidence = similarity
        labels.append(ProbabilisticLabel(neighbor, label, confidence))
    return labels

def weighted_hamming_distance(labels: List[ProbabilisticLabel]) -> float:
    distance = 0.0
    for label in labels:
        distance += label.confidence * (label.label ^ compute_phash(_ENCLAVE[label.doc_id].tolist()))
    return distance / len(labels)

def select_action(doc_id: str, k: int = 5) -> str:
    labels = hybrid_allocation(doc_id, k)
    distance = weighted_hamming_distance(labels)
    # Select action based on minimum distance
    return min(labels, key=lambda x: abs(x.confidence - distance)).doc_id

if __name__ == "__main__":
    clear_enclave()
    register_document("doc1", [1.0, 2.0, 3.0])
    register_document("doc2", [4.0, 5.0, 6.0])
    register_document("doc3", [7.0, 8.0, 9.0])
    print(select_action("doc1"))