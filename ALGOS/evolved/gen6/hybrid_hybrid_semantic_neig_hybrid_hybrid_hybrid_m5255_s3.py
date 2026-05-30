# DARWIN HAMMER — match 5255, survivor 3
# gen: 6
# parent_a: hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m2611_s0.py (gen5)
# born: 2026-05-30T00:01:00Z

import math
import random
import sys
import pathlib
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np

# ----------------------------------------------------------------------
# In‑memory semantic enclave (parent A)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# In‑memory pheromone store (simplified parent B)
# ----------------------------------------------------------------------

def compute_dhash(values: list[float]) -> int:
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: list[float]) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return int(values[0] * 2**31) # To prevent hash collision
    hash = 0
    for val in values:
        hash = hash * 31 + int(val * 2**31) # prevent potential integer overflow 
    return hash

def hamming_distance(a: int, b: int) -> int:
    """Compute the Hamming distance between two integers."""
    return bin(a ^ b).count('1')

# ----------------------------------------------------------------------
# Hybrid algorithm: semantic-pheromone-infotaxis workshare allocation
# ----------------------------------------------------------------------

def allocate_workshare(doc_id: str, k: int = 5, llm_units: float = 1.0, llm_share_pct: float = 0.5, proof_required: bool = True) -> dict:
    """Allocate workshare units using semantic-pheromone-infotaxis."""
    # Compute semantic neighbors
    neighbors = semantic_neighbors(doc_id, k)
    
    # Compute pheromone probabilities
    pheromones = [neighbor[1] for neighbor in neighbors]
    probabilities = [pheromone / sum(pheromones) for pheromone in pheromones]
    
    # Compute entropy
    entropy = -sum([p * math.log2(p) if p != 0 else 0 for p in probabilities])
    
    # Compute expected entropy
    expected_entropy = sum([p * entropy for p in probabilities])
    
    # Compute perceptual hashes
    hashes = [compute_phash([neighbor[1] for neighbor in neighbors])]
    
    # Compute Hamming distances
    distances = [hamming_distance(hashes[0], compute_phash([neighbor[1]])) for neighbor in neighbors]
    
    # Allocate workshare units
    allocation = {neighbor[0]: llm_units * llm_share_pct / (1 + distances[i]) for i, neighbor in enumerate(neighbors)}
    
    return allocation

def allocate_workshare_with_certainty(doc_id: str, k: int = 5, llm_units: float = 1.0, llm_share_pct: float = 0.5, proof_required: bool = True) -> dict:
    """Allocate workshare units using semantic-pheromone-infotaxis with certainty."""
    # Compute semantic neighbors
    neighbors = semantic_neighbors(doc_id, k)
    
    # Compute pheromone probabilities
    pheromones = [neighbor[1] for neighbor in neighbors]
    probabilities = [pheromone / sum(pheromones) for pheromone in pheromones]
    
    # Compute entropy
    entropy = -sum([p * math.log2(p) if p != 0 else 0 for p in probabilities])
    
    # Compute expected entropy
    expected_entropy = sum([p * entropy for p in probabilities])
    
    # Compute perceptual hashes
    hashes = [compute_phash([neighbor[1] for neighbor in neighbors])]
    
    # Compute Hamming distances
    distances = [hamming_distance(hashes[0], compute_phash([neighbor[1]])) for neighbor in neighbors]
    
    # Allocate workshare units with certainty
    allocation = {neighbor[0]: llm_units * llm_share_pct / (1 + distances[i]) * (1 - entropy) for i, neighbor in enumerate(neighbors)}
    
    return allocation

def allocate_workshare_with_certainty_and_labeling_function(doc_id: str, k: int = 5, llm_units: float = 1.0, llm_share_pct: float = 0.5, proof_required: bool = True) -> dict:
    """Allocate workshare units using semantic-pheromone-infotaxis with certainty and labeling function."""
    # Compute semantic neighbors
    neighbors = semantic_neighbors(doc_id, k)
    
    # Compute pheromone probabilities
    pheromones = [neighbor[1] for neighbor in neighbors]
    probabilities = [pheromone / sum(pheromones) for pheromone in pheromones]
    
    # Compute entropy
    entropy = -sum([p * math.log2(p) if p != 0 else 0 for p in probabilities])
    
    # Compute expected entropy
    expected_entropy = sum([p * entropy for p in probabilities])
    
    # Compute perceptual hashes
    hashes = [compute_phash([neighbor[1] for neighbor in neighbors])]
    
    # Compute Hamming distances
    distances = [hamming_distance(hashes[0], compute_phash([neighbor[1]])) for neighbor in neighbors]
    
    # Allocate workshare units with certainty and labeling function
    allocation = {neighbor[0]: llm_units * llm_share_pct / (1 + distances[i]) * (1 - entropy) * (1 + 0.1 * expected_entropy) for i, neighbor in enumerate(neighbors)}
    
    return allocation

if __name__ == "__main__":
    # Register a document
    register_document("doc1", [1.0, 2.0, 3.0])
    register_document("doc2", [2.0, 3.0, 4.0])
    register_document("doc3", [3.0, 4.0, 5.0])
    
    # Allocate workshare units
    allocation = allocate_workshare("doc1")
    print(allocation)
    
    # Allocate workshare units with certainty
    allocation_certainty = allocate_workshare_with_certainty("doc1")
    print(allocation_certainty)
    
    # Allocate workshare units with certainty and labeling function
    allocation_certainty_labeling_function = allocate_workshare_with_certainty_and_labeling_function("doc1")
    print(allocation_certainty_labeling_function)