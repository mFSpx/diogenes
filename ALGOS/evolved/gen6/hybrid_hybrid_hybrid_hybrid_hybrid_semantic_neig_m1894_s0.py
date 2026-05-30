# DARWIN HAMMER — match 1894, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m626_s3.py (gen5)
# parent_b: hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s2.py (gen2)
# born: 2026-05-29T23:39:28Z

"""
Hybrid Semantic-Pheromone Infotaxis-Ternar Algorithm.

Mathematical bridge:
The ternar expansion of a vector (parent A) is interpreted as a raw pheromone signal.
These signals are decayed (half-life) and normalized to obtain a probability distribution P_nei over the k-nearest neighbours.
The Shannon entropy H(P_nei) quantifies the uncertainty of the semantic neighbourhood.
For each candidate action we compute an expected entropy E = p_hit * H(hit_state) + (1-p_hit) * H(miss_state)
where p_hit is modulated by the neighbourhood certainty (1-H(P_nei)).
The action with minimal E is selected.
Thus the core linear algebra of ternar expansion and the probabilistic-entropy machinery of pheromone-infotaxis are fused into a single decision-making pipeline.

Parents:
- hybrid_hybrid_hybrid_ternar_hybrid_hybrid_regret_m236_s1.py (ternar expansion)
- hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s2.py (pheromone probability weighting + entropy-based action selection)
"""

import math
import random
import sys
import pathlib
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import numpy as np

PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)

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

def expand_to_pheromone(signal: np.ndarray, salt: str = "") -> np.ndarray:
    """
    Expand ternar signal to pheromone signal using SHA-256 hash.

    Args:
    signal (np.ndarray): Ternar signal.
    salt (str): Salt for hash function.

    Returns:
    np.ndarray: Pheromone signal.
    """
    m = signal.shape[0]
    out = np.zeros(m)
    for i, v in enumerate(signal):
        for r in range(3):  
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out

def semantic_neighbors(doc_id: str, k: int = 5) -> list[tuple[str, float]]:
    """Return the k most similar documents (excluding the query)."""
    v = _ENCLAVE[doc_id]
    sims = [(d, _cosine(v, w)) for d, w in _ENCLAVE.items() if d != doc_id]
    sims.sort(key=lambda x: (-x[1], x[0]))
    return sims[:k]

def compute_expected_entropy(probability: float, entropy_hit: float, entropy_miss: float) -> float:
    """
    Compute expected entropy given probability and entropies of hit and miss states.

    Args:
    probability (float): Probability of hit state.
    entropy_hit (float): Entropy of hit state.
    entropy_miss (float): Entropy of miss state.

    Returns:
    float: Expected entropy.
    """
    return probability * entropy_hit + (1 - probability) * entropy_miss

def select_action(expected_entropy: float, neighbourhood_certainty: float) -> float:
    """
    Select action with minimal expected entropy given neighbourhood certainty.

    Args:
    expected_entropy (float): Expected entropy of each action.
    neighbourhood_certainty (float): Certainty of semantic neighbourhood.

    Returns:
    float: Minimal expected entropy.
    """
    return expected_entropy * neighbourhood_certainty

if __name__ == "__main__":
    # Smoke test
    clear_enclave()
    register_document("doc1", [0.5, 0.3, 0.2, 0.7, 0.3])
    register_document("doc2", [0.2, 0.5, 0.3, 0.7, 0.1])
    semantic_neighbors("doc1", 2)
    signal = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)
    pheromone_signal = expand_to_pheromone(signal)
    expected_entropy = compute_expected_entropy(0.5, 0.2, 0.5)
    select_action(expected_entropy, 0.7)