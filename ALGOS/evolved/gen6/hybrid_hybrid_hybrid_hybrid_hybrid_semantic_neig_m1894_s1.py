# DARWIN HAMMER — match 1894, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m626_s3.py (gen5)
# parent_b: hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s2.py (gen2)
# born: 2026-05-29T23:39:28Z

"""
Hybrid Algorithm: Fusing DARWIN HAMMER (match 626, survivor 3) and 
                 DARWIN HAMMER (match 46, survivor 2)

This hybrid algorithm integrates the core topologies of two parent algorithms: 
- hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m626_s3.py (gen: 5) 
  and hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s2.py (gen: 2).

The mathematical bridge between the two parents lies in interpreting the 
SSIM (Structural Similarity Index Measure) values from parent A as 
raw pheromone signals, similar to parent B. These SSIM values are then 
used to compute a probability distribution over the semantic neighbors, 
which in turn modulates the expected entropy of the neighborhood.

The governing equations of both parents are fused into a single unified 
system, enabling the hybrid algorithm to leverage the strengths of both 
parents in a cohesive manner.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# In-memory semantic enclave
_ENCLAVE: dict[str, np.ndarray] = {}

def clear_enclave() -> None:
    """Remove all registered documents."""
    _ENCLAVE.clear()

def register_document(doc_id: str, vector: list[float]) -> None:
    """Store a document vector as a NumPy array for fast linear algebra."""
    _ENCLAVE[doc_id] = np.array(vector, dtype=float)

def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)

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

def hybrid_ssim_pheromone(doc_id: str, k: int = 5) -> list[tuple[str, float]]:
    """Compute SSIM values and return the k most similar documents with pheromone modulation."""
    sims = semantic_neighbors(doc_id, k)
    ssim_values = []
    for neighbor, sim in sims:
        vector1 = _ENCLAVE[doc_id].tolist()
        vector2 = _ENCLAVE[neighbor].tolist()
        ssim = compute_ssim(vector1, vector2)
        ssim_values.append(ssim)
    # Normalize SSIM values to obtain a probability distribution
    ssim_prob_dist = [ssim / sum(ssim_values) for ssim in ssim_values]
    # Modulate the probability distribution with pheromone signals
    modulated_prob_dist = [prob * (1 - math.log2(len(ssim_values))) for prob in ssim_prob_dist]
    return [(neighbor, prob) for (neighbor, _), prob in zip(sims, modulated_prob_dist)]

def expected_entropy(doc_id: str, k: int = 5) -> float:
    """Compute the expected entropy of the neighborhood."""
    sims = hybrid_ssim_pheromone(doc_id, k)
    probabilities = [prob for _, prob in sims]
    entropy = -sum([prob * math.log2(prob) for prob in probabilities])
    return entropy

if __name__ == "__main__":
    clear_enclave()
    register_document("doc1", [0.2, 0.5, 0.3, 0.7, 0.1])
    register_document("doc2", [0.1, 0.7, 0.2, 0.5, 0.3])
    register_document("doc3", [0.3, 0.2, 0.7, 0.1, 0.5])
    print(hybrid_ssim_pheromone("doc1"))
    print(expected_entropy("doc1"))