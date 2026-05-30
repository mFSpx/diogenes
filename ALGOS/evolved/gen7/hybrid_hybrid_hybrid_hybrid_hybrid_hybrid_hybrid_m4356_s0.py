# DARWIN HAMMER — match 4356, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_semantic_neig_m1894_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1230_s0.py (gen6)
# born: 2026-05-29T23:55:07Z

"""Hybrid Semantic‑Pheromone‑SSIM Algorithm

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_semantic_neig_m1894_s0.py (ternar expansion → pheromone, neighbourhood entropy)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1230_s0.py (hash‑based vector expansion, Gini coefficient, SSIM similarity)

Mathematical bridge:
Both parents map a low‑dimensional signal to a high‑dimensional “pheromone” vector.
Parent A uses a ternar expansion that is interpreted as raw pheromone; Parent B hashes each element into a sparse high‑dimensional space.
We therefore reuse the hash‑based expansion for the ternar signal, obtaining a common pheromone space.
In this space we can (i) normalise the vector to a probability distribution and compute Shannon entropy (A), (ii) compute the Gini coefficient of the raw counts (B), and (iii) compare two pheromone vectors with the Structural Similarity Index (SSIM) (B).  
The hybrid decision metric combines these three quantities into a single score used for action selection.
"""

import math
import random
import sys
import pathlib
import hashlib
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared document enclave (Parent A)
# ----------------------------------------------------------------------
_PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)
_ENCLAVE: dict[str, np.ndarray] = {}

def clear_enclave() -> None:
    """Remove all registered documents."""
    _ENCLAVE.clear()

def register_document(doc_id: str, vector: List[float]) -> None:
    """Store a document vector as a NumPy array for fast linear algebra."""
    _ENCLAVE[doc_id] = np.array(vector, dtype=float)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0.0:
        return 0.0
    return float(np.dot(a, b) / den)

# ----------------------------------------------------------------------
# Hash‑based expansion (Parent B)
# ----------------------------------------------------------------------
def hash_expand(v: List[float], dim: int = 128) -> np.ndarray:
    """
    Expand an arbitrary low‑dimensional vector into a sparse high‑dimensional
    vector using a deterministic hash of (index, value).

    Args:
        v: Input vector (list of scalars).
        dim: Target dimensionality.

    Returns:
        A NumPy array of shape (dim,) with integer counts.
    """
    e = np.zeros(dim, dtype=np.int32)
    for i, val in enumerate(v):
        # deterministic hash, reduced modulo dim
        h = hash((i, float(val))) % dim
        e[h] += 1
    return e

# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def ternar_to_pheromone(signal: np.ndarray, dim: int = 128, salt: str = "") -> np.ndarray:
    """
    Convert a ternar signal (elements in {-1, 0, 1}) into a pheromone vector.
    The conversion re‑uses the hash‑based expansion of Parent B, thus providing
    a common high‑dimensional representation.

    Args:
        signal: 1‑D array of ternar values.
        dim: Target dimensionality.
        salt: Optional string mixed into the hash to diversify the mapping.

    Returns:
        Integer count vector of length ``dim``.
    """
    e = np.zeros(dim, dtype=np.int32)
    for i, v in enumerate(signal):
        for r in range(3):  # ternar has three possible symbols
            payload = f"{salt}:{i}:{r}"
            h = int(hashlib.sha256(payload.encode()).hexdigest(), 16) % dim
            # Encode the ternar value: -1 → 0, 0 → 1, 1 → 2
            if int(v) == -1 and r == 0:
                e[h] += 1
            elif int(v) == 0 and r == 1:
                e[h] += 1
            elif int(v) == 1 and r == 2:
                e[h] += 1
    return e

def shannon_entropy(p: np.ndarray) -> float:
    """Compute Shannon entropy of a probability vector (base‑e)."""
    if p.size == 0:
        return 0.0
    # Guard against log(0)
    mask = p > 0
    return -float(np.sum(p[mask] * np.log(p[mask])))

def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient as defined in Parent B."""
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values.astype(float))
    n = values.size
    cumulative = np.cumsum(sorted_vals)
    if cumulative[-1] == 0:
        return 0.0
    return float(((np.arange(1, n + 1) / n) - (cumulative / cumulative[-1])).sum())

def compute_ssim(x: np.ndarray, y: np.ndarray,
                 dynamic_range: float = 1.0,
                 k1: float = 0.01,
                 k2: float = 0.03) -> float:
    """Structural Similarity Index for two 1‑D signals (Parent B)."""
    if x.shape != y.shape:
        raise ValueError("SSIM inputs must have the same shape")
    mx = float(x.mean())
    my = float(y.mean())
    vx = float(((x - mx) ** 2).mean())
    vy = float(((y - my) ** 2).mean())
    cov = float(((x - mx) * (y - my)).mean())

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx ** 2 + my ** 2 + c1) * (vx + vy + c2)
    return float(numerator / denominator) if denominator != 0 else 0.0

def neighbour_metrics(query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, float, float]:
    """
    For a query vector, find the k‑nearest neighbours in the enclave,
    return the normalised probability distribution, its entropy, and the Gini
    coefficient of the raw cosine similarity scores.

    Returns:
        P_nei: probability distribution (softmax of similarities)
        H: Shannon entropy of P_nei
        G: Gini coefficient of the raw similarity scores
    """
    if not _ENCLAVE:
        raise RuntimeError("Enclave is empty – register some documents first.")
    sims = []
    for vec in _ENCLAVE.values():
        sims.append(_cosine(query, vec))
    sims = np.array(sims, dtype=np.float64)

    # Take top‑k similarities
    if k < len(sims):
        idx = np.argpartition(-sims, k)[:k]
        top_sims = sims[idx]
    else:
        top_sims = sims

    # Softmax to obtain a probability distribution
    max_sim = np.max(top_sims)
    exp_vals = np.exp(top_sims - max_sim)  # stability
    P_nei = exp_vals / exp_vals.sum()

    H = shannon_entropy(P_nei)
    G = gini_coefficient(top_sims)

    return P_nei, H, G

def hybrid_score(candidate: np.ndarray,
                 prototype: np.ndarray,
                 dim: int = 128,
                 k: int = 5,
                 w_entropy: float = 1.0,
                 w_gini: float = 1.0,
                 w_ssim: float = -1.0) -> float:
    """
    Compute a composite score for a candidate vector.

    The score combines:
        – Entropy of the semantic neighbourhood (Parent A)
        – Gini coefficient of the neighbourhood similarity distribution (Parent B)
        – Negative SSIM between candidate and prototype pheromone vectors (Parent B)

    Lower scores are preferred.

    Args:
        candidate: Candidate vector.
        prototype: Prototype vector against which SSIM is evaluated.
        dim: Dimensionality of the pheromone space.
        k: Number of neighbours for entropy/Gini.
        w_entropy, w_gini, w_ssim: Weights for the three terms.

    Returns:
        Composite float score.
    """
    # 1. neighbourhood metrics (entropy, gini)
    _, H, G = neighbour_metrics(candidate, k=k)

    # 2. pheromone vectors via hash expansion (shared space)
    pher_candidate = hash_expand(candidate.tolist(), dim=dim)
    pher_prototype = hash_expand(prototype.tolist(), dim=dim)

    # 3. SSIM similarity
    ssim_val = compute_ssim(pher_candidate.astype(float), pher_prototype.astype(float))

    # Composite linear combination
    score = w_entropy * H + w_gini * G + w_ssim * ssim_val
    return score

def select_best_candidate(candidates: List[np.ndarray],
                          prototype: np.ndarray,
                          dim: int = 128,
                          k: int = 5) -> Tuple[int, float]:
    """
    Evaluate a list of candidate vectors and return the index of the best
    candidate together with its score.

    Returns:
        (best_index, best_score)
    """
    best_idx = -1
    best_score = math.inf
    for i, cand in enumerate(candidates):
        sc = hybrid_score(cand, prototype, dim=dim, k=k)
        if sc < best_score:
            best_score = sc
            best_idx = i
    return best_idx, best_score

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Populate enclave with random document vectors
    clear_enclave()
    for i in range(10):
        vec = np.random.rand(5)  # 5‑dimensional semantic vectors
        register_document(f"doc_{i}", vec.tolist())

    # Prototype (could be a goal state)
    prototype_vec = np.random.rand(5)

    # Generate a few candidate vectors
    candidates = [np.random.rand(5) for _ in range(4)]

    # Run the hybrid decision routine
    best_idx, best_score = select_best_candidate(candidates, prototype_vec)

    print(f"Best candidate index: {best_idx}")
    print(f"Score: {best_score:.6f}")
    print(f"Chosen vector: {candidates[best_idx]}")