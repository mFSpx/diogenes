# DARWIN HAMMER — match 1510, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m1019_s1.py (gen5)
# parent_b: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s3.py (gen2)
# born: 2026-05-29T23:37:06Z

"""Hybrid Stylometry‑Curvature – Tropical Broadcast – Hoeffding Leader Election

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m1019_s1.py (stylometry → graph → Ollivier‑Ricci curvature → confidence)
- hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s3.py (tropical max‑plus broadcast, Hoeffding bound split test, simulated‑annealing acceptance)

Mathematical bridge:
1. A stylometry‑derived weighted graph G provides node strengths **w** and edge
   weights **W** (cosine similarity).  
2. Curvature κ_{ij}=1‑|w_i‑w_j|/(w_i+w_j+ε) is mapped linearly to a confidence
   value c_{ij}∈[0,10000].  The confidence matrix **C** is interpreted as a
   tropical (max‑plus) adjacency matrix.
3. Repeated tropical matrix multiplication propagates confidence over the graph,
   yielding a broadcast strength vector **b**.  Each component b_i is an
   empirical mean of bounded observations (0…1) and can be tested with the
   Hoeffding bound.
4. Nodes whose broadcast exceeds the Hoeffding threshold become *candidate
   leaders*.  A simulated‑annealing Metropolis step with temperature T decides
   whether the candidate set replaces the current leader set, using ΔE as the
   change in the number of leaders.

The module implements this pipeline in three core functions and a small
driver for smoke‑testing.
"""

import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Hashable, Mapping

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class CertaintyFlag:
    """Simple wrapper for epistemic confidence."""
    label: str
    confidence_bps: int  # basis points, 0‑10000

# ----------------------------------------------------------------------
# Parent A – stylometry → weighted graph → curvature → confidence
# ----------------------------------------------------------------------
def stylometry_features(text: str) -> Dict[str, int]:
    """Very coarse stylometry: count occurrences of a few word categories."""
    # For demonstration we use three categories: pronouns, conjunctions, verbs.
    pronouns = {"i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them"}
    conjunctions = {"and", "or", "but", "nor", "so", "yet", "for", "although"}
    verbs = {"is", "are", "was", "were", "be", "been", "being", "have", "has", "do", "does"}

    tokens = [t.lower().strip(".,!?;:()[]\"'") for t in text.split()]
    counts = Counter(tokens)

    return {
        "pronoun": sum(counts[w] for w in pronouns),
        "conj":   sum(counts[w] for w in conjunctions),
        "verb":   sum(counts[w] for w in verbs),
        "len":    len(tokens),
    }

def build_weighted_graph(features_list: List[Dict[str, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a symmetric similarity matrix W (cosine similarity) and node strengths.
    Returns:
        W        – (n, n) ndarray of edge weights in [0,1]
        strengths – (n,) ndarray of node total weights (L2 norm of feature vector)
    """
    n = len(features_list)
    # Convert dicts to vectors
    keys = sorted({k for d in features_list for k in d})
    mat = np.zeros((n, len(keys)), dtype=float)
    for i, d in enumerate(features_list):
        for j, k in enumerate(keys):
            mat[i, j] = d.get(k, 0)

    strengths = np.linalg.norm(mat, axis=1) + 1e-12  # avoid zero
    # Cosine similarity
    norm_mat = mat / strengths[:, None]
    W = np.clip(norm_mat @ norm_mat.T, 0.0, 1.0)
    np.fill_diagonal(W, 1.0)
    return W, strengths

def curvature_to_confidence(W: np.ndarray, strengths: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, Dict[Tuple[int, int], CertaintyFlag]]:
    """
    Compute Ollivier‑Ricci surrogate curvature on each edge and map to confidence.
    Returns:
        C          – (n, n) ndarray of confidence values in [0,10000]
        cert_dict  – mapping edge tuple -> CertaintyFlag
    """
    w_i = strengths[:, None]
    w_j = strengths[None, :]
    curvature = 1.0 - np.abs(w_i - w_j) / (w_i + w_j + eps)   # in [-1,1]
    confidence = ((curvature + 1.0) / 2.0 * 10000).astype(int)

    cert_dict = {}
    n = W.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            label = "high" if confidence[i, j] > 7000 else "low"
            cert_dict[(i, j)] = CertaintyFlag(label=label, confidence_bps=int(confidence[i, j]))
            cert_dict[(j, i)] = cert_dict[(i, j)]  # symmetric

    return confidence, cert_dict

# ----------------------------------------------------------------------
# Parent B – tropical max‑plus broadcast, Hoeffding test, simulated annealing
# ----------------------------------------------------------------------
def max_plus_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tropical (max‑plus) matrix multiplication: (A⊗B)_{ij}=max_k(A_{ik}+B_{kj})."""
    # Use broadcasting: compute A[:,:,None] + B[None,:,:] then max over axis=1
    return np.max(A[:, :, None] + B[None, :, :], axis=1)

def tropical_broadcast(confidence: np.ndarray, iterations: int = 5) -> np.ndarray:
    """
    Propagate confidence via repeated tropical multiplication.
    Returns a vector b where b_i = max_j (C_{ij}) after power iterations.
    """
    # Normalise confidence to a bounded range [0,1] for probabilistic interpretation
    C = confidence.astype(float) / 10000.0
    # Initialise broadcast vector as the max of each row (self‑influence)
    b = np.max(C, axis=1)
    for _ in range(iterations):
        # b_new_i = max_j (C_{ij} + b_j)
        b = np.max(C + b[None, :], axis=1)
    # Clip to [0,1]
    return np.clip(b, 0.0, 1.0)

def hoeffding_candidate_selection(broadcast: np.ndarray,
                                   n_samples: int = 30,
                                   delta: float = 0.05) -> List[int]:
    """
    Apply Hoeffding bound to each broadcast value.
    Treat each b_i as empirical mean of a bounded variable in [0,1].
    Nodes with b_i > 0.5 + ε are selected as candidate leaders.
    Returns list of node indices.
    """
    epsilon = math.sqrt(math.log(2.0 / delta) / (2.0 * n_samples))
    threshold = 0.5 + epsilon
    candidates = [i for i, val in enumerate(broadcast) if val > threshold]
    return candidates

def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance used in simulated annealing."""
    if delta_e < 0:
        return 1.0
    try:
        return math.exp(-delta_e / temperature)
    except OverflowError:
        return 0.0

def simulated_annealing_update(current_leaders: List[int],
                               candidates: List[int],
                               temperature: float) -> List[int]:
    """
    Decide whether to replace the current leader set with the candidate set.
    ΔE is defined as the absolute change in the number of leaders.
    """
    delta_e = abs(len(candidates) - len(current_leaders))
    prob = acceptance_probability(delta_e, temperature)
    if random.random() < prob:
        return candidates.copy()
    else:
        return current_leaders.copy()

# ----------------------------------------------------------------------
# Integrated hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_stylometry_tropical_election(texts: List[str],
                                        temperature: float = 1.0,
                                        n_samples: int = 30,
                                        delta: float = 0.05) -> List[int]:
    """
    Full hybrid algorithm:
    1. Extract stylometry features from each text.
    2. Build weighted graph and compute curvature‑derived confidence.
    3. Perform tropical broadcast to obtain per‑node strengths.
    4. Use Hoeffding bound to select candidate leaders.
    5. Apply simulated‑annealing acceptance to obtain final leader indices.
    Returns the list of node indices that are elected as leaders.
    """
    # Step 1
    feats = [stylometry_features(t) for t in texts]

    # Step 2
    W, strengths = build_weighted_graph(feats)
    confidence, _ = curvature_to_confidence(W, strengths)

    # Step 3
    broadcast = tropical_broadcast(confidence, iterations=5)

    # Step 4
    candidates = hoeffding_candidate_selection(broadcast,
                                                n_samples=n_samples,
                                                delta=delta)

    # Step 5
    leaders = simulated_annealing_update([], candidates, temperature)
    return leaders

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "I am happy and you are sad.",
        "She is running but he is walking.",
        "We have been there and they have not.",
        "The quick brown fox jumps over the lazy dog.",
        "Although it was raining, we went out."
    ]

    # Run the hybrid algorithm with a modest temperature schedule
    temp = 1.0
    leaders = hybrid_stylometry_tropical_election(sample_texts,
                                                  temperature=temp,
                                                  n_samples=40,
                                                  delta=0.1)
    print(" elected leader node indices:", leaders)
    for idx in leaders:
        print(f" Leader {idx}: \"{sample_texts[idx]}\"")