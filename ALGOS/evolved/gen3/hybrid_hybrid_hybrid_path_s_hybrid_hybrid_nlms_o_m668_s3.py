# DARWIN HAMMER — match 668, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_path_signatur_hybrid_krampus_brain_m25_s3.py (gen2)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s5.py (gen2)
# born: 2026-05-29T23:30:21Z

"""Hybrid Path‑Signature + NLMS‑Graph Fusion

Parents:
- A: hybrid_path_signature_kan … – builds a lead‑lag augmented path from
  deterministic master‑vectors extracted from texts and computes level‑1
  and level‑2 iterated‑integral signatures.
- B: hybrid_nlms_omni_cha … – learns a weight vector `w` with the Normalised
  Least‑Mean‑Squares (NLMS) rule and uses `w` to weight edges of a similarity
  graph before solving a Minimum‑Cost Spanning Tree (MST).

Mathematical Bridge
-------------------
Let `x_i ∈ ℝ^D` be the master‑vector of text `t_i` (Parent B).  
The ordered set `X = (x_1,…,x_N)` defines a discrete path.  
Applying the lead‑lag transform `L` yields an augmented path `X̃ = L(X)`.  
From `X̃` we compute the **signature** `σ = [σ¹, σ²]` where  


σ¹ = Σ_k Δ_k                         (level‑1, Δ_k = X̃_{k+1}−X̃_k)
σ² = Σ_{k<ℓ} Δ_k ⊗ Δ_ℓ               (level‑2, outer product)


`σ` is a feature vector for each node.  
We now treat `σ_i` as the input vector `x_i` of an NLMS adaptive filter.
Given a desired importance `d_i` (here the ℓ₂‑norm of `σ_i`) the NLMS rule
updates a global weight vector `w ∈ ℝ^{dim(σ)}`:


y_i   = w·σ_i
e_i   = d_i − y_i
w←w + μ * e_i * σ_i / (‖σ_i‖² + ε)


The learned `w` is finally used to define edge‑costs of the complete graph
on the texts:


c_{ij} = w·|σ_i − σ_j|


A Minimum‑Cost Spanning Tree on `c_{ij}` yields a structure that reflects
both the geometric information captured by path signatures and the
data‑driven relevance learned by NLMS.

The module below implements this unified pipeline with three public
functions demonstrating the hybrid operation.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent B – deterministic master‑vector extractor
# ----------------------------------------------------------------------
_FEATURE_KEYS = [
    "operator_visceral_ratio", "operator_tech_ratio",
    "operator_legal_osint_ratio", "operator_ledger_density",
    "operator_recursion_score", "operator_directive_ratio",
    "operator_target_density", "psyche_forensic_shield_ratio",
    "psyche_poetic_entropy", "psyche_dissociative_index",
    "psyche_wrath_velocity", "resilience_bureaucratic_weaponization_index",
    "resilience_resource_exhaustion_metric", "resilience_swar"
]

def extract_master_vector(text: str) -> np.ndarray:
    """Deterministic pseudo‑random feature vector from a string."""
    rnd = random.Random(hash(text))
    vec = np.array([rnd.random() for _ in _FEATURE_KEYS], dtype=np.float64)
    # Normalise to unit ℓ2 norm for stability
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


# ----------------------------------------------------------------------
# Parent A – lead‑lag transform and level‑1/2 signature
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag augmentation.

    Given an (N, D) array `path`, returns an (2N‑1, 2D) array where each
    step contains the current point duplicated (lead) and the next point
    (lag).  This construction preserves causality and yields a path that
    lives in a higher‑dimensional space.
    """
    N, D = path.shape
    augmented = []
    for i in range(N - 1):
        lead = np.concatenate([path[i], path[i]])
        lag  = np.concatenate([path[i], path[i + 1]])
        augmented.append(lead)
        augmented.append(lag)
    # Append final lead (last point duplicated) to close the path
    augmented.append(np.concatenate([path[-1], path[-1]]))
    return np.stack(augmented, axis=0)  # shape (2N‑1, 2D)


def compute_signature(aug_path: np.ndarray) -> np.ndarray:
    """
    Compute level‑1 and level‑2 signatures of a discrete augmented path.

    Returns a 1‑D vector: [σ¹ (flattened), σ² (flattened)].
    """
    increments = np.diff(aug_path, axis=0)                # (M‑1, 2D)
    sigma1 = increments.sum(axis=0)                      # (2D,)

    # Level‑2: sum_{i<j} Δ_i ⊗ Δ_j  →  outer products accumulated
    M = increments.shape[0]
    dim = increments.shape[1]
    sigma2 = np.zeros((dim, dim), dtype=np.float64)
    for i in range(M):
        di = increments[i][:, None]                      # (dim,1)
        # contributions of di with all later increments
        later = increments[i + 1:]                       # (M-i-1, dim)
        sigma2 += di @ later                               # broadcasting outer product
    sigma2_flat = sigma2.ravel()
    return np.concatenate([sigma1, sigma2_flat])


# ----------------------------------------------------------------------
# NLMS core (Parent A)
# ----------------------------------------------------------------------
def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS adaptation step.

    Returns (new_weights, error) where error = target - y.
    """
    y = predict(weights, x)
    error = target - y
    norm_sq = np.dot(x, x) + eps
    new_weights = weights + (mu * error / norm_sq) * x
    return new_weights, error


# ----------------------------------------------------------------------
# Graph utilities (Parent B)
# ----------------------------------------------------------------------
def build_edge_cost_matrix(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute a symmetric cost matrix C where
    C_{ij} = w·|f_i - f_j| .
    """
    diff = np.abs(features[:, None, :] - features[None, :, :])   # (N,N,D)
    cost = np.tensordot(diff, weights, axes=([2], [0]))          # (N,N)
    # Ensure zero diagonal
    np.fill_diagonal(cost, 0.0)
    return cost


def prim_mst(cost_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Simple Prim's algorithm returning the list of edges (i, j, cost)
    of the Minimum‑Cost Spanning Tree for a dense undirected graph.
    """
    N = cost_matrix.shape[0]
    visited = [False] * N
    visited[0] = True
    edges = []
    while len(edges) < N - 1:
        min_cost = math.inf
        min_edge = (-1, -1)
        for i in range(N):
            if not visited[i]:
                continue
            for j in range(N):
                if visited[j]:
                    continue
                c = cost_matrix[i, j]
                if c < min_cost:
                    min_cost = c
                    min_edge = (i, j)
        i, j = min_edge
        visited[j] = True
        edges.append((i, j, min_cost))
    return edges


# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_signature_features(texts: List[str]) -> np.ndarray:
    """
    Convert a list of texts into signature feature vectors.
    Each text is first mapped to a master vector, then a lead‑lag path
    (of length 2) is built, and finally its signature is computed.
    The result is an (N, D_sig) array.
    """
    signatures = []
    for txt in texts:
        vec = extract_master_vector(txt)                # (D,)
        # Build a trivial 2‑point path: start at zero, then the vector
        path = np.vstack([np.zeros_like(vec), vec])      # (2, D)
        aug = lead_lag_transform(path)                   # (3, 2D)
        sig = compute_signature(aug)                     # (2D + (2D)^2,)
        signatures.append(sig)
    return np.stack(signatures, axis=0)


def train_nlms_weights(
    features: np.ndarray,
    mu: float = 0.5,
    epochs: int = 5,
) -> np.ndarray:
    """
    Learn a global weight vector `w` using NLMS.
    Target for each node is the ℓ2‑norm of its feature vector
    (a simple proxy for “importance”).
    """
    dim = features.shape[1]
    w = np.zeros(dim, dtype=np.float64)
    for _ in range(epochs):
        for x in features:
            target = np.linalg.norm(x)                     # scalar desired output
            w, _ = nlms_update(w, x, target, mu=mu)
    return w


def hybrid_pipeline(
    texts: List[str],
    mu: float = 0.5,
    epochs: int = 5,
) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
    """
    End‑to‑end hybrid operation:
    1. Extract signature features for each text.
    2. Train NLMS weights on those features.
    3. Build an edge‑cost matrix using the learned weights.
    4. Return the Minimum‑Cost Spanning Tree and the final weight vector.
    """
    feats = hybrid_signature_features(texts)               # (N, D_sig)
    w = train_nlms_weights(feats, mu=mu, epochs=epochs)    # (D_sig,)
    cost_mat = build_edge_cost_matrix(feats, w)            # (N, N)
    mst = prim_mst(cost_mat)
    return mst, w


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence enables new forms of creativity.",
        "Quantum computing may revolutionize cryptography.",
        "Climate change impacts biodiversity worldwide."
    ]
    tree, final_weights = hybrid_pipeline(sample_texts, mu=0.6, epochs=10)
    print("MST edges (i, j, cost):")
    for e in tree:
        print(e)
    print("\nFinal NLMS weight vector (first 10 components):")
    print(final_weights[:10])