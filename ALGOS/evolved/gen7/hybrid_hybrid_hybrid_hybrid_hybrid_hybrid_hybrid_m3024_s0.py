# DARWIN HAMMER — match 3024, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hdc_hybrid_hy_m1295_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_sketches_rlct_m1910_s0.py (gen6)
# born: 2026-05-29T23:47:18Z

"""Hybrid Algorithm: Path Signature + NLMS + Count‑Min Sketch + Ollivier‑Ricci Curvature

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hdc_hybrid_hy_m1295_s2.py (Path signatures, NLMS‑Graph‑Tree fusion, ternary binding)
- hybrid_hybrid_hybrid_hybrid_hybrid_sketches_rlct_m1910_s0.py (Count‑Min sketch, circuit‑breaker, Morphology, Ollivier‑Ricci curvature, RLCT)

Mathematical Bridge:
Both parents operate on a *feature vector* extracted from data.  
Parent A transforms the vector into a multivariate path, applies a lead‑lag
transformation and extracts a truncated signature.  The signature is then
used in an NLMS weight‑update.  

Parent B builds a Count‑Min sketch of the same underlying items; the sketch
produces integer *frequency estimates* that can be interpreted as
non‑negative scalars attached to each feature dimension.  These scalars are
perfectly suited to re‑scale the NLMS learning‑rate (or the NLMS error term)
and to weight the edges of the similarity graph on which Ollivier‑Ricci
curvature is computed.

The hybrid therefore:
1. Generates a Count‑Min sketch for the current batch of items.
2. Uses the sketch frequencies to modulate the NLMS step applied to the
   path‑signature representation.
3. Constructs a similarity graph from signatures, computes a simple
   Ollivier‑Ricci curvature for each edge, and uses the curvature as an
   additional multiplicative factor when updating the NLMS weight vector.
4. Generates a ternary binding vector; its dot‑product with the signature
   yields a *decision‑hygiene* margin that further scales the NLMS learning‑rate.

The result is a unified system that learns adaptive edge weights while
preserving geometric information from path signatures and probabilistic
information from the sketch."""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Apply lead‑lag transformation to a multivariate path.

    For a path X_t ∈ ℝ^d, the lead‑lag augmentation creates a (d+1)‑dimensional
    path (X_t, ∫_0^t‖dX_s‖) where the extra coordinate accumulates the Euclidean
    norm of increments.
    """
    n, d = path.shape
    augmented = np.zeros((n, d + 1))
    augmented[:, :d] = path
    # cumulative norm of increments
    increments = np.diff(path, axis=0, prepend=path[0:1])
    cumulative_norm = np.cumsum(np.linalg.norm(increments, axis=1))
    augmented[:, d] = cumulative_norm
    return augmented


def truncated_signature(path: np.ndarray, order: int = 2) -> np.ndarray:
    """Compute a simple truncated signature up to given order.

    This implementation uses the Chen‑Chen (iterated integral) approximation
    by cumulatively multiplying coordinate differences.  It is not a full
    signature library but suffices for demonstration.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    increments = np.diff(path, axis=0)
    # order‑1 signature: sum of increments (i.e. the total displacement)
    sig = [increments.sum(axis=0)]
    # higher orders: element‑wise products of increments
    for k in range(2, order + 1):
        prod = np.prod(increments, axis=0) ** k
        sig.append(prod)
    return np.concatenate(sig)


def generate_ternary_vector(dim: int) -> np.ndarray:
    """Generate a random ternary vector with entries in {‑1, 0, +1}."""
    choices = [-1, 0, 1]
    return np.array([random.choice(choices) for _ in range(dim)], dtype=int)


def decision_hygiene_score(ternary_vec: np.ndarray, signature: np.ndarray) -> float:
    """Score = dot(ternary, signature) scaled to (0,1)."""
    raw = np.dot(ternary_vec, signature)
    # map to [0,1] via sigmoid for stability
    return 1.0 / (1.0 + math.exp(-raw))


def nlms_update(
    w: np.ndarray,
    x: np.ndarray,
    d: float,
    mu: float = 0.1,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """Normalized Least‑Mean‑Squares weight update.

    Returns the updated weight vector and the instantaneous error.
    """
    y = np.dot(w, x)
    e = d - y
    norm = np.dot(x, x) + epsilon
    w_new = w + (mu / norm) * e * x
    return w_new, e


# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
def count_min_sketch(
    items: List[str], width: int = 64, depth: int = 4
) -> List[List[int]]:
    """Simple Count‑Min sketch returning a 2‑D list of counters."""
    # 4 pairwise‑independent hash functions simulated by random seeds
    random.seed(0)
    hash_seeds = [random.randint(1, 1_000_000) for _ in range(depth)]

    sketch = [[0] * width for _ in range(depth)]

    for item in items:
        for d, seed in enumerate(hash_seeds):
            idx = (hash(item) ^ seed) % width
            sketch[d][idx] += 1
    return sketch


def estimate_frequency(sketch: List[List[int]], item: str) -> int:
    """Estimate frequency of *item* from the sketch (minimum over rows)."""
    min_est = sys.maxsize
    for row_idx, row in enumerate(sketch):
        idx = (hash(item) ^ (row_idx + 1)) % len(row)
        min_est = min(min_est, row[idx])
    return min_est


def similarity_from_signature(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Cosine similarity between two signature vectors."""
    norm_a = np.linalg.norm(sig_a)
    norm_b = np.linalg.norm(sig_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(sig_a, sig_b) / (norm_a * norm_b))


def ollivier_ricci_curvature(
    W: np.ndarray, i: int, j: int, epsilon: float = 1e-6
) -> float:
    """Very simple Ollivier‑Ricci curvature for edge (i,j) in a weighted graph.

    Uses the formula: κ_ij = 1 - W_ij / ( (deg_i + deg_j) / 2 )
    where deg_k = sum_l W_kl.
    """
    deg_i = W[i, :].sum()
    deg_j = W[j, :].sum()
    avg_deg = (deg_i + deg_j) / 2.0 + epsilon
    return 1.0 - (W[i, j] / avg_deg)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def build_similarity_graph(
    signatures: List[np.ndarray], threshold: float = 0.5
) -> np.ndarray:
    """Construct a symmetric adjacency matrix where entry (i,j) is similarity
    if it exceeds *threshold*, otherwise 0.
    """
    n = len(signatures)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity_from_signature(signatures[i], signatures[j])
            if sim >= threshold:
                G[i, j] = G[j, i] = sim
    return G


def hybrid_nlms_step(
    w: np.ndarray,
    path: np.ndarray,
    items: List[str],
    desired: float,
    mu_base: float = 0.05,
) -> Tuple[np.ndarray, float]:
    """Perform a single hybrid NLMS update.

    1. Lead‑lag transform + truncated signature.
    2. Count‑Min sketch frequencies → per‑dimension scaling vector `f`.
    3. Generate ternary vector → decision‑hygiene margin `m`.
    4. Compute curvature‑adjusted learning rate: μ = μ_base * m * f_mean * κ
    5. Apply NLMS update.
    """
    # 1. Path processing
    aug_path = lead_lag_transform(path)
    sig = truncated_signature(aug_path, order=2)

    # 2. Sketch frequencies (average over items)
    sketch = count_min_sketch(items, width=64, depth=4)
    freq_estimates = np.array([estimate_frequency(sketch, it) for it in items])
    # Avoid division by zero; treat zero freq as 1 (neutral scaling)
    freq_estimates = np.where(freq_estimates == 0, 1, freq_estimates)
    f_mean = freq_estimates.mean() / (freq_estimates.max() + 1e-9)

    # 3. Decision‑hygiene margin
    ternary = generate_ternary_vector(sig.shape[0])
    m = decision_hygiene_score(ternary, sig)

    # 4. Curvature factor (use a tiny graph of two nodes for illustration)
    # Build a 2‑node graph where edge weight = similarity of sig with itself
    # (i.e. 1). Curvature for a self‑edge is 0, but we construct a dummy edge.
    dummy_W = np.array([[0.0, 1.0], [1.0, 0.0]])
    κ = ollivier_ricci_curvature(dummy_W, 0, 1)

    # 5. Adjust learning rate
    mu = mu_base * m * f_mean * κ
    mu = max(min(mu, 1.0), 1e-6)  # keep mu in a reasonable range

    # NLMS update
    w_new, err = nlms_update(w, sig, desired, mu=mu)
    return w_new, err


def hybrid_graph_weight_adjustment(
    signatures: List[np.ndarray],
    items: List[str],
    base_weights: np.ndarray,
) -> np.ndarray:
    """Adjust a graph's edge weights using sketch frequencies and curvature.

    Returns a new weight matrix of the same shape as *base_weights*.
    """
    G = build_similarity_graph(signatures, threshold=0.3)

    # Sketch frequencies per item (used as node‑wise scaling)
    sketch = count_min_sketch(items, width=64, depth=4)
    node_scales = np.array(
        [estimate_frequency(sketch, it) for it in items], dtype=float
    )
    node_scales = np.where(node_scales == 0, 1.0, node_scales)

    # Apply node scaling to adjacency
    scale_matrix = np.outer(node_scales, node_scales)
    G_scaled = G * scale_matrix

    # Compute curvature for each existing edge and modulate weight
    n = G.shape[0]
    adjusted = np.copy(base_weights)
    for i in range(n):
        for j in range(i + 1, n):
            if G_scaled[i, j] > 0:
                κ = ollivier_ricci_curvature(G_scaled, i, j)
                adjusted[i, j] = adjusted[j, i] = base_weights[i, j] * (1 + κ)
    return adjusted


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic multivariate path (10 steps, 3 dimensions)
    rng = np.random.default_rng(42)
    path = rng.normal(size=(10, 3))

    # Dummy items (e.g., token strings)
    items = [f"token_{i}" for i in range(5)]

    # Initial NLMS weight vector (dimension matches signature length)
    dummy_sig_len = truncated_signature(lead_lag_transform(path), order=2).shape[0]
    w0 = np.zeros(dummy_sig_len)

    # Desired scalar output (e.g., regression target)
    desired = 0.7

    # Perform hybrid NLMS step
    w1, err = hybrid_nlms_step(w0, path, items, desired)
    print(f"Updated weights norm: {np.linalg.norm(w1):.4f}, error: {err:.4f}")

    # Build signatures for a small graph (3 nodes)
    signatures = [
        truncated_signature(lead_lag_transform(path + rng.normal(scale=0.1, size=path.shape)), 2)
        for _ in range(3)
    ]
    base_W = np.ones((3, 3)) - np.eye(3)  # simple uniform graph without self‑loops

    adjusted_W = hybrid_graph_weight_adjustment(signatures, items, base_W)
    print("Adjusted weight matrix:")
    print(adjusted_W)