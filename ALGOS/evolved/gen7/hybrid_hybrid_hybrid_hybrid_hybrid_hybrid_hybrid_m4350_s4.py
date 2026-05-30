# DARWIN HAMMER — match 4350, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1449_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hdc_hy_hybrid_hybrid_regret_m846_s1.py (gen5)
# born: 2026-05-29T23:55:04Z

"""Hybrid Algorithm Fusion of DARWIN HAMMER Parents A and B.

Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1449_s0.py) provides:
- Shannon entropy of decision‑hygiene feature counts.
- TTT‑Linear weight matrix initialization and Gaussian‑based kernels.

Parent B (hybrid_hybrid_hybrid_hdc_hy_hybrid_hybrid_regret_m846_s1.py) provides:
- High‑dimensional binary hyper‑vectors, binding, bundling and similarity
  based on element‑wise multiplication.
- A regret‑weighted RBF surrogate concept (implemented here as a weighted
  radial‑basis function).

**Mathematical Bridge**

The bridge is built by projecting the binary hyper‑vectors of Parent B
through the TTT‑Linear weight matrix of Parent A, turning them into
continuous embeddings.  The Shannon entropy of decision‑hygiene feature
counts modulates the effective temperature of an RBF kernel, while a
regret scalar further weights the kernel output.  This yields a single
scalar “hybrid score” that simultaneously respects:
1. The information‑theoretic hygiene signal (entropy),
2. The linear transformation dynamics (TTT‑Linear matrix),
3. The high‑dimensional binding/bundling algebra (hyper‑vectors),
4. The regret‑aware similarity measure (RBF).

The module implements three core hybrid functions demonstrating this
integration.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Core primitives from Parent A
# ---------------------------------------------------------------------------

def shannon_entropy(counts: list[int]) -> float:
    """Compute Shannon entropy from a list of integer counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel used in Parent A."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same shape")
    return float(np.linalg.norm(a - b))

def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize the TTT‑Linear weight matrix."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

# ---------------------------------------------------------------------------
# Core primitives from Parent B
# ---------------------------------------------------------------------------

Vector = list[int]  # binary hyper‑vector with entries {-1, +1}

def random_vector(dim: int = 10000, seed: int | None = None) -> Vector:
    """Generate a random binary hyper‑vector (+1 / -1)."""
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise binding (multiplication) of two hyper‑vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: list[Vector]) -> Vector:
    """Bundling (majority vote) of a list of hyper‑vectors."""
    if not vectors:
        raise ValueError("at least one vector is required")
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError("all vectors must have equal length")
    sums = [0] * dim
    for v in vectors:
        for i, val in enumerate(v):
            sums[i] += val
    return [1 if s >= 0 else -1 for s in sums]

def hamming_similarity(a: Vector, b: Vector) -> float:
    """Similarity based on proportion of equal components (Hamming‑like)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / len(a)

# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------

def ttt_transform(binary_vec: Vector, weight_mat: np.ndarray) -> np.ndarray:
    """
    Project a binary hyper‑vector into the continuous space defined by the
    TTT‑Linear matrix.

    The binary vector is first cast to float (±1 → ±1.0) and then multiplied
    by the weight matrix:   y = W · x .
    """
    x = np.array(binary_vec, dtype=np.float64)
    if weight_mat.shape[1] != x.shape[0]:
        raise ValueError("Weight matrix column dimension must match vector length")
    return weight_mat @ x

def hybrid_regret_rbf(
    vec_a: Vector,
    vec_b: Vector,
    weight_mat: np.ndarray,
    feature_counts: list[int],
    regret: float = 1.0,
    gamma: float = 0.1,
) -> float:
    """
    Compute a regret‑weighted RBF similarity between two hyper‑vectors.

    Steps:
    1. Transform both vectors with the TTT‑Linear matrix (continuous embeddings).
    2. Compute Euclidean distance d in the transformed space.
    3. Apply a Gaussian RBF kernel:  k = exp(-gamma * d²).
    4. Modulate k by:
       - entropy_factor = 1 + shannon_entropy(feature_counts)
       - regret_factor = 1 / (1 + regret)   (higher regret → lower score)
    5. Return the product:  score = k * entropy_factor * regret_factor
    """
    # 1. Linear projection
    proj_a = ttt_transform(vec_a, weight_mat)
    proj_b = ttt_transform(vec_b, weight_mat)

    # 2. Euclidean distance
    d = euclidean(proj_a, proj_b)

    # 3. Gaussian RBF kernel
    k = math.exp(-gamma * (d ** 2))

    # 4. Entropy and regret modulation
    entropy_factor = 1.0 + shannon_entropy(feature_counts)
    regret_factor = 1.0 / (1.0 + regret)

    # 5. Hybrid score
    return k * entropy_factor * regret_factor

def hybrid_bundle_similarity(
    vectors: list[Vector],
    weight_mat: np.ndarray,
    feature_counts: list[int],
    regret: float = 1.0,
    gamma: float = 0.1,
) -> float:
    """
    Bundle a collection of hyper‑vectors, then compare the bundled result
    against the first vector using the hybrid regret‑RBF metric.

    This demonstrates the interaction of binding/bundling (Parent B) with
    the TTT‑Linear transformation and entropy‑regulated regret weighting
    (Parent A).
    """
    if not vectors:
        raise ValueError("At least one vector required")
    bundled = bundle(vectors)
    return hybrid_regret_rbf(
        vectors[0], bundled, weight_mat, feature_counts, regret=regret, gamma=gamma
    )

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Configuration
    DIM = 1024
    SEED = 42

    # Generate two random hyper‑vectors
    v1 = random_vector(dim=DIM, seed=SEED)
    v2 = random_vector(dim=DIM, seed=SEED + 1)

    # Initialize TTT‑Linear matrix (same input dimension as hyper‑vector)
    W = init_ttt(d_in=DIM, scale=0.02, seed=SEED)

    # Example decision‑hygiene feature counts (could be any non‑negative ints)
    feature_counts = [12, 7, 3, 0, 5]

    # Compute hybrid regret‑RBF score
    score = hybrid_regret_rbf(v1, v2, W, feature_counts, regret=0.5, gamma=0.05)
    print(f"Hybrid regret‑RBF score: {score:.6f}")

    # Compute hybrid bundle similarity using three vectors
    vecs = [v1, v2, random_vector(dim=DIM, seed=SEED + 2)]
    bundle_score = hybrid_bundle_similarity(vecs, W, feature_counts, regret=0.5, gamma=0.05)
    print(f"Hybrid bundle similarity score: {bundle_score:.6f}")

    # Verify that the module runs without raising exceptions
    sys.exit(0)