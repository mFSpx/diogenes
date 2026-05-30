# DARWIN HAMMER — match 4621, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_rectified_flo_hybrid_hybrid_gini_c_m1244_s4.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hdc_hy_hybrid_hybrid_fisher_m1920_s2.py (gen5)
# born: 2026-05-29T23:56:59Z

"""Hybrid Rectified‑Flow / Gini / RBF / Tropical / Binary Fusion
================================================================

Parents
-------
* **Parent A** – *hybrid_hybrid_rectified_flo_hybrid_hybrid_gini_c_m1244_s4.py*  
  Provides straight‑line interpolants, Gini impurity on those interpolants,
  a Gaussian RBF similarity matrix and a tropical (max‑plus) belief propagation.

* **Parent B** – *hybrid_hybrid_hdc_hy_hybrid_hybrid_fisher_m1920_s2.py*  
  Supplies binary random vectors, binding/bundling operations, a dot‑product
  similarity, and a Gaussian kernel based on Euclidean distance.

Mathematical Bridge
-------------------
Both parents rely on a **Gaussian radial‑basis function** that maps Euclidean
distances to similarities.  In the fused algorithm we

1. generate rectified‑flow interpolants `Z_t`,
2. encode each `Z_t` as a high‑dimensional binary vector using the
   deterministic `symbol_vector`/`random_vector` machinery of Parent B,
3. compute an RBF similarity matrix `S` from Euclidean distances between the
   *real‑valued* interpolants (Parent A) – the same kernel used in Parent B,
4. run tropical belief propagation on `log S`,
5. evaluate a Gini impurity on the interpolants (Parent A) and a Fisher‑like
   dot‑product similarity on the binary encodings (Parent B).

The three ingredients are finally fused into a single split score.

The module below implements this pipeline and exposes three public functions
that demonstrate the hybrid operation.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utilities from Parent A – Rectified Flow
# ----------------------------------------------------------------------


def interpolant(x0: np.ndarray, x1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Straight‑line interpolant Z_t = t * x1 + (1‑t) * x0.

    Parameters
    ----------
    x0, x1 : np.ndarray
        Arrays of shape (d,) or (B, d).  If one‑dimensional they are treated as a
        single batch element.
    t : np.ndarray
        Parameter(s) in [0, 1]; broadcastable to the batch dimension.

    Returns
    -------
    np.ndarray
        Interpolated points of the same shape as the broadcasted inputs.
    """
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    t = np.asarray(t, dtype=float)

    # Broadcast to a common shape
    if x0.ndim == 1:
        x0 = x0[None, :]
    if x1.ndim == 1:
        x1 = x1[None, :]
    if t.ndim == 0:
        t = np.full((x0.shape[0], 1), t)
    elif t.ndim == 1:
        t = t[:, None]

    return t * x1 + (1.0 - t) * x0


def gini_impurity(values: np.ndarray) -> float:
    """
    Gini impurity of a 1‑D array of real numbers.

    The values are turned into a probability distribution by absolute‑value
    normalisation.

    Parameters
    ----------
    values : np.ndarray
        1‑D array.

    Returns
    -------
    float
        Gini impurity in [0, 1].
    """
    vals = np.abs(values).astype(float)
    if vals.sum() == 0:
        return 0.0
    probs = vals / vals.sum()
    return 1.0 - np.sum(probs ** 2)


# ----------------------------------------------------------------------
# Utilities from Parent B – Binary Vectors & Gaussian RBF
# ----------------------------------------------------------------------


Vector = List[int]
FloatVector = Sequence[float]


def random_vector(dim: int = 1024, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 1024) -> Vector:
    import hashlib

    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Component‑wise binding (multiplication) of two binary vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: List[Vector]) -> Vector:
    """Superposition (majority vote) of a list of binary vectors."""
    if not vectors:
        raise ValueError("at least one vector is required")
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError("vectors must have equal length")
    sums = np.zeros(dim, dtype=int)
    for v in vectors:
        sums += np.array(v, dtype=int)
    return [1 if s >= 0 else -1 for s in sums]


def similarity(a: Vector, b: Vector) -> float:
    """Normalised dot‑product similarity in [-1, 1]."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    if not a:
        raise ValueError("vectors must not be empty")
    return sum(x * y for x, y in zip(a, b)) / len(a)


def euclidean(a: FloatVector, b: FloatVector) -> float:
    """Euclidean distance between two real‑valued vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


# ----------------------------------------------------------------------
# Hybrid Core
# ----------------------------------------------------------------------


def encode_interpolants(
    interpolants: np.ndarray, dim: int = 1024
) -> List[Vector]:
    """
    Encode each interpolant row as a deterministic binary vector using the
    `symbol_vector` machinery.  The symbol is derived from a hash of the row
    values, guaranteeing reproducibility.

    Parameters
    ----------
    interpolants : np.ndarray
        Shape (N, d) where N is the number of interpolants.
    dim : int
        Length of the binary vectors.

    Returns
    -------
    List[Vector]
        List of length N containing binary vectors.
    """
    vectors: List[Vector] = []
    for row in interpolants:
        # Create a reproducible symbol from the floating point data
        h = hash(row.tobytes())
        symbol = f"vec_{h}"
        vectors.append(symbol_vector(symbol, dim))
    return vectors


def rbf_similarity_matrix(points: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Build a symmetric similarity matrix S_{ij} = exp(- (ε * ||p_i-p_j||)^2).

    Parameters
    ----------
    points : np.ndarray
        Shape (N, d).
    epsilon : float
        Kernel width.

    Returns
    -------
    np.ndarray
        Shape (N, N) with entries in (0, 1].
    """
    N = points.shape[0]
    S = np.empty((N, N), dtype=float)
    for i in range(N):
        for j in range(i, N):
            r = euclidean(points[i], points[j])
            val = gaussian(r, epsilon)
            S[i, j] = val
            S[j, i] = val
    return S


def tropical_propagation(logS: np.ndarray, steps: int = 3) -> np.ndarray:
    """
    Perform tropical (max‑plus) belief propagation.

    ℓ_{k+1} = ℓ_k ⊗ logS  ⇔  (ℓ_{k+1})_i = max_j ( ℓ_k_j + logS_{j,i} ).

    Parameters
    ----------
    logS : np.ndarray
        Log‑similarity matrix (N, N).
    steps : int
        Number of propagation iterations.

    Returns
    -------
    np.ndarray
        Final belief vector (N,).
    """
    N = logS.shape[0]
    # Initialise with zeros (log‑probability seed)
    ell = np.zeros(N, dtype=float)
    for _ in range(steps):
        new = np.full(N, -np.inf, dtype=float)
        for i in range(N):
            # max over j of ell_j + logS_{j,i}
            new[i] = np.max(ell + logS[:, i])
        ell = new
    return ell


def hybrid_split_score(
    x0: np.ndarray,
    x1: np.ndarray,
    t_vals: np.ndarray,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.1,
    delta: float = 0.1,
    epsilon: float = 1.0,
) -> float:
    """
    Compute the fused split score for a pair of points.

    The score combines:
    * Gini impurity of the interpolants,
    * tropical belief mean,
    * average RBF similarity,
    * Fisher‑like binary similarity between bundled encodings.

    Parameters
    ----------
    x0, x1 : np.ndarray
        End points (1‑D or batch).
    t_vals : np.ndarray
        Parameter values for interpolation.
    alpha, beta, gamma, delta : float
        Weights for the four components.
    epsilon : float
        RBF kernel width.

    Returns
    -------
    float
        The hybrid split score.
    """
    # 1. Interpolants
    Z = interpolant(x0, x1, t_vals)  # shape (N, d)

    # 2. Gini impurity (average over all interpolants)
    I = np.mean([gini_impurity(z) for z in Z])

    # 3. RBF similarity matrix and tropical belief
    S = rbf_similarity_matrix(Z, epsilon=epsilon)
    logS = np.log(S + 1e-12)  # avoid log(0)
    ell = tropical_propagation(logS, steps=3)
    B = float(np.mean(ell))

    # 4. Binary encoding, bundling and Fisher‑like similarity
    binary_vecs = encode_interpolants(Z, dim=1024)
    bundled = bundle(binary_vecs)  # single prototype vector
    # similarity between bundled vector and each encoded interpolant
    fisher_scores = np.array([similarity(bundled, v) for v in binary_vecs])
    F = float(np.mean(fisher_scores))

    # 5. Combine
    score = alpha * I + beta * B + gamma * np.mean(S) + delta * F
    return score


# ----------------------------------------------------------------------
# Demonstration Functions
# ----------------------------------------------------------------------


def compute_features(x0: np.ndarray, x1: np.ndarray, t_vals: np.ndarray) -> Tuple[np.ndarray, List[Vector]]:
    """
    Return the interpolants and their binary encodings.
    """
    Z = interpolant(x0, x1, t_vals)
    binary = encode_interpolants(Z, dim=1024)
    return Z, binary


def compute_belief_and_similarity(Z: np.ndarray, epsilon: float = 1.0) -> Tuple[float, float]:
    """
    Given interpolants Z, compute the tropical belief mean and the average RBF similarity.
    """
    S = rbf_similarity_matrix(Z, epsilon=epsilon)
    logS = np.log(S + 1e-12)
    ell = tropical_propagation(logS, steps=3)
    B = float(np.mean(ell))
    avg_sim = float(np.mean(S))
    return B, avg_sim


def demo_hybrid_score() -> float:
    """
    Run a small synthetic example and return the hybrid split score.
    """
    rng = np.random.default_rng(42)
    x0 = rng.normal(size=5)
    x1 = rng.normal(size=5)
    t_vals = np.linspace(0.0, 1.0, num=7)  # 7 interpolants
    return hybrid_split_score(x0, x1, t_vals)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Simple sanity check that all components run without error
    score = demo_hybrid_score()
    print(f"Hybrid split score (demo): {score:.6f}")

    # Additional checks
    rng = np.random.default_rng(0)
    a = rng.normal(size=8)
    b = rng.normal(size=8)
    t = np.array([0.0, 0.5, 1.0])
    Z, bin_vecs = compute_features(a, b, t)
    print(f"Interpolants shape: {Z.shape}")
    print(f"Number of binary vectors: {len(bin_vecs)}")
    B, avgS = compute_belief_and_similarity(Z)
    print(f"Tropical belief mean: {B:.4f}, average RBF similarity: {avgS:.4f}")