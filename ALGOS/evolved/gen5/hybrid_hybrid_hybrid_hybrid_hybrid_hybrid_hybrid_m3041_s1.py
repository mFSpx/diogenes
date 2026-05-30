# DARWIN HAMMER — match 3041, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m405_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hard_t_m625_s0.py (gen4)
# born: 2026-05-29T23:47:24Z

"""Hybrid Algorithm integrating:
- Parent A: minimum‑cost tree Bayesian bandit router with hybrid privacy health metric.
- Parent B: ternary‑lens audit, path‑signature (lead‑lag) extraction, KAN spline basis, and linguistic similarity scoring.

Mathematical bridge:
The health metric from Parent A (≈ (1‑risk·failure_rate)·(1‑recovery)·log(count))
is used as a scalar weight for the signature vector derived from the audit‑derived
path of Parent B.  The weighted signature is then projected onto a spline basis
(KAN) and finally combined with a linguistic similarity score.  The core equation
of the hybrid decision score 𝚜 is:

    𝚜 = 𝛼·health·(Φ·w) + β·similarity

where
- Φ is the spline‑basis matrix built on a fixed grid,
- w are learned (here randomly‑initialised) KAN weights,
- similarity is a cosine similarity between linguistic feature vectors,
- α,β are tunable scalars.

The implementation below follows this formulation, providing three core functions:
`compute_health`, `lead_lag_signature`, and `hybrid_decision`.  A small smoke test
exercises the full pipeline on synthetic data."""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Parent A – tree utilities and health metric
# ----------------------------------------------------------------------
def euclidean_length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a, b) → length
    node_dist : dict mapping node → root‑to‑node distance
    """
    adj: Dict[str, List[str]] = {}
    edge_len: Dict[Edge, float] = {}
    node_dist: Dict[str, float] = {}

    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)  # undirected tree for distance calc
        edge_len[(a, b)] = euclidean_length(nodes[a], nodes[b])
        edge_len[(b, a)] = edge_len[(a, b)]

    # BFS to compute root‑to‑node distances
    visited = {root}
    queue = [(root, 0.0)]
    while queue:
        cur, dist = queue.pop(0)
        node_dist[cur] = dist
        for nb in adj.get(cur, []):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, dist + edge_len[(cur, nb)]))
    return adj, edge_len, node_dist


def compute_health(
    reconstruction_risk: float,
    failures: int,
    failure_threshold: int,
    recovery_priority: float,
    count: int,
) -> float:
    """
    Health metric from Parent A.

    health = (1 - (reconstruction_risk * failure_rate)) *
             (1 - recovery_priority) *
             log(count)

    Parameters
    ----------
    reconstruction_risk : float in [0,1]
    failures : int
    failure_threshold : int > 0
    recovery_priority : float in [0,1]
    count : int > 0 (bandit‑router count)

    Returns
    -------
    health : float
    """
    if failure_threshold <= 0:
        raise ValueError("failure_threshold must be > 0")
    if count <= 0:
        raise ValueError("count must be > 0")
    failure_rate = failures / failure_threshold
    health = (1.0 - (reconstruction_risk * failure_rate)) * (1.0 - recovery_priority) * math.log(count)
    return health


# ----------------------------------------------------------------------
# Parent B – audit classification, lead‑lag signature, and KAN spline basis
# ----------------------------------------------------------------------
CLASSIFICATIONS = {
    "usable_now": 0,
    "research_only": 1,
    "needs_conversion": 2,
    "deprecated": 3,
    "unknown": 4,
}


def one_hot(classification: str, num_classes: int = len(CLASSIFICATIONS)) -> np.ndarray:
    """Convert a textual classification into a one‑hot vector."""
    vec = np.zeros(num_classes, dtype=float)
    idx = CLASSIFICATIONS.get(classification, -1)
    if idx >= 0:
        vec[idx] = 1.0
    return vec


def lead_lag_signature(path: np.ndarray) -> np.ndarray:
    """
    Simple lead‑lag transform producing linear and quadratic features.

    For a sequence X_t ∈ ℝ^d, the transform returns
        [X_t, X_{t+1}, X_t * X_{t+1}]
    concatenated over t.

    Parameters
    ----------
    path : (T, d) array

    Returns
    -------
    sig : ( (T‑1)*(2d + d²) , ) vector
    """
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array")
    T, d = path.shape
    if T < 2:
        return np.array([])

    lead = path[1:]          # X_{t+1}
    lag = path[:-1]          # X_t
    linear = np.hstack([lag, lead])                     # 2d per step
    quadratic = (lag[:, :, None] * lead[:, None, :]).reshape(T - 1, -1)  # d² per step
    sig = np.hstack([linear, quadratic]).ravel()
    return sig


def spline_basis(grid: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Construct a polynomial spline basis (Vandermonde matrix) on the supplied grid.

    Parameters
    ----------
    grid : (N,) array of points where basis functions are evaluated
    degree : polynomial degree (default cubic)

    Returns
    -------
    Phi : (N, degree+1) matrix where Phi[i, j] = grid[i]**j
    """
    N = grid.shape[0]
    Phi = np.vander(grid, N=degree + 1, increasing=True)
    return Phi


def kan_mix(basis: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Linear mixing of spline basis rows with learned weights (KAN style).

    Parameters
    ----------
    basis : (N, K) matrix
    weights : (K,) vector

    Returns
    -------
    mixed : (N,) vector = basis @ weights
    """
    return basis @ weights


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


# ----------------------------------------------------------------------
# Hybrid core: combine health, signature, spline (KAN) and linguistic similarity
# ----------------------------------------------------------------------
def hybrid_decision(
    health: float,
    signature: np.ndarray,
    spline_grid: np.ndarray,
    kan_weights: np.ndarray,
    linguistic_feat_a: np.ndarray,
    linguistic_feat_b: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> float:
    """
    Compute the final hybrid decision score.

    𝚜 = α·health·(Φ·w)·⟨sig, Φ·w⟩ + β·similarity

    For simplicity we contract the signature with the KAN‑mixed basis
    via a dot product.

    Parameters
    ----------
    health : scalar from Parent A
    signature : vector from lead‑lag transform (Parent B)
    spline_grid : points where the spline basis is evaluated
    kan_weights : weight vector for the spline basis
    linguistic_feat_a, linguistic_feat_b : feature vectors for similarity
    alpha, beta : scaling constants

    Returns
    -------
    decision_score : float
    """
    # Build spline basis and mix with KAN weights
    Phi = spline_basis(spline_grid)                # (N, K)
    mixed = kan_mix(Phi, kan_weights)              # (N,)

    # Contract signature with the mixed basis (simple dot product after truncation/padding)
    min_len = min(signature.size, mixed.size)
    if min_len == 0:
        sig_proj = 0.0
    else:
        sig_proj = float(np.dot(signature[:min_len], mixed[:min_len]))

    # Linguistic similarity
    similarity = cosine_similarity(linguistic_feat_a, linguistic_feat_b)

    decision = alpha * health * sig_proj + beta * similarity
    return decision


# ----------------------------------------------------------------------
# Helper to generate synthetic data for the smoke test
# ----------------------------------------------------------------------
def _synthetic_tree() -> Tuple[Dict[str, Point], List[Edge], str]:
    nodes = {
        "root": (0.0, 0.0),
        "A": (1.0, 0.0),
        "B": (0.0, 1.0),
        "C": (1.0, 1.0),
    }
    edges = [("root", "A"), ("root", "B"), ("A", "C")]
    return nodes, edges, "root"


def _synthetic_audit_path() -> np.ndarray:
    # 3‑step path in 2‑D space (could represent audit scores over time)
    return np.array([[0.2, 0.8],
                     [0.5, 0.4],
                     [0.9, 0.1],
                     [0.7, 0.3]])


def _synthetic_linguistic_features() -> Tuple[np.ndarray, np.ndarray]:
    # Random TF‑IDF‑like vectors
    rng = np.random.default_rng(42)
    a = rng.random(10)
    b = rng.random(10)
    return a, b


def _synthetic_kan_weights(k: int) -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.normal(scale=0.1, size=k)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Tree and health
    nodes, edges, root = _synthetic_tree()
    _, _, node_dist = tree_metrics(nodes, edges, root)
    # Use the distance of node "C" as a proxy for count (bandit statistic)
    count = max(1, int(node_dist.get("C", 1) * 10))

    health = compute_health(
        reconstruction_risk=0.2,
        failures=3,
        failure_threshold=10,
        recovery_priority=0.15,
        count=count,
    )
    print(f"Health metric: {health:.4f}")

    # 2. Audit signature
    path = _synthetic_audit_path()
    signature = lead_lag_signature(path)
    print(f"Signature length: {signature.size}")

    # 3. KAN / spline
    grid = np.linspace(0, 1, num=signature.size // 4 + 1)  # modest grid
    kan_w = _synthetic_kan_weights(degree := 3 + 1)  # degree+1 columns

    # 4. Linguistic similarity
    ling_a, ling_b = _synthetic_linguistic_features()

    # 5. Hybrid decision
    score = hybrid_decision(
        health=health,
        signature=signature,
        spline_grid=grid,
        kan_weights=kan_w,
        linguistic_feat_a=ling_a,
        linguistic_feat_b=ling_b,
        alpha=1.2,
        beta=0.3,
    )
    print(f"Hybrid decision score: {score:.6f}")

    sys.exit(0)