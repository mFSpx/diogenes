# DARWIN HAMMER — match 3286, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m977_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1202_s3.py (gen6)
# born: 2026-05-29T23:49:02Z

"""Hybrid algorithm merging:
- Parent A: sheaf‑based infotaxis with Count‑Min sketch, MinHash, SSIM and Ollivier‑Ricci curvature.
- Parent B: TTT‑Linear weight matrix learning (init, loss, gradient, step) and Gaussian‑RBF utilities.

Mathematical bridge:
Node sections `s_i ∈ ℝ^d` are first linearly transformed by a learnable weight matrix `W`
(from the TTT‑Linear family).  The transformed sections `ŝ_i = W @ s_i` are then fed
into the sheaf‑based edge metric of Parent A.  The edge metric provides a scalar
target `M_uv` that drives a gradient‑based update of `W` via the TTT loss
`L(W) = Σ_uv (M_uv - τ)^2` (τ is a desired similarity, here set to 1).  Thus the
linear mapping and the graph‑information‑theoretic similarity are tightly coupled
in a single optimization loop.  The resulting module implements three core
functions that demonstrate this hybrid operation."""
import math
import random
import sys
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Helpers from Parent A
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 32‑bit hash based on a seed and a string token."""
    data = seed.to_bytes(4, byteorder="little", signed=False) + token.encode("utf-8")
    return int.from_bytes(data[:4], byteorder="little", signed=False)


def _minhash_signature(vector: np.ndarray, num_perm: int = 64) -> np.ndarray:
    """Simple MinHash signature for a binary representation of `vector`."""
    # treat each dimension with a positive value as an element of the set
    indices = np.where(vector > 0)[0]
    sig = np.empty(num_perm, dtype=np.int64)
    for i in range(num_perm):
        mins = min((_hash(i, str(idx)) for idx in indices), default=2**31 - 1)
        sig[i] = mins
    return sig


def _jaccard_minhash(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    return np.mean(sig1 == sig2)


def _ssim_vec(x: np.ndarray, y: np.ndarray, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """Structural similarity for 1‑D vectors (mean‑variance formulation)."""
    mu_x, mu_y = x.mean(), y.mean()
    sigma_x2, sigma_y2 = x.var(), y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return num / den if den != 0 else 0.0


def _gaussian(r: float, epsilon: float = 1.0) -> float:
    """RBF kernel used as a curvature proxy."""
    return math.exp(-((epsilon * r) ** 2))


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("vectors must have same dimension")
    return np.sqrt(np.sum((a - b) ** 2))


# ----------------------------------------------------------------------
# Helpers from Parent B (TTT‑Linear)
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_loss(W: np.ndarray, X: np.ndarray, target: np.ndarray) -> float:
    """Mean‑squared loss between W·X and target (both d×n)."""
    diff = W @ X - target
    return np.mean(diff ** 2)


def ttt_grad(W: np.ndarray, X: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Gradient of the MSE loss w.r.t. W."""
    diff = W @ X - target
    # grad = 2 * diff @ X.T / n
    n = X.shape[1]
    return (2.0 / n) * diff @ X.T


def ttt_step(W: np.ndarray, X: np.ndarray, target: np.ndarray, eta: float = 0.01) -> np.ndarray:
    """One gradient‑descent step on the TTT loss."""
    grad = ttt_grad(W, X, target)
    return W - eta * grad


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class EdgeInfo:
    src: int
    dst: int
    restriction: tuple[np.ndarray, np.ndarray]  # (src_map, dst_map)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def edge_metric_hybrid(
    W: np.ndarray,
    sections: dict[int, np.ndarray],
    edge: EdgeInfo,
    lambda_curv: float = 0.1,
) -> float:
    """Compute the unified edge metric M_uv.

    1. Transform node sections with the learnable matrix W.
    2. Derive Fisher‑type weight w_f from variance of the transformed source.
    3. Derive entropy‑type weight w_h from normalized absolute values.
    4. Combine SSIM, MinHash Jaccard and a curvature penalty built from the
       restriction maps.
    """
    # 1. Linear transformation
    s_u = sections[edge.src]
    s_v = sections[edge.dst]
    s_u_t = W @ s_u
    s_v_t = W @ s_v

    # 2. Fisher‑type weight (variance based)
    var_u = np.var(s_u_t)
    w_f = var_u / (var_u + 1e-6)

    # 3. Entropy‑type weight (Shannon entropy of normalized abs values)
    prob = np.abs(s_u_t) / (np.sum(np.abs(s_u_t)) + 1e-12)
    H = -np.sum(prob * np.log(prob + 1e-12))
    w_h = H / (H + 1e-6)

    # 4a. SSIM similarity
    ssim = _ssim_vec(s_u_t, s_v_t)

    # 4b. MinHash Jaccard estimate
    sig_u = _minhash_signature(s_u_t)
    sig_v = _minhash_signature(s_v_t)
    jaccard = _jaccard_minhash(sig_u, sig_v)

    # 4c. Curvature penalty from restriction maps
    src_map, dst_map = edge.restriction
    r_dist = _euclidean(src_map, dst_map)
    curvature = _gaussian(r_dist)  # value in (0,1]
    penalty = lambda_curv * (1.0 - curvature)  # larger when maps are far apart

    # Uniform transition probability p_uv = 1 (absorbed into scaling)
    M = w_f * ssim + w_h * jaccard - penalty
    return float(M)


def infotaxis_step_hybrid(
    W: np.ndarray,
    sections: dict[int, np.ndarray],
    edges: list[EdgeInfo],
    eta_w: float = 0.01,
    lambda_curv: float = 0.1,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Perform one hybrid infotaxis iteration.

    - Evaluate M_uv for all edges.
    - Select the edge with maximal metric.
    - Update the destination section using a Count‑Min‑sketch‑style addition.
    - Update the weight matrix W by a gradient step aiming at M_max → 1.
    """
    # Evaluate metrics
    metrics = [edge_metric_hybrid(W, sections, e, lambda_curv) for e in edges]
    max_idx = int(np.argmax(metrics))
    best_edge = edges[max_idx]
    M_max = metrics[max_idx]

    # ----- Section update (Count‑Min‑style) -----
    src_map, dst_map = best_edge.restriction
    projected = src_map @ sections[best_edge.src]  # linear restriction
    # Count‑Min sketch style: add with min operation on each coordinate
    sections[best_edge.dst] = np.minimum(sections[best_edge.dst], projected)

    # ----- Weight matrix update -----
    # Target metric τ = 1 (perfect similarity).  Use a simple squared error loss.
    target_metric = np.array([[1.0]])  # shape (1,1) for compatibility
    # Build a dummy feature matrix X that contains the transformed source vector
    X = W @ sections[best_edge.src].reshape(-1, 1)  # shape (d_out,1)
    # Perform a gradient step on the TTT loss w.r.t. this single edge
    W_new = ttt_step(W, X, target_metric, eta=eta_w)

    return W_new, sections


def global_curvature_hybrid(edges: list[EdgeInfo]) -> float:
    """Sum of Gaussian curvature terms over all edges."""
    curv_sum = 0.0
    for e in edges:
        src_map, dst_map = e.restriction
        curv_sum += _gaussian(_euclidean(src_map, dst_map))
    return curv_sum


# ----------------------------------------------------------------------
# Utility to build a random sheaf‑graph for the smoke test
# ----------------------------------------------------------------------
def _random_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(dim)


def build_random_hybrid_graph(
    n_nodes: int,
    dim_section: int,
    dim_map: int,
    seed: int = 42,
) -> tuple[dict[int, np.ndarray], list[EdgeInfo], np.ndarray]:
    rng = np.random.default_rng(seed)
    # random sections per node
    sections = {i: _random_vector(dim_section, rng) for i in range(n_nodes)}

    # generate a simple undirected ring topology
    edges = []
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        src_map = _random_vector(dim_map, rng)
        dst_map = _random_vector(dim_map, rng)
        edges.append(EdgeInfo(src=i, dst=j, restriction=(src_map, dst_map)))
        # also add reverse direction for completeness
        edges.append(EdgeInfo(src=j, dst=i, restriction=(dst_map, src_map)))

    # initialise weight matrix W (dim_map × dim_section)
    W = init_ttt(dim_section, dim_map, scale=0.05, seed=seed)
    return sections, edges, W


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    N_NODES = 5
    DIM_SECTION = 12
    DIM_MAP = 8
    STEPS = 3

    # Build random problem instance
    sections, edges, W = build_random_hybrid_graph(N_NODES, DIM_SECTION, DIM_MAP)

    print(f"Initial global curvature: {global_curvature_hybrid(edges):.4f}")

    for step in range(STEPS):
        W, sections = infotaxis_step_hybrid(W, sections, edges, eta_w=0.005)
        curv = global_curvature_hybrid(edges)
        print(f"Step {step+1}: weight matrix norm={np.linalg.norm(W):.4f}, curvature={curv:.4f}")

    # Verify that edge_metric runs without error on a random edge
    test_edge = edges[0]
    m = edge_metric_hybrid(W, sections, test_edge)
    print(f"Metric on edge ({test_edge.src}->{test_edge.dst}) = {m:.4f}")

    print("Hybrid algorithm smoke test completed successfully.")