# DARWIN HAMMER — match 977, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_infotaxis_min_m242_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_krampu_m11_s2.py (gen4)
# born: 2026-05-29T23:32:07Z

"""Hybrid algorithm merging:
- hybrid_hybrid_hybrid_hybrid_hybrid_infotaxis_min_m242_s1 (sheaf + Count‑Min sketch + MinHash + infotaxis)
- hybrid_hybrid_hybrid_fisher_hybrid_hybrid_krampu_m11_s2 (Fisher‑weighted SSIM, entropy, Ollivier‑Ricci curvature regularization)

Mathematical bridge:
Both parents manipulate *information* on a graph.  The sheaf stores a vector (section) per node and linear restriction maps per edge.
Parent B provides a scalar weight derived from Fisher information I(θ) and Shannon entropy H that modulates similarity measures.
We therefore use the Fisher‑derived weight `w_f = I/(I+ε)` and entropy‑derived weight `w_h = H/(H+ε)` to blend:
* the SSIM similarity between two node sections,
* the MinHash Jaccard similarity (information‑loss estimate),
* a curvature penalty Ω(W) built from the Ollivier‑Ricci curvature of each restriction map.

The unified decision metric for an edge (u,v) becomes

    M_uv = p_uv * [ w_f·SSIM(u,v) + w_h·Jaccard(u,v) + λ·Ω(R_uv) ]

where `p_uv` is a transition probability (here uniform), `R_uv` is the restriction pair
(src_map, dst_map) for the edge, and λ is a tunable regularization constant.
The algorithm updates sections by selecting the edge with maximal M_uv (infotaxis step)
and then projects the source section through its restriction map, adding the result
to the destination section using a Count‑Min sketch‑style update.

The code below implements this hybrid system with three public functions:
`edge_metric`, `infotaxis_step`, and `global_curvature`.  A small smoke test
exercises the full pipeline."""
import numpy as np
import math
import random
import sys
from pathlib import Path
from collections import defaultdict, Counter

# ----------------------------------------------------------------------
# Helper hash utilities (from parent A)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 32‑bit hash based on a seed and a string token."""
    data = seed.to_bytes(4, byteorder="little", signed=False) + token.encode("utf-8")
    return int.from_bytes(data[:4], byteorder="little", signed=False)

def _minhash_signature(vector: np.ndarray, num_perm: int = 64) -> np.ndarray:
    """Compute a simple MinHash signature for a numeric vector."""
    sig = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    for idx, val in enumerate(vector):
        token = f"{idx}:{val}"
        for i in range(num_perm):
            h = _hash(i, token)
            if h < sig[i]:
                sig[i] = h
    return sig

def _jaccard_from_minhash(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if sig1.shape != sig2.shape:
        raise ValueError("signatures must have equal length")
    return np.mean(sig1 == sig2)

# ----------------------------------------------------------------------
# Core sheaf structure (from parent A, trimmed)
# ----------------------------------------------------------------------
class HybridSheaf:
    def __init__(self, node_dims, edge_list, width=64, depth=4):
        """
        node_dims: dict {node_id: dimension}
        edge_list: iterable of (u, v) tuples
        width, depth: parameters for an internal Count‑Min sketch (unused in this
                      simplified implementation but kept for API compatibility)
        """
        self.node_dims = dict(node_dims)
        self.edges = list(edge_list)
        self._restrictions = {}          # (u,v) -> (src_map, dst_map)
        self._sections = {}              # node -> np.ndarray
        self.width = width
        self.depth = depth

    # ------------------------------------------------------------------
    # API for building the sheaf
    # ------------------------------------------------------------------
    def set_restriction(self, edge, src_map, dst_map):
        u, v = edge
        self._restrictions[(u, v)] = (np.array(src_map, dtype=float),
                                      np.array(dst_map, dtype=float))

    def set_section(self, node, value):
        self._sections[node] = np.array(value, dtype=float)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _edge_dim(self, u, v):
        if (u, v) in self._restrictions:
            return self._restrictions[(u, v)][0].shape[0]
        if (v, u) in self._restrictions:
            return self._restrictions[(v, u)][1].shape[0]
        raise KeyError(f"No restriction map for edge ({u}, {v})")

    def _c0_layout(self):
        nodes = list(self.node_dims.keys())
        offsets = {}
        pos = 0
        for n in nodes:
            offsets[n] = pos
            pos += self.node_dims[n]
        return nodes, offsets, pos

# ----------------------------------------------------------------------
# Fisher‑SSIM, entropy, and curvature utilities (from parent B)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural similarity index for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    sigma_xy = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * sigma_xy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))

def shannon_entropy(vec: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a normalized absolute‑value vector."""
    prob = np.abs(vec).astype(float)
    total = prob.sum()
    if total == 0:
        return 0.0
    prob = prob / total + eps
    return -np.sum(prob * np.log(prob))

def ollivier_ricci_curvature(src_map: np.ndarray, dst_map: np.ndarray) -> float:
    """
    Simple Ollivier‑Ricci curvature approximation for a linear map pair.
    Treat each column of src_map and dst_map as probability distributions,
    compute Wasserstein‑1 distance (L1) and map it to curvature:
        κ = 1 - d / d_max
    where d_max is the maximal possible L1 distance (2 for probability vectors).
    """
    # Normalize columns to sum to 1 (avoid division by zero)
    def normalize_cols(m):
        col_sums = m.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        return m / col_sums

    src_n = normalize_cols(src_map)
    dst_n = normalize_cols(dst_map)
    # L1 distance per column, then average
    d = np.mean(np.sum(np.abs(src_n - dst_n), axis=0))
    d_max = 2.0
    return 1.0 - d / d_max

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def edge_metric(sheaf: HybridSheaf, edge, eps: float = 1e-12, lam: float = 0.5) -> float:
    """
    Compute the unified decision metric M_uv for edge (u, v).

    Returns:
        float – the metric value.
    """
    u, v = edge
    if u not in sheaf._sections or v not in sheaf._sections:
        raise KeyError("Both nodes must have sections defined.")
    xu = sheaf._sections[u]
    xv = sheaf._sections[v]

    # --- Fisher weight ------------------------------------------------
    center_u = np.mean(xu)
    width_u = np.std(xu) + eps
    I_u = fisher_score(theta=center_u, center=center_u, width=width_u, eps=eps)
    w_f = I_u / (I_u + eps)

    # --- Entropy weight ------------------------------------------------
    H_u = shannon_entropy(xu, eps=eps)
    w_h = H_u / (H_u + eps)

    # --- Similarities --------------------------------------------------
    ssim_sim = ssim(xu, xv)
    minhash_u = _minhash_signature(xu)
    minhash_v = _minhash_signature(xv)
    jaccard_sim = _jaccard_from_minhash(minhash_u, minhash_v)

    # --- Curvature regularization --------------------------------------
    if edge not in sheaf._restrictions:
        raise KeyError(f"Restriction maps missing for edge {edge}")
    src_map, dst_map = sheaf._restrictions[edge]
    curvature = ollivier_ricci_curvature(src_map, dst_map)
    curvature_penalty = lam * (1.0 - curvature)  # larger penalty when curvature low

    # --- Uniform transition probability (could be replaced) ------------
    p_uv = 1.0 / len(sheaf.edges)

    M = p_uv * (w_f * ssim_sim + w_h * jaccard_sim + curvature_penalty)
    return M

def infotaxis_step(sheaf: HybridSheaf, current_node, eps: float = 1e-12, lam: float = 0.5):
    """
    Perform one infotaxis iteration:
    1. Evaluate M_uv for all outgoing edges from `current_node`.
    2. Choose the edge with maximal metric.
    3. Project the source section through the restriction map and add it
       to the destination section (Count‑Min sketch style additive update).
    Returns the chosen neighbor node.
    """
    outgoing = [e for e in sheaf.edges if e[0] == current_node]
    if not outgoing:
        raise ValueError(f"No outgoing edges from node {current_node}")

    metrics = [(edge, edge_metric(sheaf, edge, eps=eps, lam=lam)) for edge in outgoing]
    best_edge, best_metric = max(metrics, key=lambda item: item[1])
    u, v = best_edge

    # Projection via restriction map
    src_map, dst_map = sheaf._restrictions[best_edge]
    src_section = sheaf._sections[u]
    # Apply src_map (assumed compatible dimensions)
    projected = src_map @ src_section
    # Add to destination using a simple additive rule (mirrors Count‑Min sketch update)
    if v not in sheaf._sections:
        sheaf._sections[v] = np.zeros(dst_map.shape[0])
    sheaf._sections[v] += dst_map @ projected

    return v

def global_curvature(sheaf: HybridSheaf) -> float:
    """
    Compute the average Ollivier‑Ricci curvature over all edges.
    """
    if not sheaf._restrictions:
        return 0.0
    curvatures = [ollivier_ricci_curvature(src, dst) for src, dst in sheaf._restrictions.values()]
    return float(np.mean(curvatures))

# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph
    nodes = {0: 5, 1: 5, 2: 5}
    edges = [(0, 1), (1, 2)]

    # Initialise sheaf
    sheaf = HybridSheaf(node_dims=nodes, edge_list=edges)

    # Random sections
    rng = np.random.default_rng(42)
    for n in nodes:
        sheaf.set_section(n, rng.integers(0, 256, size=nodes[n]))

    # Random linear restriction maps (compatible dimensions)
    for (u, v) in edges:
        dim_u = nodes[u]
        dim_v = nodes[v]
        src_map = rng.normal(size=(dim_v, dim_u))
        dst_map = rng.normal(size=(dim_v, dim_v))
        sheaf.set_restriction((u, v), src_map, dst_map)

    # Run a few infotaxis steps
    current = 0
    for step in range(3):
        try:
            nxt = infotaxis_step(sheaf, current)
            print(f"Step {step}: {current} -> {nxt}, metric={edge_metric(sheaf, (current, nxt)):.4f}")
            current = nxt
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break

    print(f"Average Ollivier‑Ricci curvature: {global_curvature(sheaf):.4f}")