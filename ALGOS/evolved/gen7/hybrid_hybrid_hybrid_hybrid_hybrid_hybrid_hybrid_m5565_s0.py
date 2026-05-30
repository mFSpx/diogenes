# DARWIN HAMMER — match 5565, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1235_s6.py (gen6)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_model_vram_sc_m61_s0.py (gen4)
# born: 2026-05-30T00:02:50Z

"""Hybrid Algorithm: fusion of
- PARENT ALGORITHM A (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1235_s6.py)
- PARENT ALGORITHM B (hybrid_hybrid_hybrid_rbf_su_hybrid_model_vram_sc_m61_s0.py)

Mathematical Bridge
-------------------
Both parents rely on Hoeffding‑type confidence bounds and on pairwise
similarities between nodes.  The bridge is built by:

1. **Regret‑aware Hoeffding bound** – the regret scalar *ε* from A scales
   the classic Hoeffding term.
2. **Similarity‑modulated confidence** – the RBF‑derived similarity matrix
   *S* from B rescales the confidence level *δ* (higher similarity → tighter
   bound).
3. **Tropical‑regret aggregation** – tropical (max‑plus) algebra combines gain
   *g* and regret *r* into a single scalar *T(g,r)=g+r* while keeping the
   max‑operation explicit for interpretability.
4. **Graph‑diffused LSM‑NLMS update** – the Least‑Squares Magnitude (LSM) vector
   from B guides the NLMS step size, and a graph diffusion term from A spreads
   updates across neighbouring nodes.

The three core functions below realise this unified system and can be
plugged into any streaming‑learning pipeline."""


from __future__ import annotations

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Mapping, Hashable, Iterable, List, Dict, Set, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
FeatureVec = Sequence[float]

@dataclass(frozen=True)
class MathAction:
    """Atomic action used by both parent algorithms."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for a given action."""
    action_id: str
    outcome_value: float
    probability: float


# ----------------------------------------------------------------------
# 1. Regret‑aware, similarity‑modulated Hoeffding bound
# ----------------------------------------------------------------------
def hybrid_hoeffding_bound(
    gain: float,
    n_samples: int,
    delta: float,
    regret: float,
    similarity_factor: float = 1.0,
) -> float:
    """
    Compute a Hoeffding‑type confidence bound that is scaled by:
    * regret (energy‑like factor ε) – from Parent A
    * similarity_factor (derived from RBF similarity matrix) – from Parent B

    The bound is:

        B = sqrt( g·(1+ε)·ln( 2 / (δ·σ) ) / (2·n) )

    where σ = similarity_factor ≥ 1 (higher similarity → more stringent bound).

    Parameters
    ----------
    gain: observed gain g
    n_samples: number of observations n
    delta: base confidence level δ (e.g. 0.05)
    regret: regret scalar ε
    similarity_factor: σ ≥ 1, typically 1 + mean(S) where S is the similarity matrix

    Returns
    -------
    float – the computed bound
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0,1)")
    if similarity_factor <= 0:
        raise ValueError("similarity_factor must be positive")

    adjusted_delta = delta * similarity_factor
    # protect against log of non‑positive
    adjusted_delta = min(max(adjusted_delta, 1e-12), 0.999999)
    log_term = math.log(2.0 / adjusted_delta)
    bound = math.sqrt(gain * (1.0 + regret) * log_term / (2.0 * n_samples))
    return bound


# ----------------------------------------------------------------------
# 2. Tropical regret (max‑plus) combined with LSM‑guided NLMS update
# ----------------------------------------------------------------------
def tropical_regret(gain: float, regret: float) -> float:
    """
    Tropical (max‑plus) aggregation of gain and regret.
    Explicitly shows the max‑operation, although algebraically it reduces to g+r.
    """
    max_part = max(gain, regret)
    min_part = min(gain, regret)
    return max_part + min_part  # equals gain + regret


def lsm_guided_nlms_update(
    weights: np.ndarray,
    lsm_vector: np.ndarray,
    gradient: np.ndarray,
    graph: Graph,
    node: Node,
    diffusion_rate: float = 0.1,
) -> np.ndarray:
    """
    Perform a Normalised Least‑Mean‑Squares (NLMS) weight update where the
    step size μ is modulated by the LSM vector (from Parent B) and a
    diffusion term spreads the update to neighbours (from Parent A).

    NLMS step:
        μ = (lsm·||gradient||) / (ε + ||gradient||²)
    Diffusion:
        w_i ← w_i + diffusion_rate * Σ_{j∈N(i)} (w_j - w_i)

    Returns the updated weight vector for the given node.
    """
    eps = 1e-12
    grad_norm_sq = np.dot(gradient, gradient)
    lsm_norm = np.linalg.norm(lsm_vector)
    mu = (lsm_norm * math.sqrt(grad_norm_sq)) / (eps + grad_norm_sq)

    # Base NLMS update
    updated = weights - mu * gradient

    # Graph diffusion (average of neighbours)
    neighbours = graph.get(node, set())
    if neighbours:
        neighbour_weights = np.stack([weights_dict[n] for n in neighbours if n in weights_dict])
        if neighbour_weights.size:
            avg_neighbour = neighbour_weights.mean(axis=0)
            updated = updated + diffusion_rate * (avg_neighbour - updated)

    return updated


# ----------------------------------------------------------------------
# 3. RBF similarity matrix with VRAM‑budget‑adjusted bandwidth
# ----------------------------------------------------------------------
def gaussian_kernel(r: float, epsilon: float) -> float:
    """Radial basis function with tunable bandwidth ε."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean_distance(a: FeatureVec, b: FeatureVec) -> float:
    if len(a) != len(b):
        raise ValueError("Feature vectors must have equal length")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    """
    Simple perceptual hash: binarise values w.r.t. their mean,
    keep up to 64 bits.
    """
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def similarity_matrix(
    features: Dict[Node, FeatureVec],
    vram_budget_mb: int,
) -> Tuple[np.ndarray, List[Node]]:
    """
    Build an (n×n) similarity matrix S where each entry is the Gaussian
    kernel applied to a combined distance:
        d_ij = α·hamming(phash_i,phash_j) + (1-α)·euclidean(fi,fj)

    The bandwidth ε is derived from the VRAM budget as in Parent B.
    """
    nodes = list(features.keys())
    n = len(nodes)
    if n == 0:
        return np.empty((0, 0)), nodes

    # Bandwidth scaling: larger VRAM → smaller epsilon (sharper kernel)
    epsilon = 1.0 / max(vram_budget_mb / 1024.0, 0.1)

    # Pre‑compute perceptual hashes
    phashes = {node: compute_phash(list(features[node])) for node in nodes}

    S = np.empty((n, n), dtype=np.float64)
    alpha = 0.5  # balance between Hamming and Euclidean components

    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
                continue
            h_dist = hamming_distance(phashes[ni], phashes[nj])
            e_dist = euclidean_distance(features[ni], features[nj])
            combined = alpha * h_dist + (1 - alpha) * e_dist
            S[i, j] = gaussian_kernel(combined, epsilon)
    return S, nodes


# ----------------------------------------------------------------------
# 4. Hybrid split decision using the fused bound
# ----------------------------------------------------------------------
def should_split_node(
    node: Node,
    features: Dict[Node, FeatureVec],
    gain_estimate: float,
    n_samples: int,
    delta: float,
    regret: float,
    vram_budget_mb: int,
    graph: Graph,
    similarity_threshold: float = 0.7,
) -> bool:
    """
    Decide whether to split a Hoeffding tree node.
    Steps:
    1. Build similarity matrix and obtain a global similarity factor σ.
    2. Compute the regret‑aware, similarity‑modulated Hoeffding bound.
    3. Apply tropical regret to combine gain and regret.
    4. If the bound is smaller than the tropical regret *and* the average
       similarity of the node's neighbours exceeds `similarity_threshold`,
       return True (split); otherwise False.
    """
    # 1. similarity matrix & factor
    S, nodes = similarity_matrix(features, vram_budget_mb)
    # mean similarity across all pairs (excluding diagonal)
    if S.size == 0:
        sigma = 1.0
    else:
        off_diag = S[~np.eye(S.shape[0], dtype=bool)]
        sigma = 1.0 + float(off_diag.mean())  # ensure σ ≥ 1

    # 2. bound
    bound = hybrid_hoeffding_bound(gain_estimate, n_samples, delta, regret, sigma)

    # 3. tropical regret
    trop = tropical_regret(gain_estimate, regret)

    # 4. neighbour similarity check
    neighbours = graph.get(node, set())
    if neighbours:
        idx_map = {n: i for i, n in enumerate(nodes)}
        neigh_idxs = [idx_map[n] for n in neighbours if n in idx_map]
        if neigh_idxs:
            neigh_sim = S[idx_map[node], neigh_idxs].mean()
        else:
            neigh_sim = 0.0
    else:
        neigh_sim = 0.0

    return (bound < trop) and (neigh_sim >= similarity_threshold)


# ----------------------------------------------------------------------
# Global weight store used by lsm_guided_nlms_update (demo purpose)
# ----------------------------------------------------------------------
weights_dict: Dict[Node, np.ndarray] = {}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal synthetic setup
    random.seed(42)
    np.random.seed(42)

    # Create a tiny graph of 3 nodes
    graph: Graph = {
        "A": {"B", "C"},
        "B": {"A"},
        "C": {"A"},
    }

    # Random feature vectors (dimension 5)
    features: Dict[Node, FeatureVec] = {
        n: np.random.rand(5).tolist() for n in graph.keys()
    }

    # Initialise weights and LSM vectors
    for n in graph.keys():
        weights_dict[n] = np.random.randn(5)
    lsm_vec = np.abs(np.random.randn(5))  # positive LSM magnitudes

    # Simulate a gradient for node "A"
    grad = np.random.randn(5) * 0.1

    # Perform LSM‑guided NLMS update for node "A"
    updated_weights = lsm_guided_nlms_update(
        weights=weights_dict["A"],
        lsm_vector=lsm_vec,
        gradient=grad,
        graph=graph,
        node="A",
        diffusion_rate=0.05,
    )
    weights_dict["A"] = updated_weights
    print(f"Updated weights for A: {updated_weights}")

    # Decide whether to split node "A"
    split = should_split_node(
        node="A",
        features=features,
        gain_estimate=0.6,
        n_samples=120,
        delta=0.05,
        regret=0.12,
        vram_budget_mb=2048,
        graph=graph,
        similarity_threshold=0.5,
    )
    print(f"Should split node 'A'? {split}")

    # Demonstrate tropical regret
    tr = tropical_regret(0.6, 0.12)
    print(f"Tropical regret of gain=0.6 and regret=0.12 -> {tr}")

    sys.exit(0)