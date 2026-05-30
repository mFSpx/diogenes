# DARWIN HAMMER — match 974, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_krampu_m9_s5.py (gen3)
# parent_b: hybrid_hybrid_ternary_route_hybrid_endpoint_circ_m233_s2.py (gen3)
# born: 2026-05-29T23:32:00Z

"""Hybrid Bandit‑Tree‑Curvature‑Morphology Algorithm
=================================================
Parents:
- **hybrid_hybrid_bandit_router_hybrid_hybrid_krampu_m9_s5.py**  
  Provides a contextual bandit that selects a graph node, a Honey‑bee
  store delta that perturbs the adjacency matrix, and a simple
  Ollivier‑Ricci curvature recomputation.
- **hybrid_hybrid_ternary_route_hybrid_endpoint_circ_m233_s2.py**  
  Supplies geometry → morphology conversion, sphericity/flatness
  descriptors, a Bayesian edge‑probability update and a circuit‑breaker
  threshold that depends on the morphology.

Mathematical Bridge
-------------------
The selected bandit node becomes the *root* of a spanning‑tree built on the
graph.  The Honey‑bee delta modifies the adjacency matrix **A**; from the
updated **A** we compute a curvature matrix **C** (treated as a prior for
edge reliability).  The node positions give a bounding‑box morphology
(length L, width W, height H); from L,W,H we derive sphericity **S** and
flatness **F**.  **S** and **F** modulate the circuit‑breaker threshold
`τ = τ₀·(1+α·(1‑S)+β·F)`.  The curvature matrix supplies Bayesian priors
for edge probabilities `p(e) = λ·C_ij + (1‑λ)·ℓ(e)`, where `ℓ(e)` is the
geometric length.  The expected cost of the tree is then

    cost = τ · Σ_{e∈T} p(e)·ℓ(e)

Thus the algorithm closes a loop:
features → bandit → root → adjacency update → curvature → morphology →
threshold → Bayesian edge weights → expected tree cost.

The implementation below provides a compact, runnable hybrid pipeline.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Shared Bandit / Store components (Parent A)
# ----------------------------------------------------------------------

_POLICY: Dict[str, List[float]] = {}  # action_id -> [total_reward, count]

def reset_policy() -> None:
    """Clear the global reward statistics."""
    _POLICY.clear()

def update_policy(updates: List["BanditUpdate"]) -> None:
    """Accumulate rewards for each action."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def _mean_reward(action: str) -> float:
    """Mean reward observed for *action*."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _action_counts() -> float:
    """Total number of pulls across all actions (used for UCB)."""
    return sum(cnt for _, cnt in _POLICY.values())

@dataclass(frozen=True)
class BanditUpdate:
    action_id: str
    reward: float

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "UCB"

def bandit_select_action(
    features: Dict[str, np.ndarray],
    exploration_coef: float = 2.0,
) -> BanditAction:
    """
    Upper‑Confidence‑Bound (UCB) selection.
    *features* maps an action_id to a feature vector (unused in the pure UCB
    formulation but kept for compatibility with the original bandit).
    """
    total_counts = _action_counts() + 1e-9  # avoid division by zero
    best_score = -math.inf
    best_action: str = ""
    for aid in features.keys():
        mu = _mean_reward(aid)
        n = _POLICY.get(aid, [0.0, 0.0])[1] + 1e-9
        ucb = mu + exploration_coef * math.sqrt(math.log(total_counts) / n)
        if ucb > best_score:
            best_score = ucb
            best_action = aid
    # Propensity is a softmax‑like probability derived from the scores
    prop = math.exp(best_score) / sum(math.exp(_mean_reward(a) + exploration_coef *
                     math.sqrt(math.log(total_counts) / (_POLICY.get(a, [0.0, 0.0])[1] + 1e-9)))
                     for a in features.keys())
    return BanditAction(
        action_id=best_action,
        propensity=prop,
        expected_reward=_mean_reward(best_action),
        confidence_bound=best_score,
    )

# ----------------------------------------------------------------------
# Graph curvature utilities (derived from Parent A)
# ----------------------------------------------------------------------

def curvature_from_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Very lightweight surrogate for Ollivier‑Ricci curvature.
    For each edge (i,j) we set
        C_ij = A_ij * (deg_i + deg_j) / (2 * m)
    where m = total number of edges (counted with weight).
    The matrix is symmetric and zero on the diagonal.
    """
    deg = adj.sum(axis=1)
    m = deg.sum() / 2.0 + 1e-9
    C = np.zeros_like(adj)
    rows, cols = np.where(adj > 0)
    for i, j in zip(rows, cols):
        if i == j:
            continue
        C[i, j] = adj[i, j] * (deg[i] + deg[j]) / (2.0 * m)
    return C

def apply_honeybee_delta(adj: np.ndarray, node_idx: int, delta: float) -> np.ndarray:
    """
    The Honey‑bee store delta perturbs all edges incident to *node_idx*.
    Positive delta strengthens connections, negative weakens them.
    """
    new_adj = adj.copy()
    new_adj[node_idx, :] = np.clip(new_adj[node_idx, :] + delta, 0.0, None)
    new_adj[:, node_idx] = np.clip(new_adj[:, node_idx] + delta, 0.0, None)
    # Keep diagonal zero (no self‑loops)
    np.fill_diagonal(new_adj, 0.0)
    return new_adj

# ----------------------------------------------------------------------
# Geometry → Morphology (Parent B)
# ----------------------------------------------------------------------

Point2D = Tuple[float, float]

def bounding_box(points: List[Point2D]) -> Tuple[float, float, float]:
    """Return (length, width, height) of the axis‑aligned bounding box.
    Height is set to zero for pure 2‑D data (kept for API compatibility)."""
    xs, ys = zip(*points)
    length = max(xs) - min(xs)
    width = max(ys) - min(ys)
    height = 0.0
    return (length, width, height)

def sphericity_and_flatness(morph: Tuple[float, float, float]) -> Tuple[float, float]:
    """Simple shape descriptors.
    - Sphericity S = min(dim) / max(dim)  (1 → perfect sphere)
    - Flatness    F = (max(dim) - min(dim)) / max(dim) (0 → isotropic)"""
    dims = np.array(morph)
    max_dim = dims.max()
    min_dim = dims.min()
    if max_dim == 0:
        return (0.0, 0.0)
    S = min_dim / max_dim
    F = (max_dim - min_dim) / max_dim
    return (S, F)

def breaker_threshold(
    base: float,
    sphericity: float,
    flatness: float,
    alpha: float = 0.5,
    beta: float = 0.3,
) -> float:
    """Circuit‑breaker failure threshold modulated by morphology."""
    return base * (1.0 + alpha * (1.0 - sphericity) + beta * flatness)

# ----------------------------------------------------------------------
# Bayesian edge probability update (Parent B)
# ----------------------------------------------------------------------

def bayesian_edge_probabilities(
    curvature: np.ndarray,
    lengths: np.ndarray,
    lambda_prior: float = 0.6,
) -> np.ndarray:
    """
    Combine curvature (treated as a reliability prior) with geometric
    length (treated as a likelihood).  Both matrices are symmetric and
    zero on the diagonal.
    """
    # Normalise curvature to [0,1]
    Cnorm = curvature - curvature.min()
    if Cnorm.max() > 0:
        Cnorm = Cnorm / Cnorm.max()
    else:
        Cnorm = np.zeros_like(Cnorm)

    # Normalise lengths to [0,1] (shorter edges are more reliable)
    Lnorm = lengths - lengths.min()
    if Lnorm.max() > 0:
        Lnorm = 1.0 - (Lnorm / Lnorm.max())
    else:
        Lnorm = np.ones_like(Lnorm)

    # Bayesian blend
    prob = lambda_prior * Cnorm + (1.0 - lambda_prior) * Lnorm
    return prob

# ----------------------------------------------------------------------
# Expected tree cost (hybrid core)
# ----------------------------------------------------------------------

def expected_tree_cost(
    nodes: List[str],
    positions: Dict[str, Point2D],
    adjacency: np.ndarray,
    root: str,
    threshold: float,
) -> float:
    """
    Build a minimum‑spanning‑tree (MST) rooted at *root* using the current
    adjacency weights as edge costs, then compute the expected cost using
    Bayesian edge probabilities.
    """
    n = len(nodes)
    idx_map = {node: i for i, node in enumerate(nodes)}
    # Edge lengths matrix
    lengths = np.zeros((n, n))
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i == j:
                continue
            pi = positions[ni]
            pj = positions[nj]
            lengths[i, j] = math.hypot(pi[0] - pj[0], pi[1] - pj[1])

    # Curvature as prior
    curvature = curvature_from_adjacency(adjacency)

    # Bayesian probabilities
    edge_probs = bayesian_edge_probabilities(curvature, lengths)

    # Prim's algorithm for MST (simple O(n^2) implementation)
    in_mst = [False] * n
    key = [math.inf] * n
    parent = [-1] * n
    root_idx = idx_map[root]
    key[root_idx] = 0.0

    for _ in range(n):
        # pick the vertex with minimum key not yet in MST
        u = min((i for i in range(n) if not in_mst[i]), key=lambda i: key[i])
        in_mst[u] = True
        for v in range(n):
            if not in_mst[v] and adjacency[u, v] > 0:
                w = adjacency[u, v]  # use current edge weight as cost
                if w < key[v]:
                    key[v] = w
                    parent[v] = u

    # Accumulate expected cost over the selected edges
    total = 0.0
    for v in range(n):
        u = parent[v]
        if u == -1:
            continue
        prob = edge_probs[u, v]
        geom_len = lengths[u, v]
        total += prob * geom_len

    return threshold * total

# ----------------------------------------------------------------------
# High‑level hybrid pipeline (demonstrates the fusion)
# ----------------------------------------------------------------------

def hybrid_step(
    node_features: Dict[str, np.ndarray],
    positions: Dict[str, Point2D],
    adjacency: np.ndarray,
    store_delta: float,
    base_threshold: float = 1.0,
) -> Tuple[float, str]:
    """
    Execute one hybrid iteration:
    1. Bandit selects a node (treated as root).
    2. Honey‑bee delta perturbs adjacency around the root.
    3. Curvature is recomputed.
    4. Morphology → sphericity/flatness → breaker threshold.
    5. Expected tree cost is returned.
    """
    # 1. Bandit decision
    action = bandit_select_action(node_features)
    root = action.action_id

    # 2. Apply delta to adjacency
    root_idx = list(node_features.keys()).index(root)
    adj_prime = apply_honeybee_delta(adjacency, root_idx, store_delta)

    # 3. Curvature (implicitly used later)
    # (already computed inside expected_tree_cost)

    # 4. Morphology & threshold
    points = [positions[n] for n in node_features.keys()]
    morph = bounding_box(points)
    S, F = sphericity_and_flatness(morph)
    τ = breaker_threshold(base_threshold, S, F)

    # 5. Expected cost
    cost = expected_tree_cost(
        nodes=list(node_features.keys()),
        positions=positions,
        adjacency=adj_prime,
        root=root,
        threshold=τ,
    )
    return cost, root

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Define a tiny graph with 4 nodes
    node_ids = ["A", "B", "C", "D"]
    # Random 2‑D positions
    positions = {
        nid: (random.uniform(0, 10), random.uniform(0, 10)) for nid in node_ids
    }

    # Random feature vectors (dimension 3)
    node_features = {
        nid: np.random.rand(3) for nid in node_ids
    }

    # Initial adjacency (symmetric, no self‑loops)
    adj = np.zeros((4, 4))
    for i in range(4):
        for j in range(i + 1, 4):
            w = random.uniform(0.1, 1.0)
            adj[i, j] = adj[j, i] = w

    # Initialise bandit policy with some dummy updates
    reset_policy()
    dummy_updates = [
        BanditUpdate(action_id="A", reward=1.0),
        BanditUpdate(action_id="B", reward=0.5),
        BanditUpdate(action_id="C", reward=0.2),
        BanditUpdate(action_id="D", reward=0.8),
    ]
    update_policy(dummy_updates)

    # Run a single hybrid iteration
    delta = 0.2  # honey‑bee store delta
    cost, root = hybrid_step(
        node_features=node_features,
        positions=positions,
        adjacency=adj,
        store_delta=delta,
        base_threshold=1.0,
    )

    print(f"Selected root node: {root}")
    print(f"Expected hybrid cost: {cost:.4f}")
    sys.exit(0)