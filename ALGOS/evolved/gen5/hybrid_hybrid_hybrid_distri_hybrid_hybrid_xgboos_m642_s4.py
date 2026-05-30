# DARWIN HAMMER — match 642, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s3.py (gen2)
# parent_b: hybrid_hybrid_xgboost_objec_hybrid_hybrid_regret_m283_s2.py (gen4)
# born: 2026-05-29T23:30:12Z

"""Hybrid Leader–Tree XGBoost‑Regret Algorithm (HLTXR)

This module fuses the two parent algorithms:

* **Parent A** – Hybrid Leader‑Tree Election (tropical max‑plus broadcast,
  Hoeffding‑bound split test, simulated‑annealing acceptance).
* **Parent B** – Hybrid XGBoost‑Regret Analyzer (logistic gradient/hessian,
  MinHash similarity & Shannon‑entropy regulariser).

**Mathematical bridge**

The broadcast strength vector `b` produced by the tropical max‑plus
propagation is interpreted as a *margin* `m` for a binary logistic loss.
The XGBoost utilities (gradient, hessian, split‑gain) are therefore
applicable to the election problem.  To embed the regret‑weighted
information from Parent B we augment the logistic loss with the term
`α·s·H` where `s` is a MinHash‑derived similarity between a node’s token
set and a reference set, and `H` is the Shannon entropy of the combined
signature distribution.  The resulting adjusted gradient/hessian are fed
into the Hoeffding‑bound test that decides whether the observed gain is
statistically sufficient to *elect* a leader.  Finally a simulated‑annealing
Metropolis step accepts or rejects the new leader set.

The three public functions below demonstrate the integrated workflow:
* `tropical_matmul` – max‑plus matrix multiplication.
* `adjusted_grad_hess` – logistic gradient/hessian with similarity‑entropy
  regularisation.
* `hybrid_leader_tree_step` – one iteration of the fused algorithm.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
from collections import Counter
from typing import Mapping, Hashable, Set, Tuple, List, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
TokenSet = Set[str]

# ----------------------------------------------------------------------
# Tropical (max‑plus) matrix operations – Parent A core
# ----------------------------------------------------------------------
def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Max‑plus matrix multiplication: (A ⊗ B)[i,j] = max_k (A[i,k] + B[k,j])"""
    n, m = A.shape
    p = B.shape[1]
    result = np.full((n, p), -np.inf, dtype=float)
    for i in range(n):
        for k in range(m):
            if np.isneginf(A[i, k]):
                continue
            tmp = A[i, k] + B[k, :]  # vector addition
            result[i, :] = np.maximum(result[i, :], tmp)
    return result


def tropical_power(adj: np.ndarray, steps: int) -> np.ndarray:
    """Iteratively apply tropical multiplication to propagate broadcast."""
    power = np.copy(adj)
    acc = np.copy(adj)
    for _ in range(steps - 1):
        power = tropical_matmul(power, adj)
        acc = np.maximum(acc, power)
    return acc


def compute_broadcast_strength(adj: np.ndarray, steps: int = 3) -> np.ndarray:
    """Return a vector of broadcast strengths for each node."""
    reach = tropical_power(adj, steps)
    # strength of node i is max over its row (including self‑loops)
    return np.max(reach, axis=1)


# ----------------------------------------------------------------------
# Hoeffding bound – Parent A core
# ----------------------------------------------------------------------
def hoeffding_bound(num_samples: int, range_R: float, delta: float) -> float:
    """Hoeffding bound ε = sqrt(R^2 * ln(1/δ) / (2·n))."""
    if num_samples <= 0:
        return float('inf')
    return math.sqrt((range_R ** 2 * math.log(1.0 / delta)) / (2.0 * num_samples))


# ----------------------------------------------------------------------
# MinHash similarity & Shannon entropy – Parent B core (simplified)
# ----------------------------------------------------------------------
def jaccard_similarity(a: TokenSet, b: TokenSet) -> float:
    """Simple Jaccard similarity (used as a proxy for MinHash similarity)."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def shannon_entropy(tokens: Iterable[str]) -> float:
    """Shannon entropy of token frequency distribution."""
    cnt = Counter(tokens)
    total = sum(cnt.values())
    if total == 0:
        return 0.0
    ent = -sum((c / total) * math.log(c / total, 2) for c in cnt.values())
    return ent


# ----------------------------------------------------------------------
# Logistic gradient / hessian with similarity‑entropy regulariser – Parent B core
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    x_arr = np.asarray(x, dtype=float)
    out = np.empty_like(x_arr)
    pos = x_arr >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(out)
    return out


def adjusted_grad_hess(
    y: np.ndarray,
    margin: np.ndarray,
    similarity: np.ndarray,
    entropy: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient and hessian of the regularised logistic loss:

        L = ℓ(y, m) + α·s·H

    where ℓ is binary logistic loss, s similarity, H entropy.
    The regulariser contributes a multiplicative factor (1 + α·s·H) to both
    gradient and hessian.
    """
    prob = sigmoid(margin)
    base_grad = prob - y                       # ∂ℓ/∂m
    base_hess = prob * (1.0 - prob)            # ∂²ℓ/∂m²
    factor = 1.0 + alpha * similarity * entropy
    grad = base_grad * factor
    hess = base_hess * factor
    return grad, hess


def split_gain(
    grad_left: np.ndarray,
    hess_left: np.ndarray,
    grad_right: np.ndarray,
    hess_right: np.ndarray,
    lambda_reg: float = 1.0,
) -> float:
    """
    XGBoost split gain formula with L2 regularisation λ.
    """
    def gain(g, h):
        return (g @ g) / (h + lambda_reg)

    total_gain = gain(grad_left, hess_left) + gain(grad_right, hess_right)
    parent_gain = gain(grad_left + grad_right, hess_left + hess_right)
    return 0.5 * (total_gain - parent_gain)


# ----------------------------------------------------------------------
# Simulated‑annealing acceptance – Parent A core
# ----------------------------------------------------------------------
def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance probability."""
    if delta_e < 0:
        return 1.0
    return math.exp(-delta_e / temperature)


# ----------------------------------------------------------------------
# Hybrid iteration combining all pieces
# ----------------------------------------------------------------------
def hybrid_leader_tree_step(
    graph: Graph,
    node_tokens: Mapping[Node, TokenSet],
    reference_tokens: TokenSet,
    current_leaders: Set[Node],
    temperature: float,
    alpha: float = 0.5,
    delta: float = 0.05,
    hoeffding_delta: float = 0.01,
    steps_tropical: int = 3,
) -> Set[Node]:
    """
    Perform one hybrid HLTXR iteration.

    1. Compute tropical broadcast strengths → margin vector `m`.
    2. For each node compute similarity `s` and entropy `H` w.r.t. the
       reference token set.
    3. Obtain adjusted gradient/hessian using `adjusted_grad_hess`.
    4. For each edge (u, v) evaluate a split gain between the two
       sub‑populations defined by the edge.  The gain is tested with a
       Hoeffding bound; if significant, the node with larger gain becomes
       a *candidate leader*.
    5. Apply Metropolis acceptance based on the change in number of leaders.
    """
    # --- 1. Tropical broadcast -------------------------------------------------
    nodes = list(graph.keys())
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    adj = np.full((n, n), -np.inf, dtype=float)
    for u, neigh in graph.items():
        i = index[u]
        adj[i, i] = 0.0                     # self‑loop (zero cost)
        for v in neigh:
            j = index[v]
            adj[i, j] = 0.0                 # edge weight = 0 in max‑plus sense
    broadcast = compute_broadcast_strength(adj, steps=steps_tropical)
    margin = broadcast                       # treat strength as logistic margin

    # --- 2. Similarity & entropy ------------------------------------------------
    sim_vec = np.empty(n, dtype=float)
    ent_vec = np.empty(n, dtype=float)
    for node, i in index.items():
        tokens = node_tokens.get(node, set())
        sim = jaccard_similarity(tokens, reference_tokens)
        # entropy of the union of both token multisets (approximate)
        union_tokens = list(tokens) + list(reference_tokens)
        ent = shannon_entropy(union_tokens)
        sim_vec[i] = sim
        ent_vec[i] = ent

    # --- 3. Adjusted gradient / hessian ----------------------------------------
    y = np.array([1.0 if node in current_leaders else 0.0 for node in nodes], dtype=float)
    grad, hess = adjusted_grad_hess(y, margin, sim_vec, ent_vec, alpha=alpha)

    # --- 4. Hoeffding‑bound split test -----------------------------------------
    candidate_leaders: Set[Node] = set()
    for u in graph:
        i = index[u]
        # define left side = {u} , right side = rest of nodes
        left_mask = np.zeros(n, dtype=bool)
        left_mask[i] = True
        right_mask = ~left_mask

        grad_left = grad[left_mask]
        hess_left = hess[left_mask]
        grad_right = grad[right_mask]
        hess_right = hess[right_mask]

        gain = split_gain(grad_left, hess_left, grad_right, hess_right)

        # Hoeffding bound: we treat gain as observed mean, range_R=1 (since
        # gradients are bounded by [-1,1]), n = number of samples in left+right.
        bound = hoeffding_bound(num_samples=n, range_R=1.0, delta=hoeffding_delta)

        if gain - bound > 0:                # statistically significant improvement
            candidate_leaders.add(u)

    # --- 5. Simulated‑annealing acceptance --------------------------------------
    delta_e = len(candidate_leaders) - len(current_leaders)
    prob = acceptance_probability(delta_e, temperature)
    if random.random() < prob:
        return candidate_leaders
    else:
        return current_leaders


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple undirected triangle graph
    G: Graph = {
        "A": {"B", "C"},
        "B": {"A", "C"},
        "C": {"A", "B"},
    }

    # Token sets per node (toy example)
    tokens: Mapping[Node, TokenSet] = {
        "A": {"apple", "banana"},
        "B": {"banana", "cherry"},
        "C": {"apple", "cherry"},
    }

    reference: TokenSet = {"apple", "banana", "cherry", "date"}

    leaders: Set[Node] = set()
    temp = 1.0
    for epoch in range(5):
        leaders = hybrid_leader_tree_step(
            graph=G,
            node_tokens=tokens,
            reference_tokens=reference,
            current_leaders=leaders,
            temperature=temp,
            alpha=0.3,
            delta=0.05,
            hoeffding_delta=0.01,
            steps_tropical=3,
        )
        print(f"Epoch {epoch}: leaders = {sorted(leaders)}")
        temp *= 0.8  # simple cooling schedule
    sys.exit(0)