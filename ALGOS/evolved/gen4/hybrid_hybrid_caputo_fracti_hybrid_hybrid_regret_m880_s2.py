# DARWIN HAMMER — match 880, survivor 2
# gen: 4
# parent_a: hybrid_caputo_fractional_minimum_cost_tree_m35_s7.py (gen1)
# parent_b: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s0.py (gen3)
# born: 2026-05-29T23:31:25Z

"""
Hybrid Fractional‑Memory Tree Regret Analyzer
================================================

Parents
-------
* **Parent A** – ``caputo_fractional.py`` and ``minimum_cost_tree.py``:
  Provides the Caputo fractional kernel ϕ(t‑τ;α) and the incremental
  material + path cost for each edge as a tree is grown.

* **Parent B** – ``hybrid_regret_engine_hybrid_ternary_lens_router``:
  Supplies a regret‑weighted decision framework, a MinHash projection of
  identifiers, a ternary representation of signed regret, and Shannon entropy
  over that ternary alphabet.

Mathematical Bridge
-------------------
The sequence of edge insertions **e₀,…,e_{T‑1}** is a common temporal spine.
For each insertion we obtain

* a *cost increment*   Δc_k  (Parent A),
* a *regret*           r_k   (difference to the best possible increment),
* a *hash‑based* identifier → MinHash signature,
* a *ternary* symbol   s_k = sign(r_k) ∈ {‑1,0,+1}   (Parent B).

The Caputo kernel supplies fractional weights  

 w_k = ϕ(T‑1‑k;α) / Σ_j ϕ(T‑1‑j;α) ,

which are applied **simultaneously** to the cost increments and to the
empirical distribution of the ternary symbols.  The hybrid metric is

 H = Σ_k w_k·Δc_k  + λ·H_entropy ,

where  

 H_entropy = − Σ_{s∈{‑1,0,+1}} p_s log p_s ,

and p_s is the fractional‑kernel‑weighted frequency of symbol *s*.
Thus the tree cost remembers its construction history with a power‑law
memory, while the decision‑quality (regret) is encoded in an information‑theoretic
term that shares the same memory kernel.

The implementation below realises this fusion and provides three public
functions that demonstrate the hybrid operation.
"""

from __future__ import annotations

import math
import random
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Lanczos Gamma, Caputo kernel and incremental tree cost
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of Γ(z) for real z>0."""
    if z < 0.5:
        # reflection formula
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, len(_LANCZOS_C)):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

def caputo_weights(T: int, alpha: float) -> np.ndarray:
    """Return normalized Caputo kernel weights w_k for k=0..T‑1."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0,1]")
    ks = np.arange(T - 1, -1, -1, dtype=float)   # T‑1‑k
    phi = np.power(ks, alpha - 1.0) / gamma_lanczos(alpha)  # (t‑τ)^{α‑1}/Γ(α)
    phi[ks == 0] = 0.0 if alpha < 1.0 else 1.0   # handle 0^{α‑1}
    w = phi / phi.sum()
    return w

@dataclass(frozen=True)
class Edge:
    """Undirected edge with a unique identifier."""
    u: int
    v: int
    material: float = 1.0          # intrinsic cost of the edge
    id: str = ""                    # optional human‑readable label

def _tree_distances(adj: Dict[int, List[int]]) -> Dict[int, int]:
    """Breadth‑first distances from node 0 (chosen as root)."""
    root = 0
    dist = {root: 0}
    queue = [root]
    while queue:
        cur = queue.pop(0)
        for nb in adj.get(cur, []):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    return dist

def incremental_costs(edges: List[Edge]) -> List[float]:
    """
    Simulate incremental addition of edges in the given order.
    For each insertion compute Δc = material + path_weight·ΔΣdist(root, v).
    Path weight is fixed to 1.0 for simplicity.
    """
    adj: Dict[int, List[int]] = {}
    costs: List[float] = []
    path_weight = 1.0

    for e in edges:
        # add edge to adjacency
        adj.setdefault(e.u, []).append(e.v)
        adj.setdefault(e.v, []).append(e.u)

        # recompute distances from root (node 0)
        dists = _tree_distances(adj)
        total_dist = sum(dists.values())

        # compute incremental cost: material + path contribution
        delta = e.material + path_weight * total_dist
        costs.append(delta)

    return costs

# ----------------------------------------------------------------------
# Parent B – Regret, MinHash, ternary mapping, entropy
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def minhash_signature(token: str, k: int = 64) -> List[int]:
    """Return a MinHash signature of length k for a single token."""
    if k <= 0:
        raise ValueError("k must be positive")
    return [_hash(i, token) for i in range(k)]

def ternary_from_regret(regret: float) -> int:
    """Map signed regret to ternary alphabet {-1,0,+1}."""
    if regret > 0:
        return 1
    if regret < 0:
        return -1
    return 0

def shannon_entropy(freqs: np.ndarray) -> float:
    """Compute Shannon entropy (base e) of a probability vector."""
    eps = np.finfo(float).eps
    probs = np.clip(freqs, eps, 1.0)
    return -np.sum(probs * np.log(probs))

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def compute_regrets(cost_increments: List[float]) -> List[float]:
    """
    Simple regret definition: best possible increment (minimum) minus the
    actual increment. Positive regret means the edge was costlier than the
    optimal choice at that step.
    """
    best = min(cost_increments) if cost_increments else 0.0
    return [best - c for c in cost_increments]

def fractional_weighted_entropy(ternary_symbols: List[int], alpha: float) -> float:
    """
    Compute the entropy of the ternary symbol stream using the same
    Caputo weights that are applied to the cost increments.
    """
    T = len(ternary_symbols)
    if T == 0:
        return 0.0
    w = caputo_weights(T, alpha)          # shape (T,)
    # frequencies for -1, 0, +1
    symbols = np.array(ternary_symbols)
    freq_minus = (w * (symbols == -1)).sum()
    freq_zero  = (w * (symbols == 0)).sum()
    freq_plus  = (w * (symbols == 1)).sum()
    freqs = np.array([freq_minus, freq_zero, freq_plus])
    return shannon_entropy(freqs)

def hybrid_metric(
    edges: List[Edge],
    alpha: float = 0.5,
    lambda_entropy: float = 1.0,
    minhash_k: int = 64,
) -> float:
    """
    Compute the hybrid metric H = Σ w_k·Δc_k + λ·H_entropy.

    Steps
    -----
    1. Incremental costs Δc_k from the tree growth (Parent A).
    2. Regret values r_k from Δc_k (Parent B).
    3. Ternary symbols s_k = sign(r_k).
    4. Caputo weights w_k from α.
    5. Weighted sum of costs and entropy of ternary symbols.
    """
    # 1. incremental costs
    cost_increments = incremental_costs(edges)

    # 2. regrets
    regrets = compute_regrets(cost_increments)

    # 3. ternary symbols
    ternary_symbols = [ternary_from_regret(r) for r in regrets]

    # 4. fractional weights
    T = len(cost_increments)
    w = caputo_weights(T, alpha)

    # 5. weighted cost
    weighted_cost = float(np.dot(w, cost_increments))

    # 6. entropy term (shares the same kernel)
    entropy_term = fractional_weighted_entropy(ternary_symbols, alpha)

    return weighted_cost + lambda_entropy * entropy_term

def hybrid_signature(edges: List[Edge], k: int = 64) -> List[int]:
    """
    Produce a combined MinHash signature for the whole edge sequence.
    Each edge identifier is hashed; the per‑position signature is the
    minimum across all edges (standard MinHash aggregation).
    """
    if not edges:
        return [(1 << 64) - 1] * k
    signatures = [minhash_signature(e.id or f"{e.u}-{e.v}", k) for e in edges]
    # element‑wise minimum
    agg = [min(col) for col in zip(*signatures)]
    return agg

def sample_hybrid_demo() -> Tuple[float, List[int]]:
    """
    Demonstration helper that builds a random small tree, evaluates the
    hybrid metric, and returns the metric together with the aggregated
    MinHash signature.
    """
    # Build a random tree on 5 nodes
    nodes = list(range(5))
    random.shuffle(nodes)
    edges: List[Edge] = []
    parent = nodes[0]
    for child in nodes[1:]:
        e = Edge(u=parent, v=child, material=random.uniform(0.5, 2.0), id=f"{parent}-{child}")
        edges.append(e)
        parent = child

    metric = hybrid_metric(edges, alpha=0.7, lambda_entropy=0.3, minhash_k=32)
    signature = hybrid_signature(edges, k=32)
    return metric, signature

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    metric, sig = sample_hybrid_demo()
    print(f"Hybrid metric: {metric:.6f}")
    print(f"Aggregated MinHash signature (first 8 values): {sig[:8]}")
    # sanity checks
    assert isinstance(metric, float)
    assert isinstance(sig, list) and len(sig) == 32
    print("Smoke test passed.")