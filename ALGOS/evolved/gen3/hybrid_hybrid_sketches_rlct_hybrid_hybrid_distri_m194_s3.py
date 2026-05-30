# DARWIN HAMMER — match 194, survivor 3
# gen: 3
# parent_a: hybrid_sketches_rlct_grokking_m5_s0.py (gen1)
# parent_b: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s3.py (gen2)
# born: 2026-05-29T23:27:29Z

"""Hybrid Sketch‑RLCT Leader Election

Parents:
* **hybrid_sketches_rlct_grokking_m5_s0.py** – provides Count‑Min sketch,
  MinHash LSH and a Real Log Canonical Threshold (RLCT) estimator derived from
  training losses.
* **hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s3.py** – provides a
  tropical (max‑plus) broadcast on a graph, a Hoeffding‑bound split test and a
  simulated‑annealing acceptance rule.

Mathematical bridge
-------------------
The Count‑Min sketch reduces a high‑dimensional stream to a compact frequency
matrix.  The RLCT, estimated from the non‑zero sketch counts, quantifies the
information loss incurred by this dimensionality reduction.  In the leader‑
election side, the tropical broadcast yields a vector of “broadcast strengths”
that we interpret as observed gains.  The Hoeffding bound decides whether a
node has gathered enough statistical evidence to become a *candidate leader*.
  By feeding the RLCT‑derived information‑loss term into the simulated‑annealing
temperature, the algorithm balances the trade‑off between aggressive reduction
(sketch) and reliable leader election (Hoeffding + tropical algebra)."""

import hashlib
import math
import random
import sys
import pathlib
from collections import defaultdict
from collections.abc import Mapping, Hashable
import numpy as np

# ----------------------------------------------------------------------
# Sketch & RLCT utilities (Parent A)
# ----------------------------------------------------------------------
def count_min_sketch(items, width: int = 64, depth: int = 4) -> list[list[int]]:
    """Classic Count‑Min sketch."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table


def estimate_rlct_from_losses(train_losses_per_n, n_values):
    """Estimate the Real Log Canonical Threshold from a sequence of losses."""
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)
    if np.any(ns <= np.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if len(losses) != len(ns):
        raise ValueError("train_losses_per_n and n_values must have the same length")
    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    return float((x_c * y_c).sum() / var_x)


# ----------------------------------------------------------------------
# Tropical (max‑plus) linear algebra (Parent B)
# ----------------------------------------------------------------------
def t_max_plus_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tropical matrix multiplication: (A ⊗ B)[i,j] = max_k (A[i,k] + B[k,j])"""
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible shapes for tropical multiplication")
    # Use broadcasting to compute all pairwise sums, then max over k
    # Result shape (A_rows, B_cols)
    result = np.full((A.shape[0], B.shape[1]), -np.inf)
    for k in range(A.shape[1]):
        candidate = A[:, k, None] + B[k, None, :]
        result = np.maximum(result, candidate)
    return result


def tropical_broadcast_strengths(adj: Mapping[Node, set[Node]],
                                 steps: int = 5,
                                 init_strength: float = 0.0) -> dict[Node, float]:
    """
    Propagate broadcast strengths over the graph using tropical matrix powers.
    Each node starts with ``init_strength``; after ``steps`` tropical multiplications
    the resulting vector approximates the max‑plus reachability.
    """
    nodes = list(adj.keys())
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    # Build adjacency matrix A where A[i,j] = 0 if edge i->j exists, else -inf
    A = np.full((n, n), -np.inf)
    for src, targets in adj.items():
        i = index[src]
        for tgt in targets:
            j = index[tgt]
            A[i, j] = 0.0  # cost‑free edge in tropical algebra

    # Initialise strength vector s (row vector) with init_strength on the diagonal
    s = np.full((1, n), -np.inf)
    for i in range(n):
        s[0, i] = init_strength

    # Repeated tropical multiplication: s ← s ⊗ A
    for _ in range(steps):
        s = t_max_plus_matmul(s, A)

    # Convert back to dict
    strengths = {node: float(s[0, idx]) for node, idx in index.items()}
    return strengths


# ----------------------------------------------------------------------
# Hoeffding bound & candidate selection (Parent B)
# ----------------------------------------------------------------------
def hoeffding_bound(epsilon: float, n: int, R: float = 1.0) -> float:
    """Hoeffding bound for bounded random variables in [0,R]."""
    if n <= 0:
        raise ValueError("n must be positive")
    return math.sqrt((R ** 2 * math.log(2 / epsilon)) / (2 * n))


def select_candidate_leaders(strengths: dict[Node, float],
                             epsilon: float = 0.05,
                             min_samples: int = 10) -> set[Node]:
    """
    Treat each node's broadcast strength as an observed gain.
    A node becomes a candidate if its strength exceeds the Hoeffding bound
    computed from the number of observations (here approximated by min_samples).
    """
    candidates = set()
    for node, gain in strengths.items():
        bound = hoeffding_bound(epsilon, min_samples)
        if gain > bound:
            candidates.add(node)
    return candidates


def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance used in simulated annealing."""
    if delta_e < 0:
        return 1.0
    try:
        return math.exp(-delta_e / temperature)
    except OverflowError:
        return 0.0


# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_leader_election(data_items,
                           graph: Mapping[Node, set[Node]],
                           sketch_width: int = 64,
                           sketch_depth: int = 4,
                           broadcast_steps: int = 5,
                           epsilon: float = 0.05,
                           min_samples: int = 10) -> set[Node]:
    """
    End‑to‑end hybrid procedure:

    1. **Sketch & RLCT** – compress ``data_items`` with a Count‑Min sketch,
       extract the non‑zero frequencies, and estimate the RLCT.  The RLCT is
       interpreted as an *information‑loss temperature*.
    2. **Tropical broadcast** – compute broadcast strengths on ``graph``.
    3. **Hoeffding candidate test** – pick nodes whose strength exceeds the
       Hoeffding bound.
    4. **Simulated‑annealing acceptance** – accept each candidate with a
       probability that depends on the RLCT‑derived temperature and the
       change in the number of leaders (ΔE = 1 per new leader).

    Returns the final set of elected leaders.
    """
    # --- Step 1: sketch & RLCT -------------------------------------------------
    sketch = count_min_sketch(data_items, width=sketch_width, depth=sketch_depth)
    flat = [cnt for row in sketch for cnt in row if cnt > 0]
    if len(flat) < 2:
        # Not enough information to estimate RLCT; fall back to a default temperature
        temperature = 1.0
    else:
        n_vals = list(range(1, len(flat) + 1))
        rlct = estimate_rlct_from_losses(flat, n_vals)
        # Map RLCT (which can be negative) to a positive temperature scale.
        # We use an exponential mapping to keep temperature > 0.
        temperature = math.exp(-rlct) if rlct != float('inf') else 1.0
        temperature = max(temperature, 1e-3)  # avoid division by zero later

    # --- Step 2: tropical broadcast --------------------------------------------
    strengths = tropical_broadcast_strengths(graph, steps=broadcast_steps)

    # --- Step 3: Hoeffding candidate selection ----------------------------------
    candidates = select_candidate_leaders(strengths, epsilon=epsilon, min_samples=min_samples)

    # --- Step 4: simulated‑annealing acceptance ---------------------------------
    elected = set()
    for node in candidates:
        delta_e = 1.0  # each new leader adds one unit of “energy”
        prob = acceptance_probability(delta_e, temperature)
        if random.random() < prob:
            elected.add(node)

    return elected


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic data stream
    data = [random.randint(0, 1000) for _ in range(500)]

    # Small directed graph
    G = {
        "A": {"B", "C"},
        "B": {"C", "D"},
        "C": {"D"},
        "D": {"A"},
        "E": {"A", "D"},
    }

    leaders = hybrid_leader_election(
        data_items=data,
        graph=G,
        sketch_width=32,
        sketch_depth=3,
        broadcast_steps=4,
        epsilon=0.1,
        min_samples=20,
    )
    print("Elected leaders:", leaders)