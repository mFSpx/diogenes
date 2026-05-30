# DARWIN HAMMER — match 24, survivor 3
# gen: 2
# parent_a: hybrid_distributed_leader_e_thanatosis_m65_s1.py (gen1)
# parent_b: hybrid_hoeffding_tree_tropical_maxplus_m18_s0.py (gen1)
# born: 2026-05-29T23:25:24Z

"""
Hybrid Leader–Tree Election (HLTE)

This module fuses the core topologies of:

* `hybrid_distributed_leader_election.py` – probabilistic broadcast,
  simulated‑annealing acceptance and cooling schedule.
* `hybrid_hoeffding_tree_tropical_maxplus.py` – Hoeffding bound driven split
  decisions and tropical (max‑plus) algebra for aggregating piecewise‑linear
  functions.

Mathematical bridge
-------------------
Both parents rely on a *thresholded decision*: the leader election accepts a
candidate when a probabilistic acceptance exceeds a temperature‑scaled
energy difference, while the Hoeffding tree accepts a split when the observed
gain gap exceeds a Hoeffding bound.  By treating the broadcast outcome of each
node as an observed “gain” we can use the Hoeffding bound to decide whether the
evidence is sufficient to *elect* a leader.  The tropical max‑plus algebra
provides a way to propagate broadcast probabilities over the graph in a
single matrix operation (`t_matmul`), yielding a “tropical field” of
broadcast strengths that can be interpreted as the energy term `ΔE` in the
acceptance probability.

The hybrid algorithm therefore proceeds in phases:

1. **Tropical broadcast** – compute a broadcast strength vector `b` by
   repeatedly applying tropical matrix multiplication on the adjacency matrix.
2. **Hoeffding split test** – treat `b` as observed gains and apply the Hoeffding
   bound to decide which nodes have enough statistical evidence to become
   *candidate leaders*.
3. **Simulated‑annealing acceptance** – compare the candidate count change
   `ΔE` with a cooling temperature and accept the new leaders with the usual
   Metropolis probability.

The three functions below illustrate the integrated workflow.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
from collections.abc import Mapping, Hashable
import numpy as np

# ----------------------------------------------------------------------
# Shared primitives from Parent A
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]

def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance used in simulated annealing."""
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)

def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Geometric cooling schedule."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)

# ----------------------------------------------------------------------
# Tropical max‑plus primitives from Parent B
# ----------------------------------------------------------------------
def t_add(x, y):
    """Tropical addition (max)."""
    return np.maximum(x, y)

def t_mul(x, y):
    """Tropical multiplication (plus)."""
    return np.add(x, y)

def t_matmul(A, B):
    """
    Tropical matrix product: (A ⊗ B)_{ij} = max_k (A_{ik} + B_{kj})
    Works for 2‑D numpy arrays.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # broadcasting to compute A_{ik}+B_{kj} for all i,k,j
    # shape (i,k,1) + (1,k,j) -> (i,k,j) then max over k
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

# ----------------------------------------------------------------------
# Hoeffding bound utilities from Parent B
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Classic Hoeffding bound for bounded random variables."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

# ----------------------------------------------------------------------
# Hybrid building blocks
# ----------------------------------------------------------------------
def adjacency_matrix(graph: Graph) -> tuple[np.ndarray, dict[Node, int]]:
    """
    Convert a graph (node → set(neighbours)) to a dense adjacency matrix
    suitable for tropical algebra. Returns the matrix and a mapping
    ``node → index``.
    """
    nodes = list(graph.keys())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.full((n, n), -np.inf, dtype=float)  # tropical -∞ (no edge)
    for u, neigh in graph.items():
        ui = idx[u]
        for v in neigh:
            vi = idx[v]
            A[ui, vi] = 0.0  # weight 0 for a direct broadcast edge
    # Self‑loops allow a node to keep its own broadcast strength
    np.fill_diagonal(A, 0.0)
    return A, idx

def tropical_broadcast_strength(graph: Graph, phases: int, seed: int | str | None = None) -> np.ndarray:
    """
    Compute a tropical broadcast strength vector after ``phases`` iterations.
    Each iteration multiplies the current strength vector by the adjacency
    matrix in the tropical sense, mimicking a probabilistic flood.
    """
    rng = random.Random(seed)
    A, idx = adjacency_matrix(graph)
    n = A.shape[0]
    # initialise with random broadcast seeds (tropical 0 for selected nodes,
    # -inf otherwise)
    init = np.full(n, -np.inf, dtype=float)
    for i in range(n):
        if rng.random() < 0.5:  # 50 % chance to start as a seed
            init[i] = 0.0
    strength = init
    for _ in range(phases):
        # tropical matrix‑vector product: max_j (strength_j + A_{ji})
        strength = np.max(strength[:, np.newaxis] + A, axis=0)
    return strength

def hoeffding_candidate_selection(strength: np.ndarray, r: float, delta: float, n_obs: int) -> np.ndarray:
    """
    Apply the Hoeffding bound to the tropical broadcast strengths.
    Nodes whose strength exceeds the bound are marked as *candidates* (1),
    others as 0.
    """
    bound = hoeffding_bound(r, delta, n_obs)
    # Normalise strengths to [0, 1] by shifting by the minimum finite value
    finite = strength[np.isfinite(strength)]
    if finite.size == 0:
        return np.zeros_like(strength, dtype=int)
    min_val = finite.min()
    max_val = finite.max()
    norm = (strength - min_val) / (max_val - min_val + 1e-12)
    candidates = (norm > bound).astype(int)
    return candidates

def hybrid_leader_tree_election(
    graph: Graph,
    phases: int = 6,
    r: float = 1.0,
    delta: float = 0.05,
    t0: float = 1.0,
    alpha: float = 0.9,
    seed: int | str | None = None,
) -> set[Node]:
    """
    Unified algorithm:
    1. Tropical broadcast produces a strength vector.
    2. Hoeffding bound selects candidate leaders.
    3. Simulated‑annealing acceptance decides which candidates are finally
       promoted to leaders, using the change in leader count as ΔE.
    Returns the set of elected leader nodes.
    """
    # Phase‑wise loop – each phase refines the leader set.
    rng = random.Random(seed)
    leaders: set[Node] = set()
    _, idx = adjacency_matrix(graph)
    index_to_node = {i: n for n, i in idx.items()}

    for phase in range(1, phases + 1):
        # 1. tropical broadcast strength
        strength = tropical_broadcast_strength(graph, phases=phase, seed=seed)

        # 2. Hoeffding candidate selection
        # n_obs is the total number of broadcast observations so far;
        # we approximate it by phase * |V|
        n_obs = phase * len(graph)
        candidates_arr = hoeffding_candidate_selection(strength, r, delta, n_obs)

        # translate back to node identifiers
        candidate_nodes = {index_to_node[i] for i, flag in enumerate(candidates_arr) if flag}

        # 3. acceptance based on ΔE (change in leader count)
        delta_e = len(candidate_nodes) - len(leaders)
        temperature = cooling_temperature(phase, t0, alpha)
        accept_prob = acceptance_probability(delta_e, temperature)

        if rng.random() < accept_prob:
            leaders = candidate_nodes

        # optional early stop if leader set stabilises
        if phase > 1 and len(leaders) == len(candidate_nodes):
            break

    return leaders

# ----------------------------------------------------------------------
# Additional demonstration functions
# ----------------------------------------------------------------------
def broadcast_probability(phase: int, total_phases: int) -> float:
    """
    Simple deterministic probability used for comparison with the tropical
    version. Mirrors the function from Parent A.
    """
    if phase < 1 or total_phases < 1:
        raise ValueError('phase and total_phases must be positive')
    return min(1.0, 1.0 / (2 ** max(0, total_phases - phase)))

def simulate_hybrid_process(
    graph: Graph,
    total_phases: int = 8,
    seed: int | str | None = None,
) -> list[set[Node]]:
    """
    Run the hybrid election for each phase individually and collect the
    intermediate leader sets. Demonstrates the evolution of the system.
    """
    rng = random.Random(seed)
    leaders_history: list[set[Node]] = []
    current_leaders: set[Node] = set()
    for phase in range(1, total_phases + 1):
        # Use the same core routine but with a single phase
        new_leaders = hybrid_leader_tree_election(
            graph,
            phases=phase,
            seed=seed,
        )
        # probabilistic acceptance between previous and new set
        delta_e = len(new_leaders) - len(current_leaders)
        temp = cooling_temperature(phase)
        if rng.random() < acceptance_probability(delta_e, temp):
            current_leaders = new_leaders
        leaders_history.append(set(current_leaders))
    return leaders_history

def evaluate_tropical_network(x: np.ndarray, weight_matrices: list[np.ndarray]) -> np.ndarray:
    """
    Evaluate a simple tropical “network” where each layer consists of a weight
    matrix. The activation is the tropical addition (max) after each tropical
    multiplication.
    """
    h = np.asarray(x, dtype=float).ravel()
    for W in weight_matrices:
        h = t_matmul(W, h[:, np.newaxis]).ravel()
    return h

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny directed graph
    tiny_graph: dict[int, set[int]] = {
        0: {1, 2},
        1: {2, 3},
        2: {3},
        3: set(),
    }

    # Run the hybrid leader–tree election
    elected = hybrid_leader_tree_election(tiny_graph, phases=5, seed=42)
    print("Elected leaders:", elected)

    # Show phase‑wise evolution
    history = simulate_hybrid_process(tiny_graph, total_phases=5, seed=42)
    for i, s in enumerate(history, 1):
        print(f"Phase {i} leaders:", s)

    # Tiny tropical network demo
    W1 = np.array([[0, -np.inf], [-np.inf, 0]])  # identity in tropical sense
    W2 = np.array([[1, 2], [3, 0]])
    out = evaluate_tropical_network(np.array([0.0, 1.0]), [W1, W2])
    print("Tropical network output:", out)