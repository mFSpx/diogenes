# DARWIN HAMMER — match 5534, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m1512_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m560_s4.py (gen5)
# born: 2026-05-30T00:02:42Z

"""Hybrid Leader Election & Simulated Annealing

This module fuses the core topologies of two parent algorithms:

* **Parent A** – a distributed leader election that uses pheromone‑guided
  maximal‑independent‑set selection together with a simulated‑annealing style
  probabilistic acceptance rule.
* **Parent B** – a lightweight pheromone record with exponential decay and a
  set of stylometric utilities.

The mathematical bridge is the **energy function** used in simulated annealing.
In Parent A the acceptance probability depends on an energy difference ΔE.
In Parent B each node carries a pheromone value that decays exponentially.
We therefore define the energy of a candidate leader set **L** as

    E(L) = - Σ_{i∈L} ϕ_i          (reward proportional to pheromone)
           + λ Σ_{i,j∈L, i≠j} A_{ij}   (penalty for violating independence)

where ϕ_i is the current pheromone strength of node *i*, A is the adjacency
matrix of the graph, and λ>0 controls the independence penalty.
The acceptance probability is

    p = min(1, exp(-ΔE / T))

with temperature *T* supplied by a schedule.  After each accepted move we
increase the pheromone of the newly elected leaders and decay all pheromones
according to the exponential rule from Parent B.  The resulting system
performs distributed leader election while continuously adapting pheromone
trails via simulated‑annealing dynamics.

The implementation below provides three public functions that showcase this
hybrid operation:
    * `compute_phash` – perceptual hash utility (from Parent A).
    * `apply_decay` – decay all pheromone entries (from Parent B).
    * `anneal_leader_election` – full hybrid annealing loop.
"""

import sys
import random
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Hashable, Set, List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]

# ----------------------------------------------------------------------
# Parent A utilities (perceptual hash)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Perceptual hash: one bit per value indicating >= average (max 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

# ----------------------------------------------------------------------
# Parent B pheromone entry with exponential decay
# ----------------------------------------------------------------------
class PheromoneEntry:
    """Lightweight pheromone record with exponential decay."""

    __slots__ = (
        "uuid",
        "node",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(self, node: Node, signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.node = node
        self.signal_value = float(signal_value)
        self.half_life_seconds = max(1, half_life_seconds)  # avoid zero
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        """Seconds elapsed since last decay."""
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative decay factor since last decay event."""
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        """Apply exponential decay to the stored signal value."""
        self.signal_value *= self.decay_factor()
        self.last_decay = datetime.now(timezone.utc)

# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def adjacency_matrix(graph: Graph) -> Tuple[np.ndarray, List[Node]]:
    """
    Convert a graph into a NumPy adjacency matrix.
    Returns the matrix and the ordered list of nodes.
    """
    nodes = list(graph.keys())
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    mat = np.zeros((n, n), dtype=np.float64)
    for src, nbrs in graph.items():
        i = index[src]
        for dst in nbrs:
            if dst in index:
                j = index[dst]
                mat[i, j] = 1.0
    return mat, nodes


def energy_of_leader_set(
    leader_set: Set[Node],
    pheromones: Dict[Node, PheromoneEntry],
    adj_mat: np.ndarray,
    node_order: List[Node],
    independence_lambda: float = 1.0,
) -> float:
    """
    Compute the energy of a candidate leader set.
    E = - Σ ϕ_i   + λ Σ A_ij for i,j in leaders, i≠j
    Lower energy is better.
    """
    # reward term (negative pheromone sum)
    reward = -sum(pheromones[n].signal_value for n in leader_set if n in pheromones)

    # penalty term for adjacency violations
    idx = [node_order.index(n) for n in leader_set if n in node_order]
    if len(idx) <= 1:
        penalty = 0.0
    else:
        sub = adj_mat[np.ix_(idx, idx)]
        # subtract diagonal (self‑loops) and count each edge twice
        penalty = independence_lambda * (sub.sum() - np.trace(sub)) / 2.0
    return reward + penalty


def apply_decay(pheromones: Dict[Node, PheromoneEntry]) -> None:
    """Decay all pheromone entries in place (Parent B operation)."""
    for entry in pheromones.values():
        entry.apply_decay()


def anneal_leader_election(
    graph: Graph,
    initial_leaders: Set[Node],
    pheromones: Dict[Node, PheromoneEntry],
    temperature_schedule: List[float],
    independence_lambda: float = 1.0,
    pheromone_increment: float = 0.1,
) -> Set[Node]:
    """
    Hybrid simulated‑annealing leader election.

    Parameters
    ----------
    graph : mapping of node → set(neighbours)
        Undirected graph on which leaders are selected.
    initial_leaders : set
        Starting independent set (may be empty).
    pheromones : dict node → PheromoneEntry
        Current pheromone strengths.
    temperature_schedule : list of float
        Temperatures T_t for each annealing iteration.
    independence_lambda : float, optional
        Weight of the adjacency penalty term.
    pheromone_increment : float, optional
        Amount added to pheromone of nodes that become leaders.

    Returns
    -------
    set
        Final leader set after the annealing schedule.
    """
    adj_mat, order = adjacency_matrix(graph)
    leaders = set(initial_leaders)

    for step, T in enumerate(temperature_schedule):
        # 1. Decay pheromones globally (continuous time approximation)
        apply_decay(pheromones)

        # 2. Propose a move: flip the membership of a random node
        candidate = random.choice(order)
        new_leaders = set(leaders)
        if candidate in leaders:
            new_leaders.remove(candidate)
        else:
            new_leaders.add(candidate)

        # 3. Compute energy difference
        E_current = energy_of_leader_set(leaders, pheromones, adj_mat, order, independence_lambda)
        E_proposed = energy_of_leader_set(new_leaders, pheromones, adj_mat, order, independence_lambda)
        delta_E = E_proposed - E_current

        # 4. Acceptance test (simulated annealing rule)
        accept = False
        if delta_E <= 0:
            accept = True
        else:
            prob = math.exp(-delta_E / max(T, 1e-12))
            accept = random.random() < prob

        if accept:
            leaders = new_leaders
            # 5. Reinforce pheromone for nodes that are leaders after the move
            for node in leaders:
                if node in pheromones:
                    pheromones[node].signal_value += pheromone_increment
                else:
                    # initialise missing entries with a default half‑life
                    pheromones[node] = PheromoneEntry(node, pheromone_increment, half_life_seconds=60)

    return leaders


# ----------------------------------------------------------------------
# Simple demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a small undirected graph (a 5‑node ring)
    G: Dict[int, Set[int]] = {
        0: {1, 4},
        1: {0, 2},
        2: {1, 3},
        3: {2, 4},
        4: {3, 0},
    }

    # Initialise pheromone entries with random strengths
    pheromone_dict: Dict[int, PheromoneEntry] = {}
    for node in G:
        init_val = random.uniform(0.5, 1.5)
        pheromone_dict[node] = PheromoneEntry(node, init_val, half_life_seconds=30)

    # Temperature schedule: geometric cooling
    temps = [10.0 * (0.95 ** i) for i in range(100)]

    # Run hybrid annealing
    final_leaders = anneal_leader_election(
        graph=G,
        initial_leaders=set(),
        pheromones=pheromone_dict,
        temperature_schedule=temps,
        independence_lambda=2.0,
        pheromone_increment=0.05,
    )

    print("Final leader set:", final_leaders)
    print("Final pheromone values:")
    for n, entry in sorted(pheromone_dict.items()):
        print(f"  node {n}: {entry.signal_value:.4f}")
    # Verify that no two leaders are adjacent (independent set)
    for u in final_leaders:
        for v in G[u]:
            assert v not in final_leaders, f"Adjacency violation: {u}-{v}"
    print("Independent set verified.")