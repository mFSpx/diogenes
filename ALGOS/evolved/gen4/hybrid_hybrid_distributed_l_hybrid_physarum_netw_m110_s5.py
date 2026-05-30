# DARWIN HAMMER — match 110, survivor 5
# gen: 4
# parent_a: hybrid_distributed_leader_e_thanatosis_m65_s2.py (gen1)
# parent_b: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s5.py (gen3)
# born: 2026-05-29T23:26:52Z

"""Hybrid Leader‑Physarum Algorithm
=================================
Parent A: ``hybrid_distributed_leader_e_thanatosis_m65_s2.py`` – a
simulated‑annealing leader election that treats the broadcast probability
as a temperature.

Parent B: ``hybrid_physarum_network_hybrid_hybrid_bandit_m11_s5.py`` – a
Physarum‑inspired flow network where edge conductances evolve according
to the absolute flux.

**Mathematical bridge**

Both parents rely on exponential‑decay schedules:

* A’s broadcast probability  p = 2^{-(phases‑phase)}  ⇔  p = e^{‑λ·phase}
* B’s cooling temperature   T = t₀·α^{phase‑1}   ⇔  T = e^{‑μ·phase}

We fuse them by defining a *joint temperature*


temp = cooling_temperature(phase‑1) * broadcast_probability(phases, phase)


The temperature scales the Physarum conductance update (higher temperature
→ faster adaptation) and is also used as the Metropolis temperature for the
leader‑selection acceptance probability.  Thus a single scalar drives both
the stochastic election and the deterministic flux dynamics, providing a
coherent hybrid system.

The module supplies three core functions that demonstrate the combined
behaviour:

hybrid_temperature(...)
physarum_step(...)
leader_election_step(...)

"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, Set, Tuple, Hashable, Mapping, Any

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
Edge = Tuple[Node, Node]

# ----------------------------------------------------------------------
# Parent A – probability & cooling utilities
# ----------------------------------------------------------------------
def broadcast_probability(phases: int, phase: int) -> float:
    """Original A: p = 1 / 2^{phases‑phase}, clamped to [0,1]."""
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Original B: exponential cooling schedule."""
    if k < 0 or t0 < 0 or not (0.0 <= alpha <= 1.0):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


def hybrid_temperature(
    phases: int,
    phase: int,
    t0: float = 1.0,
    alpha: float = 0.95,
) -> float:
    """
    Joint temperature = cooling schedule * broadcast probability.
    """
    p = broadcast_probability(phases, phase)
    T = cooling_temperature(phase - 1, t0, alpha)
    return T * p


# ----------------------------------------------------------------------
# Parent B – Physarum flux primitives
# ----------------------------------------------------------------------
def flux(
    conductance: float,
    edge_length: float,
    pressure_a: float,
    pressure_b: float,
    eps: float = 1e-12,
) -> float:
    """Physarum flux on a single edge."""
    if edge_length <= 0:
        raise ValueError("edge_length must be positive")
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(
    conductance: float,
    q: float,
    dt: float,
    gain: float,
    decay: float,
    temperature: float,
) -> float:
    """
    Conductance evolution with temperature‑scaled gain.
    𝒞 ← max(0, 𝒞 + dt·(gain·temp·|q| – decay·𝒞))
    """
    delta = dt * (gain * temperature * abs(q) - decay * conductance)
    new_c = max(0.0, conductance + delta)
    return new_c


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def physarum_step(
    graph: Graph,
    conductances: Dict[Edge, float],
    edge_lengths: Dict[Edge, float],
    pressures: Dict[Node, float],
    dt: float = 0.1,
    gain: float = 1.0,
    decay: float = 0.01,
    temperature: float = 1.0,
) -> Dict[Edge, float]:
    """
    Perform one Physarum update over all edges.

    Returns a new conductance dictionary.
    """
    new_conductances: Dict[Edge, float] = {}
    for u, neigh in graph.items():
        for v in neigh:
            if u == v:
                continue
            e: Edge = tuple(sorted((u, v)))
            if e in new_conductances:  # edge already processed from opposite side
                continue
            c = conductances.get(e, 1.0)
            L = edge_lengths.get(e, 1.0)
            q = flux(c, L, pressures[u], pressures[v])
            new_c = update_conductance(c, q, dt, gain, decay, temperature)
            new_conductances[e] = new_c
    return new_conductances


def leader_election_step(
    graph: Graph,
    phases: int,
    phase: int,
    conductances: Dict[Edge, float],
    edge_lengths: Dict[Edge, float],
    t0: float = 1.0,
    alpha: float = 0.95,
) -> Set[Node]:
    """
    One stochastic leader‑election round.

    * Candidates are drawn with probability p = broadcast_probability().
    * For each candidate we compute ΔE = number of already‑selected neighbours.
    * Acceptance follows Metropolis:  exp(-ΔE / temperature).
    * The temperature is the hybrid temperature defined above.
    * After the selection, a Physarum step updates conductances using the
      same temperature, closing the feedback loop.
    """
    temperature = hybrid_temperature(phases, phase, t0, alpha)
    p = broadcast_probability(phases, phase)

    # 1) draw tentative candidates
    tentative: Set[Node] = {
        n for n in graph.keys() if random.random() < p
    }

    # 2) Metropolis acceptance
    selected: Set[Node] = set()
    for n in tentative:
        conflicts = sum(1 for nb in graph[n] if nb in selected)
        if conflicts == 0:
            selected.add(n)
        else:
            accept_prob = math.exp(-conflicts / max(temperature, 1e-12))
            if random.random() < accept_prob:
                selected.add(n)

    # 3) Build node pressures from selection (selected → high pressure)
    #    Unselected nodes get a baseline pressure of 0.
    pressures: Dict[Node, float] = {
        n: (1.0 if n in selected else 0.0) for n in graph
    }

    # 4) Physarum conductance adaptation
    new_conductances = physarum_step(
        graph,
        conductances,
        edge_lengths,
        pressures,
        dt=0.1,
        gain=1.0,
        decay=0.01,
        temperature=temperature,
    )
    conductances.clear()
    conductances.update(new_conductances)

    return selected


def initialize_hybrid(
    graph: Graph,
    init_conductance: float = 1.0,
    init_edge_length: float = 1.0,
) -> Tuple[Dict[Edge, float], Dict[Edge, float]]:
    """
    Initialise conductance and edge‑length dictionaries for a given graph.
    Edge keys are stored as sorted (u, v) tuples.
    """
    conductances: Dict[Edge, float] = {}
    edge_lengths: Dict[Edge, float] = {}
    for u, neigh in graph.items():
        for v in neigh:
            if u == v:
                continue
            e: Edge = tuple(sorted((u, v)))
            if e in conductances:
                continue
            conductances[e] = init_conductance
            edge_lengths[e] = init_edge_length
    return conductances, edge_lengths


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple undirected graph (5 nodes, random edges)
    random.seed(42)
    nodes = list(range(5))
    graph: Dict[int, Set[int]] = {n: set() for n in nodes}
    # create a random connected graph
    for i in range(4):
        u = i
        v = i + 1
        graph[u].add(v)
        graph[v].add(u)
    # add a few extra random edges
    extra_edges = [(0, 2), (1, 3), (2, 4)]
    for u, v in extra_edges:
        graph[u].add(v)
        graph[v].add(u)

    conductances, edge_lengths = initialize_hybrid(graph)

    phases = 6
    all_selected: Set[int] = set()
    print("Hybrid Leader‑Physarum simulation")
    for phase in range(1, phases + 1):
        selected = leader_election_step(
            graph,
            phases=phases,
            phase=phase,
            conductances=conductances,
            edge_lengths=edge_lengths,
            t0=1.0,
            alpha=0.9,
        )
        all_selected.update(selected)
        temp = hybrid_temperature(phases, phase)
        print(
            f"Phase {phase:2d} | Temp={temp:.4f} | Selected={sorted(selected)}"
        )
    print("\nUnion of all selected leaders:", sorted(all_selected))
    sys.exit(0)