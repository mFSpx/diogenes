# DARWIN HAMMER — match 65, survivor 2
# gen: 1
# parent_a: distributed_leader_election.py (gen0)
# parent_b: thanatosis.py (gen0)
# born: 2026-05-29T23:24:13Z

"""Hybrid Leader Election via Simulated Annealing.

This module fuses **distributed_leader_election.py** (Algorithm A) and
**thanatosis.py** (Algorithm B) by recognizing that both core components
are exponential decay functions:

* Algorithm A uses a broadcast probability  
  p = 1 / 2^{phase‑step}.
* Algorithm B uses a cooling temperature  
  T = t₀ · α^{k}.

Both can be written as p = c·e^{‑λ·x} with appropriate constants.
The hybrid algorithm treats the broadcast probability as a temperature
and applies the Metropolis acceptance rule to decide whether a tentative
broadcast (candidate leader) should be kept, thus embedding simulated‑
annealing into the maximal‑independent‑set construction.

The bridge is therefore the exponential‑decay mapping:

    T_phase = t0 * (alpha ** (phase-1))
    p_phase = min(1.0, 1.0 / (2 ** max(0, phases - phase)))
    temperature = T_phase * p_phase   # combined decay

The acceptance probability for a candidate node n is
    exp( -ΔE_n / temperature )
where ΔE_n is the number of conflicts (edges to already selected
broadcasts).  This yields a unified stochastic process that respects
both the locality of Algorithm A and the annealing dynamics of
Algorithm B.
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Mapping, Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Set, Tuple

import numpy as np

Node = Hashable
Graph = Mapping[Node, Set[Node]]


def broadcast_probability(phases: int, phase: int) -> float:
    """Original A: p = 1 / 2^{phases‑phase}, clamped to [0,1]."""
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Original B: exponential cooling schedule."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


def hybrid_temperature(phases: int, phase: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """
    Combine the decay of broadcast probability and annealing temperature.

    temperature = cooling_temperature(phase‑1) * broadcast_probability(...)
    """
    p = broadcast_probability(phases, phase)
    temp = cooling_temperature(phase - 1, t0, alpha)
    return temp * p


def acceptance_probability(delta_e: int, temperature: float) -> float:
    """Metropolis rule (Algorithm B) for integer energy delta."""
    if delta_e < 0:
        return 1.0
    if temperature <= 0.0:
        return 0.0
    return math.exp(-delta_e / temperature)


def conflict_delta(node: Node, current_broadcasts: Set[Node], graph: Graph) -> int:
    """
    Energy delta for adding `node` to the broadcast set:
    number of neighboring broadcasts that would cause a conflict.
    """
    neighbors = graph.get(node, set())
    return len(neighbors & current_broadcasts)


def hybrid_maximal_independent_set(
    graph: Graph,
    phases: int = 8,
    t0: float = 1.0,
    alpha: float = 0.95,
    seed: int | str | None = None,
) -> Set[Node]:
    """
    Hybrid MIS construction.

    For each phase a candidate broadcast set is sampled uniformly.
    Each candidate is accepted with Metropolis probability using the
    hybrid temperature.  Accepted nodes become leaders; their closed
    neighbourhood is removed from further consideration.
    """
    rng = random.Random(seed)
    undecided: Set[Node] = set(graph)
    leaders: Set[Node] = set()
    blocked: Set[Node] = set()

    for phase in range(1, phases + 1):
        if not undecided:
            break

        # Sample raw broadcast candidates according to original A's p
        raw_p = broadcast_probability(phases, phase)
        raw_candidates = {n for n in undecided if rng.random() < raw_p}

        # Apply simulated‑annealing acceptance
        temperature = hybrid_temperature(phases, phase, t0, alpha)
        accepted: Set[Node] = set()
        for n in raw_candidates:
            delta = conflict_delta(n, accepted, graph)
            p_accept = acceptance_probability(delta, temperature)
            if rng.random() <= p_accept:
                accepted.add(n)

        # Resolve remaining conflicts deterministically (as in A)
        new_leaders = {
            n
            for n in accepted
            if not (graph.get(n, set()) & accepted)  # no neighbor also accepted
        }

        leaders.update(new_leaders)

        # Block the closed neighbourhood of new leaders
        newly_blocked = set().union(
            *(graph.get(n, set()) for n in new_leaders), new_leaders
        )
        blocked.update(newly_blocked)
        undecided -= blocked

    # Final sweep to guarantee maximality (identical to A)
    for n in sorted(undecided, key=str):
        if not (graph.get(n, set()) & leaders):
            leaders.add(n)

    return leaders


def hybrid_decision(
    node: Node,
    current_broadcasts: Set[Node],
    graph: Graph,
    phase: int,
    phases: int,
    t0: float = 1.0,
    alpha: float = 0.95,
    seed: int | str | None = None,
) -> Tuple[bool, float]:
    """
    Return (accept, probability) for a single node decision.
    Demonstrates the core hybrid acceptance computation.
    """
    rng = random.Random(seed)
    temperature = hybrid_temperature(phases, phase, t0, alpha)
    delta = conflict_delta(node, current_broadcasts, graph)
    prob = acceptance_probability(delta, temperature)
    accept = rng.random() <= prob
    return accept, prob


def hybrid_simulation(
    graph: Graph,
    phases: int = 8,
    trials: int = 3,
    seed: int | str | None = None,
) -> Set[Node]:
    """
    Run multiple independent hybrid MIS constructions and return the
    union of all leaders discovered.  This showcases stochastic
    variability across runs.
    """
    rng = random.Random(seed)
    aggregate: Set[Node] = set()
    for i in range(trials):
        run_seed = rng.randint(0, 2 ** 32 - 1)
        leaders = hybrid_maximal_independent_set(
            graph, phases=phases, seed=run_seed
        )
        aggregate.update(leaders)
    return aggregate


if __name__ == "__main__":
    # Simple smoke test on a small graph.
    sample_graph: Graph = {
        "A": {"B", "C"},
        "B": {"A", "D"},
        "C": {"A", "D"},
        "D": {"B", "C", "E"},
        "E": {"D"},
    }

    print("Hybrid MIS:", hybrid_maximal_independent_set(sample_graph, phases=6, seed=42))
    node = "A"
    accept, prob = hybrid_decision(
        node,
        current_broadcasts=set(),
        graph=sample_graph,
        phase=2,
        phases=6,
        seed=123,
    )
    print(f"Decision for node {node!r}: accept={accept}, prob={prob:.4f}")

    union = hybrid_simulation(sample_graph, phases=6, trials=5, seed=7)
    print("Union of leaders over 5 trials:", union)