# DARWIN HAMMER — match 42, survivor 2
# gen: 2
# parent_a: hybrid_distributed_leader_e_perceptual_dedupe_m16_s0.py (gen1)
# parent_b: chelydrid_ambush.py (gen0)
# born: 2026-05-29T23:23:34Z

"""Hybrid algorithm merging distributed leader election (parent A) with
burst‑strike kinematics (parent B).

Mathematical bridge:
- Each element is a vector of floats.  From parent B we can interpret this
  vector as a time‑series of forces and integrate it with the
  `integrate_strike` dynamics, yielding a scalar “kinetic score”
  (peak velocity or travelled distance).
- From parent A we build a perceptual‑hash similarity graph and run a
  maximal‑independent‑set leader election.
- The bridge consists in using the kinetic score to bias the broadcast
  probability of each node during the election.  Nodes with higher
  kinetic score are more likely to broadcast, so the elected leaders tend
  to be the most “energetic” representatives of each perceptual cluster.

The result is a hybrid clustering where each cluster is defined by
perceptual similarity and its representative is chosen by a physics‑
driven leader election.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
from collections.abc import Mapping, Hashable, Iterable
from dataclasses import dataclass
from typing import List, Dict, Set

import numpy as np

Node = Hashable
Graph = Mapping[Node, Set[Node]]

# ---------- Parent A utilities ----------

def compute_phash(values: List[float]) -> int:
    """64‑bit perceptual hash based on average comparison."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def build_graph(elements: List[List[float]]) -> Graph:
    """Connect two nodes if their perceptual hashes differ by ≤4 bits."""
    hashes: Dict[str, int] = {str(i): compute_phash(el) for i, el in enumerate(elements)}
    graph: Dict[str, Set[str]] = {str(i): set() for i in range(len(elements))}
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            if hamming_distance(hashes[str(i)], hashes[str(j)]) <= 4:
                graph[str(i)].add(str(j))
                graph[str(j)].add(str(i))
    return graph

# ---------- Parent B utilities ----------

@dataclass(frozen=True)
class StrikeState:
    velocity: float
    distance: float
    peak_velocity: float

def integrate_strike(
    force_series: Iterable[float],
    dt: float,
    m_head: float,
    drag_cd: float = 0.3,
    fluid_density: float = 1000.0,
    area: float = 0.001,
    v0: float = 0.0,
) -> StrikeState:
    """Integrate a 1‑D strike under quadratic drag."""
    if dt <= 0 or m_head <= 0 or drag_cd < 0 or fluid_density < 0 or area < 0:
        raise ValueError("invalid strike parameters")
    v = v0
    x = 0.0
    peak = v0
    for force in force_series:
        drag = drag_cd * fluid_density * area * v * abs(v) / (2.0 * m_head)
        acc = force / m_head - drag
        v = max(0.0, v + acc * dt)
        x += v * dt
        peak = max(peak, v)
    return StrikeState(v, x, peak)

def kinetic_score_from_element(element: List[float]) -> float:
    """
    Treat the element as a force series, integrate it, and return a
    dimension‑less score (distance travelled normalized).
    """
    state = integrate_strike(
        force_series=element,
        dt=0.01,
        m_head=1.0,
        drag_cd=0.2,
        fluid_density=1.0,
        area=1.0,
        v0=0.0,
    )
    # Normalise by number of steps to keep scores comparable across sizes
    return state.distance / max(1.0, len(element) * 0.01)

# ---------- Hybrid core ----------

def hybrid_broadcast_probability(phase: int, total_phases: int, score: float) -> float:
    """
    Base probability from parent A (1/2^(total_phases‑phase)) scaled by a
    sigmoid of the kinetic score.  High‑energy nodes broadcast more often.
    """
    if phase < 1 or total_phases < 1:
        raise ValueError("phase and total_phases must be positive")
    base = min(1.0, 1.0 / (2 ** max(0, total_phases - phase)))
    # sigmoid maps score ∈ ℝ to (0,1)
    weight = 1.0 / (1.0 + math.exp(-score))
    return min(1.0, base * (0.5 + weight / 2.0))  # keep within [0,1]

def hybrid_maximal_independent_set(
    graph: Graph,
    scores: Dict[Node, float],
    phases: int = 8,
    seed: int | str | None = None,
) -> Set[Node]:
    """
    Approximate a maximal independent set where broadcast probabilities
    are biased by kinetic scores.
    """
    rng = random.Random(seed)
    undecided: Set[Node] = set(graph)
    leaders: Set[Node] = set()
    blocked: Set[Node] = set()

    for phase in range(1, phases + 1):
        if not undecided:
            break
        # compute per‑node broadcast decision
        broadcasts = {
            n
            for n in undecided
            if rng.random()
            < hybrid_broadcast_probability(phase, phases, scores.get(n, 0.0))
        }
        # a broadcast is a leader if none of its neighbours also broadcast
        new_leaders = {n for n in broadcasts if not (graph.get(n, set()) & broadcasts)}
        leaders.update(new_leaders)

        # block leaders and all their neighbours
        newly_blocked = set(new_leaders)
        for n in new_leaders:
            newly_blocked.update(graph.get(n, set()))
        blocked.update(newly_blocked)
        undecided -= blocked

    # deterministic cleanup: any remaining undecided node that does not see a leader becomes a leader
    for n in sorted(undecided, key=str):
        if not (graph.get(n, set()) & leaders):
            leaders.add(n)
    return leaders

def hybrid_cluster_elements(elements: List[List[float]]) -> List[List[List[float]]]:
    """
    Build a perceptual similarity graph, compute kinetic scores for each
    element, run the hybrid leader election, and return clusters where each
    cluster is the set of elements reachable from its leader (including the
    leader itself).
    """
    graph = build_graph(elements)

    # compute kinetic scores once
    scores: Dict[Node, float] = {str(i): kinetic_score_from_element(el) for i, el in enumerate(elements)}

    leaders = hybrid_maximal_independent_set(graph, scores)

    # initialise clusters
    clusters: Dict[Node, List[List[float]]] = {leader: [] for leader in leaders}

    # assign each element to the first leader it is adjacent to (or to itself if isolated)
    for i, element in enumerate(elements):
        node = str(i)
        assigned = False
        for leader in leaders:
            if node == leader or node in graph.get(leader, set()):
                clusters[leader].append(element)
                assigned = True
                break
        if not assigned:
            # isolated node becomes its own cluster
            clusters[node] = [element]

    return list(clusters.values())

# ---------- Smoke test ----------

if __name__ == "__main__":
    # generate 120 random 64‑dimensional elements
    random.seed(42)
    elements = [[random.random() for _ in range(64)] for _ in range(120)]

    clusters = hybrid_cluster_elements(elements)

    print(f"Generated {len(elements)} elements.")
    print(f"Hybrid clustering produced {len(clusters)} clusters.")
    # show size distribution
    sizes = sorted([len(c) for c in clusters], reverse=True)
    print("Cluster sizes (top 10):", sizes[:10])
    sys.exit(0)