# DARWIN HAMMER — match 592, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_distributed_l_chelydrid_ambush_m42_s2.py (gen2)
# parent_b: hybrid_bayes_claim_kernel_hybrid_ternary_lens__m26_s1.py (gen2)
# born: 2026-05-29T23:29:52Z

"""Hybrid Algorithm: Distributed Leader Election + Bayesian Kinetic Scoring

Parents:
- hybrid_hybrid_distributed_l_chelydrid_ambush_m42_s2 (perceptual‑hash graph + kinetic
  score biased broadcast)
- hybrid_bayes_claim_kernel_hybrid_ternary_lens (Bayesian posterior update with
  decreasing pruning probability)

Mathematical Bridge:
Each element (vector of floats) is interpreted simultaneously as
  1) a perceptual signature → 64‑bit hash defining a similarity graph G.
  2) a time‑series of forces → integrated into a kinetic score K (travelled
     distance).

A node i is a hypothesis H_i : “i is the leader of its perceptual cluster”.
Its prior is uniform (0.5).  The likelihood ratio L_i is taken as the
normalized kinetic score multiplied by the pruning probability P(t) from the
decreasing schedule.  Bayesian updating yields a posterior p_i(t).  Nodes are
broadcast with probability proportional to p_i(t); a greedy maximal‑independent‑
set on G, ordered by posterior, produces the final elected leaders.

The implementation below fuses the core equations of both parents:
  * perceptual hashing and Hamming‑distance graph (parent A)
  * integrate‑strike dynamics, prune‑probability, and Bayesian posterior update
    (parent B)

The three public functions demonstrate the hybrid pipeline:
  * `build_perceptual_graph`
  * `compute_kinetic_scores`
  * `elect_leaders_via_bayesian_graph`
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Mapping, Hashable, Tuple

import numpy as np

# ---------- Parent A utilities (perceptual hash & graph) ----------

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
    """Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()


def build_perceptual_graph(elements: List[List[float]], max_hamming: int = 4) -> Dict[str, Set[str]]:
    """
    Build an undirected similarity graph where two nodes are connected
    if their perceptual hashes differ by at most `max_hamming` bits.
    Returns a mapping node_id → set(neighbor_ids).
    """
    hashes: Dict[str, int] = {str(i): compute_phash(el) for i, el in enumerate(elements)}
    graph: Dict[str, Set[str]] = {str(i): set() for i in range(len(elements))}
    ids = list(graph.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if hamming_distance(hashes[ids[i]], hashes[ids[j]]) <= max_hamming:
                graph[ids[i]].add(ids[j])
                graph[ids[j]].add(ids[i])
    return graph


# ---------- Parent B utilities (kinetic integration & Bayesian update) ----------

def integrate_strike(forces: List[float], dt: float = 1.0) -> Tuple[float, float]:
    """
    Simple physics integration assuming unit mass.
    Returns (final_velocity, travelled_distance).
    """
    velocity = 0.0
    distance = 0.0
    for f in forces:
        acceleration = f          # F = m*a, m=1
        velocity += acceleration * dt
        distance += velocity * dt
    return velocity, distance


def compute_kinetic_scores(elements: List[List[float]]) -> List[float]:
    """
    For each element treat the vector as a force time‑series and return the
    travelled distance (kinetic score).  The raw scores are returned; callers
    may normalize them.
    """
    scores = []
    for vec in elements:
        _, dist = integrate_strike(vec)
        scores.append(dist)
    return scores


def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Decreasing pruning schedule P(t) = min(1, λ·exp(-α·t))."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))


def bayesian_update(p_prior: float, likelihood_ratio: float) -> float:
    """
    Classic Bayesian odds update.
    p_prior ∈ [0,1], likelihood_ratio ≥ 0.
    Returns posterior probability clipped to [0,1].
    """
    if likelihood_ratio < 0:
        raise ValueError("likelihood_ratio must be non‑negative")
    p = max(0.0, min(1.0, p_prior))
    if p <= 0.0 or likelihood_ratio == 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    odds = p / (1.0 - p)
    new_odds = odds * likelihood_ratio
    posterior = new_odds / (1.0 + new_odds)
    return max(0.0, min(1.0, posterior))


# ---------- Hybrid data structures ----------

@dataclass
class NodeData:
    """Container for a single element in the hybrid system."""
    id: str
    vector: List[float]
    kinetic_score: float = 0.0
    posterior: float = 0.5   # uniform prior
    neighbors: Set[str] = field(default_factory=set)


# ---------- Core hybrid functions ----------

def initialise_nodes(elements: List[List[float]]) -> Dict[str, NodeData]:
    """
    Create NodeData objects, compute kinetic scores, and attach graph neighbours.
    """
    # kinetic scores
    raw_scores = compute_kinetic_scores(elements)
    max_score = max(raw_scores) if raw_scores else 1.0
    norm_scores = [s / max_score for s in raw_scores]

    # graph
    graph = build_perceptual_graph(elements)

    nodes: Dict[str, NodeData] = {}
    for idx, vec in enumerate(elements):
        nid = str(idx)
        node = NodeData(
            id=nid,
            vector=vec,
            kinetic_score=norm_scores[idx],
            posterior=0.5,
            neighbors=graph[nid].copy(),
        )
        nodes[nid] = node
    return nodes


def run_bayesian_iterations(nodes: Dict[str, NodeData], iterations: int = 5) -> None:
    """
    Perform a fixed number of Bayesian update rounds.
    At each round t we compute a likelihood ratio for each node:
        L_i(t) = kinetic_score_i * prune_probability(t)
    Posterior is updated in‑place.
    """
    for t in range(iterations):
        p_t = prune_probability(t)
        for node in nodes.values():
            lr = node.kinetic_score * p_t + 1e-9   # avoid zero
            node.posterior = bayesian_update(node.posterior, lr)


def elect_leaders_via_bayesian_graph(nodes: Dict[str, NodeData]) -> Set[str]:
    """
    Greedy maximal‑independent‑set election.
    Nodes are processed in descending order of posterior probability.
    A node is selected as a leader if none of its already‑selected neighbours
    have been chosen.
    Returns the set of elected leader ids.
    """
    # sort ids by posterior descending
    sorted_ids = sorted(nodes.keys(), key=lambda i: nodes[i].posterior, reverse=True)
    leaders: Set[str] = set()
    excluded: Set[str] = set()

    for nid in sorted_ids:
        if nid in excluded:
            continue
        leaders.add(nid)
        excluded.update(nodes[nid].neighbors)
        excluded.add(nid)  # also exclude self for completeness
    return leaders


def hybrid_pipeline(elements: List[List[float]], iterations: int = 5) -> Set[str]:
    """
    End‑to‑end hybrid procedure:
      1. Initialise nodes (hash graph + kinetic scores)
      2. Run Bayesian posterior updates with decreasing pruning
      3. Elect leaders using a posterior‑biased maximal independent set
    Returns the set of elected leader identifiers.
    """
    nodes = initialise_nodes(elements)
    run_bayesian_iterations(nodes, iterations=iterations)
    leaders = elect_leaders_via_bayesian_graph(nodes)
    return leaders


# ---------- Smoke test ----------

if __name__ == "__main__":
    # Generate a synthetic dataset: 30 elements, each a random 20‑dim float vector
    random.seed(42)
    np.random.seed(42)
    dataset = [np.random.uniform(-1, 1, size=20).tolist() for _ in range(30)]

    elected = hybrid_pipeline(dataset, iterations=7)

    print(f"Total elements: {len(dataset)}")
    print(f"Elected leaders ({len(elected)}): {sorted(elected)}")
    # Verify that no two elected leaders are neighbours in the perceptual graph
    graph = build_perceptual_graph(dataset)
    for lid in elected:
        for other in elected:
            if lid != other and other in graph[lid]:
                print("Error: neighboring leaders detected!", file=sys.stderr)
                sys.exit(1)
    print("Smoke test passed: elected leaders form an independent set.")