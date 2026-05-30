# DARWIN HAMMER — match 5440, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_regret_engine_hybrid_hybrid_endpoi_m1975_s0.py (gen3)
# parent_b: hybrid_hybrid_distributed_l_hybrid_hybrid_ternar_m2537_s2.py (gen5)
# born: 2026-05-30T00:02:04Z

"""Hybrid Regret‑Similarity Engine
================================

This module fuses the two parent algorithms:

* **Parent A – hybrid_hybrid_regret_engine_hybrid_hybrid_endpoi_m1975_s0.py**
  provides decision elements (`MathAction`), counter‑factual outcomes and a
  regret‑weighted probability model.

* **Parent B – hybrid_hybrid_distributed_l_hybrid_hybrid_ternar_m2537_s2.py**
  supplies perceptual hashing, structural similarity (SSIM), clustering and a
  distributed leader‑election routine that operates on a similarity matrix
  of graph nodes.

**Mathematical bridge**

For every decision element we build a low‑dimensional feature vector
`[expected_value, cost, risk]`.  The SSIM‑based similarity matrix of these
vectors (Parent B) is then *modulated* by a diagonal matrix of regret‑weights
computed from the counter‑factuals (Parent A).  The hybrid similarity is


S_hybrid = W · S_base · W


where `S_base` is the SSIM similarity matrix and `W = diag(w_i)` contains the
regret‑weights `w_i ≥ 1`.  The resulting matrix is fed to the distributed
leader‑election algorithm, yielding leaders that are both *structurally*
similar and *regret‑aware*.

The code below implements this fusion and provides three public functions that
demonstrate the hybrid operation.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Tuple
import datetime as dt
import numpy as np

# ----------------------------------------------------------------------
# Parent A – decision / regret primitives
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MathAction:
    """Elementary decision element."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True, slots=True)
class MathCounterfactual:
    """Alternative outcome information for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


def weekday_index(year: int, month: int, day: int) -> int:
    """ISO weekday index (0 = Monday … 6 = Sunday)."""
    return dt.date(year, month, day).weekday()


def gini_coefficient(values: List[float]) -> float:
    """Gini coefficient for a non‑negative distribution."""
    xs = sorted(float(v) for v in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0.0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cumulative) / (n * sum(xs)) - (n + 1) / n


def regret_weight(action: MathAction,
                  counterfactuals: List[MathCounterfactual]) -> float:
    """
    Compute a regret‑derived multiplicative weight for an action.

    The raw regret is the expected shortfall compared with each counter‑factual:

        r = Σ max(0, cf.outcome_value - action.expected_value) * cf.probability

    The final weight is `1 + r` (guaranteed ≥ 1) so that it can be used as a
    scaling factor.
    """
    r = 0.0
    for cf in counterfactuals:
        if cf.action_id != action.id:
            continue
        shortfall = cf.outcome_value - action.expected_value
        if shortfall > 0:
            r += shortfall * cf.probability
    return 1.0 + r


# ----------------------------------------------------------------------
# Parent B – perceptual hashing, SSIM, clustering, leader election
# ----------------------------------------------------------------------
Node = int
Graph = Dict[Node, Set[Node]]
FeatureVec = List[float]


def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash based on the mean of the first 64 values."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


def ssim(x: np.ndarray,
         y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """
    Structural Similarity Index (SSIM) between two 1‑D signals.
    The implementation follows the classic formulation but works on vectors.
    """
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    l = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    c = (2 * sigma_xy + C2) / (sigma_x ** 2 + sigma_y ** 2 + C2)
    return l * c


def cluster_by_phash(hashes: Dict[Node, int],
                     max_distance: int = 4) -> List[List[Node]]:
    """Cluster nodes whose perceptual hashes are within `max_distance`."""
    clusters: List[List[Node]] = []
    for node, h in hashes.items():
        for c in clusters:
            if hamming_distance(h, c[0]) <= max_distance:
                c.append(node)
                break
        else:
            clusters.append([node])
    return clusters


def get_similarity_matrix(nodes: List[Node],
                          feature_vectors: List[FeatureVec]) -> np.ndarray:
    """
    Build an SSIM similarity matrix for the supplied feature vectors.
    Diagonal entries are forced to 1.0.
    """
    hashes = {n: compute_phash(vs) for n, vs in zip(nodes, feature_vectors)}
    # clustering is not required for the matrix itself but is kept for
    # potential downstream use.
    _ = cluster_by_phash(hashes)

    n = len(nodes)
    sim = np.zeros((n, n), dtype=float)
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i == j:
                continue
            sim[i, j] = ssim(np.array(feature_vectors[ni]),
                             np.array(feature_vectors[nj]))
    np.fill_diagonal(sim, 1.0)
    return sim


def distributed_leader_election(graph: Graph,
                                similarity_matrix: np.ndarray,
                                threshold: float = 0.5) -> List[Node]:
    """
    Simple leader election:
    - A node becomes a leader if its average similarity to undecided neighbors
      exceeds `threshold` or it has no neighbors.
    - All its neighbors are then removed from the undecided pool (they become
      followers).
    The process repeats until no undecided nodes remain.
    """
    leaders: Set[Node] = set()
    undecided: Set[Node] = set(graph.keys())
    node_index = {node: idx for idx, node in enumerate(sorted(graph.keys()))}

    while undecided:
        elected_this_round: Set[Node] = set()
        for node in list(undecided):
            neighbors = graph[node] & undecided
            if not neighbors:
                leaders.add(node)
                elected_this_round.add(node)
                continue
            idx = node_index[node]
            neighbor_idxs = [node_index[n] for n in neighbors]
            avg_sim = float(np.mean(similarity_matrix[idx, neighbor_idxs]))
            if avg_sim >= threshold:
                leaders.add(node)
                elected_this_round.add(node)
        # Remove elected leaders and their immediate neighbors
        to_remove = set(elected_this_round)
        for leader in elected_this_round:
            to_remove.update(graph[leader])
        undecided -= to_remove
        # Guard against deadlock (no node satisfied the threshold)
        if not elected_this_round:
            # pick a random remaining node as leader
            fallback = next(iter(undecided))
            leaders.add(fallback)
            undecided -= {fallback} | graph[fallback]

    return sorted(leaders)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def build_feature_vectors(actions: List[MathAction]) -> List[FeatureVec]:
    """
    Convert a list of `MathAction` objects into numeric feature vectors.
    The vector layout matches the regret‑weighting bridge:
        [expected_value, cost, risk]
    """
    return [[a.expected_value, a.cost, a.risk] for a in actions]


def hybrid_similarity_matrix(actions: List[MathAction],
                             counterfactuals: List[MathCounterfactual]) -> np.ndarray:
    """
    Compute the regret‑modulated similarity matrix.

    Steps
    -----
    1. Build raw feature vectors and the base SSIM matrix `S_base`.
    2. Compute a regret weight `w_i` for each action.
    3. Form the diagonal matrix `W = diag(w_i)`.
    4. Return `S_hybrid = W @ S_base @ W`.
    """
    # 1. Base similarity
    feature_vectors = build_feature_vectors(actions)
    nodes = list(range(len(actions)))          # node ids correspond to indices
    S_base = get_similarity_matrix(nodes, feature_vectors)

    # 2. Regret weights
    weights = np.array([regret_weight(a, counterfactuals) for a in actions],
                       dtype=float)

    # 3. Diagonal scaling matrix
    W = np.diag(weights)

    # 4. Hybrid similarity
    S_hybrid = W @ S_base @ W
    # Normalise to keep values in a comparable range (optional)
    max_val = S_hybrid.max()
    if max_val > 0:
        S_hybrid /= max_val
    return S_hybrid


def hybrid_leader_election(actions: List[MathAction],
                           counterfactuals: List[MathCounterfactual],
                           graph: Graph) -> List[Node]:
    """
    Perform leader election on the action graph using the regret‑aware
    similarity matrix.
    """
    S = hybrid_similarity_matrix(actions, counterfactuals)
    return distributed_leader_election(graph, S)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny decision set
    actions = [
        MathAction(id="A", expected_value=10.0, cost=2.0, risk=0.1),
        MathAction(id="B", expected_value=12.0, cost=1.5, risk=0.2),
        MathAction(id="C", expected_value=8.0,  cost=3.0, risk=0.05),
        MathAction(id="D", expected_value=15.0, cost=2.5, risk=0.3),
    ]

    # Counter‑factual outcomes (some of them intentionally worse)
    counterfactuals = [
        MathCounterfactual(action_id="A", outcome_value=11.0, probability=0.6),
        MathCounterfactual(action_id="A", outcome_value=9.0,  probability=0.4),
        MathCounterfactual(action_id="B", outcome_value=13.5, probability=0.5),
        MathCounterfactual(action_id="C", outcome_value=9.5,  probability=0.7),
        MathCounterfactual(action_id="D", outcome_value=14.0, probability=0.3),
    ]

    # Build a fully‑connected undirected graph (excluding self‑loops)
    graph: Graph = {i: set(j for j in range(len(actions)) if j != i)
                    for i in range(len(actions))}

    leaders = hybrid_leader_election(actions, counterfactuals, graph)
    print("Hybrid leaders (by index):", leaders)
    print("Corresponding action IDs:",
          [actions[i].id for i in leaders])