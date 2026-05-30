# DARWIN HAMMER — match 5496, survivor 1
# gen: 7
# parent_a: hybrid_distributed_leader_e_perceptual_dedupe_m16_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_ssim_m1265_s1.py (gen6)
# born: 2026-05-30T00:02:25Z

"""Hybrid algorithm combining distributed leader election (graph MIS) with perceptual hashing
and SSIM‑based similarity modulation.

Parents:
- hybrid_distributed_leader_e_perceptual_dedupe_m16_s1.py (graph MIS + perceptual hashing)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ssim_m1265_s1.py (SSIM, morphology bandit, StoreState)

Mathematical bridge:
Node feature vectors are first reduced to perceptual hashes (binary integers).  
Pairwise SSIM between a node's feature vector and the global feature centroid yields a
similarity score s∈[0,1].  This score modulates the broadcast probability p used in the
maximal independent set (MIS) algorithm:
    p' = p * (1‑s) + ε,
where ε guarantees a non‑zero floor.  Consequently, nodes whose attributes are
representative of the whole graph (high similarity) broadcast less aggressively,
while outliers broadcast more, steering leader election toward diverse clusters.
StoreState dynamics are driven by the number of newly elected leaders (inflow) and
the number of nodes blocked in the current phase (outflow), providing a feedback
loop that adapts the “dance” parameter for future phases.

The resulting hybrid provides similarity‑aware leader election and a stateful
adaptive mechanism suitable for clustering heterogeneous graph data.
"""

import sys
import math
import random
from pathlib import Path
from collections.abc import Mapping, Hashable
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]

# ----------------------------------------------------------------------
# Perceptual hashing (from Parent A)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Perceptual hash: compare each value to the mean of the first 64 entries."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

# ----------------------------------------------------------------------
# Structural Similarity Index (SSIM) (from Parent B)
# ----------------------------------------------------------------------
def ssim_1d(x: np.ndarray, y: np.ndarray, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """
    Simplified SSIM for 1‑D vectors.
    Returns a similarity in [0, 1].
    """
    if x.shape != y.shape:
        raise ValueError("Vectors must have the same shape")
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator) if denominator != 0 else 0.0

# ----------------------------------------------------------------------
# StoreState (from Parent B) – used as adaptive controller
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: float, outflow: float) -> Tuple[float, float]:
        """Update internal level based on scalar inflow/outflow."""
        delta = self.alpha * inflow - self.beta * outflow
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """Derived parameter that can be fed back into probabilities."""
        delta = getattr(self, "_last_delta", 0.0)
        return max(0.0, min(self.limit, self.base + self.gain * delta))

# ----------------------------------------------------------------------
# Broadcast probability with SSIM modulation (new hybrid logic)
# ----------------------------------------------------------------------
def broadcast_probability(base_phase: int, current_phase: int, similarity: float, eps: float = 0.05) -> float:
    """
    Base probability: 1 / 2^(base_phase - current_phase)  (clamped to 1).
    similarity ∈ [0,1] reduces the chance to broadcast: higher similarity → lower p.
    eps guarantees a minimum probability.
    """
    if base_phase < 1 or current_phase < 1:
        raise ValueError("phases must be positive")
    base = min(1.0, 1.0 / (2 ** max(0, base_phase - current_phase)))
    modulated = base * (1.0 - similarity) + eps
    return min(1.0, modulated)

# ----------------------------------------------------------------------
# Hybrid maximal independent set (core function)
# ----------------------------------------------------------------------
def hybrid_maximal_independent_set(
    graph: Graph,
    node_features: Dict[Node, List[float]],
    phases: int = 8,
    seed: int | str | None = None,
) -> Set[Node]:
    """
    Computes a maximal independent set where broadcast probabilities are
    modulated by SSIM similarity to the global feature centroid.
    The process is guided by a StoreState instance that adapts over phases.
    """
    rng = random.Random(seed)
    undecided: Set[Node] = set(graph)
    leaders: Set[Node] = set()
    blocked: Set[Node] = set()

    # Pre‑compute feature matrix and global centroid
    feature_matrix = np.array([node_features[n] for n in graph], dtype=float)
    centroid = feature_matrix.mean(axis=0)

    # Pre‑compute per‑node SSIM similarity to centroid
    similarities: Dict[Node, float] = {}
    for n, vec in node_features.items():
        similarities[n] = ssim_1d(np.array(vec, dtype=float), centroid)

    # Adaptive controller
    controller = StoreState()

    for phase in range(1, phases + 1):
        if not undecided:
            break

        # Determine broadcast set with similarity‑aware probability
        broadcasts = {
            n
            for n in undecided
            if rng.random()
            < broadcast_probability(phases, phase, similarities[n])
        }

        # Nodes that broadcast without neighbor conflict become leaders
        new_leaders = {
            n
            for n in broadcasts
            if not (graph.get(n, set()) & broadcasts)
        }

        # Update state tracking
        leaders.update(new_leaders)

        newly_blocked = set().union(
            *(graph.get(n, set()) for n in new_leaders),
            new_leaders,
        )
        blocked.update(newly_blocked)

        # Remove blocked nodes from future consideration
        undecided -= blocked

        # StoreState update: inflow = #new leaders, outflow = #newly blocked - #new leaders
        inflow = float(len(new_leaders))
        outflow = float(len(newly_blocked) - len(new_leaders))
        controller.update(inflow, outflow)

        # Optionally adjust similarities using the controller's "dance" as a scaling factor
        # (makes the algorithm more/less aggressive in later phases)
        scaling = 1.0 + 0.1 * (controller.dance - controller.base)
        for n in undecided:
            # Pull similarity slightly towards 0.5 to avoid extreme freeze‑out
            sim = similarities[n]
            similarities[n] = max(0.0, min(1.0, sim * scaling))

    # Final sweep: any remaining undecided node that does not neighbor a leader becomes a leader
    for n in sorted(undecided, key=str):
        if not (graph.get(n, set()) & leaders):
            leaders.add(n)

    return leaders

# ----------------------------------------------------------------------
# Helper: build a random undirected graph for testing
# ----------------------------------------------------------------------
def random_graph(num_nodes: int, edge_prob: float = 0.2, seed: int | None = None) -> Graph:
    rng = random.Random(seed)
    nodes = [f"n{i}" for i in range(num_nodes)]
    adj: Dict[Node, Set[Node]] = {n: set() for n in nodes}
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < edge_prob:
                adj[nodes[i]].add(nodes[j])
                adj[nodes[j]].add(nodes[i])
    return adj

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a small random graph
    G = random_graph(12, edge_prob=0.25, seed=42)

    # Assign random 5‑dimensional feature vectors to each node
    rng = random.Random(123)
    feats: Dict[Node, List[float]] = {
        n: [rng.random() for _ in range(5)] for n in G
    }

    # Run the hybrid MIS algorithm
    leaders = hybrid_maximal_independent_set(G, feats, phases=6, seed=999)

    print("Graph adjacency:")
    for n, nbrs in G.items():
        print(f"  {n}: {sorted(nbrs)}")
    print("\nSelected leaders:", sorted(leaders))
    print("\nNumber of leaders:", len(leaders))