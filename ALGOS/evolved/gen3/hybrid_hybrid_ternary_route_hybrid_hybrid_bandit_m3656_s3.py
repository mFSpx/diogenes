# DARWIN HAMMER — match 3656, survivor 3
# gen: 3
# parent_a: hybrid_ternary_router_hybrid_minimum_cost__m36_s4.py (gen2)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s1.py (gen2)
# born: 2026-05-29T23:51:11Z

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable

import numpy as np


@dataclass
class Edge:
    parent: str
    child: str
    weight: float = 1.0
    prior: float = 0.5  # prior probability that the edge is "reliable"


@dataclass
class Node:
    id: str
    position: Tuple[float, float]
    children: List[str] = field(default_factory=list)
    weight: float = 1.0


class HybridTree:
    """Container for nodes and edges with utilities for distance computation."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str], Edge] = {}

    # --------------------------------------------------------------------- #
    # Tree construction helpers
    # --------------------------------------------------------------------- #
    def add_node(self, node_id: str, position: Tuple[float, float], weight: float = 1.0) -> None:
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id!r} already exists.")
        self.nodes[node_id] = Node(node_id, position, [], weight)

    def add_edge(self, parent: str, child: str, weight: float = 1.0, prior: float = 0.5) -> None:
        if parent not in self.nodes or child not in self.nodes:
            raise KeyError("Both parent and child must be added as nodes before creating an edge.")
        edge_key = (parent, child)
        if edge_key in self.edges:
            raise ValueError(f"Edge {edge_key!r} already exists.")
        self.edges[edge_key] = Edge(parent, child, weight, prior)
        self.nodes[parent].children.append(child)

    # --------------------------------------------------------------------- #
    # Geometry utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def compute_pairwise_distances(self, root: str) -> Dict[str, float]:
        """Breadth‑first traversal computing cumulative Euclidean distance from root."""
        if root not in self.nodes:
            raise KeyError(f"Root node {root!r} not found.")
        distances: Dict[str, float] = {root: 0.0}
        queue = deque([root])
        while queue:
            cur = queue.popleft()
            cur_pos = self.nodes[cur].position
            for child in self.nodes[cur].children:
                if child in distances:
                    continue
                child_pos = self.nodes[child].position
                distances[child] = distances[cur] + self._euclidean(cur_pos, child_pos)
                queue.append(child)
        return distances

    # --------------------------------------------------------------------- #
    # Bayesian edge update
    # --------------------------------------------------------------------- #
    def bayesian_edge_update(
        self,
        evidence: Dict[Tuple[str, str], float],
        likelihood: float = 0.8,
        false_positive: float = 0.1,
    ) -> None:
        """Update each edge's prior using evidence supplied by the bandit module."""
        for key, edge in self.edges.items():
            prior = edge.prior
            eff_likelihood = likelihood * evidence.get(key, 1.0)
            marginal = eff_likelihood * prior + false_positive * (1.0 - prior)
            posterior = prior * eff_likelihood / marginal if marginal > 0 else 0.0
            edge.prior = posterior
            # optionally adjust weight to reflect posterior confidence
            edge.weight = max(0.1, posterior)  # keep a minimal weight to avoid zero‑division

    # --------------------------------------------------------------------- #
    # Integration point: apply bandit propensities to edge weights
    # --------------------------------------------------------------------- #
    def integrate_bandit_scores(self, propensity: Dict[str, float]) -> None:
        """Scale edge weights by the average propensity of its incident nodes."""
        for (parent, child), edge in self.edges.items():
            avg_prop = (propensity.get(parent, 0.0) + propensity.get(child, 0.0)) / 2.0
            # Blend the original weight with the propensity (tunable mixing factor)
            mixing = 0.6
            edge.weight = mixing * edge.weight + (1 - mixing) * avg_prop


class BanditPolicy:
    """Simple incremental estimator for expected reward per action."""

    def __init__(self) -> None:
        self._stats: Dict[str, Tuple[float, int]] = defaultdict(lambda: (0.0, 0))

    def update(self, updates: Iterable[Tuple[str, float]]) -> None:
        """`updates` is an iterable of (action_id, reward)."""
        for action_id, reward in updates:
            total, count = self._stats[action_id]
            self._stats[action_id] = (total + reward, count + 1)

    def expected_reward(self, action_id: str) -> float:
        total, count = self._stats.get(action_id, (0.0, 0))
        return total / count if count else 0.0

    def propensity_scores(self) -> Dict[str, float]:
        """Map each action to a bounded propensity in (0,1)."""
        scores = {}
        for aid in self._stats:
            exp = self.expected_reward(aid)
            scores[aid] = exp / (1.0 + exp)  # logistic‑like scaling
        return scores


def hybrid_bandit_router(
    tree: HybridTree,
    root: str,
    steps: int = 100,
    likelihood: float = 0.8,
    false_positive: float = 0.1,
    random_seed: int | None = None,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Executes the bandit‑tree fusion loop.

    Returns
    -------
    total_material : float
        Sum of Euclidean lengths weighted by posterior edge probabilities.
    propensity : dict
        Propensity scores per node after the simulation.
    distances : dict
        Cumulative distances from ``root`` after the final tree state.
    """
    rng = np.random.default_rng(random_seed)
    policy = BanditPolicy()

    # ----------------------------------------------------------------- #
    # 1. Simulate bandit feedback on each node independently.
    # ----------------------------------------------------------------- #
    for _ in range(steps):
        updates = []
        for node_id in tree.nodes:
            reward = rng.random()  # synthetic reward ∈ [0,1)
            updates.append((node_id, reward))
        policy.update(updates)

    # ----------------------------------------------------------------- #
    # 2. Convert rewards to propensities and feed them back to the tree.
    # ----------------------------------------------------------------- #
    propensities = policy.propensity_scores()
    tree.integrate_bandit_scores(propensities)

    # ----------------------------------------------------------------- #
    # 3. Perform a Bayesian update of edge priors using the same propensities
    #    as evidence (higher propensity → stronger evidence of reliability).
    # ----------------------------------------------------------------- #
    evidence = {edge_key: propensities.get(edge_key[0], 0.0) * propensities.get(edge_key[1], 0.0)
                for edge_key in tree.edges}
    tree.bayesian_edge_update(evidence, likelihood, false_positive)

    # ----------------------------------------------------------------- #
    # 4. Compute the final material metric and distances.
    # ----------------------------------------------------------------- #
    total_material = 0.0
    for (parent, child), edge in tree.edges.items():
        length = HybridTree._euclidean(tree.nodes[parent].position, tree.nodes[child].position)
        total_material += length * edge.prior

    distances = tree.compute_pairwise_distances(root)
    return total_material, propensities, distances


# --------------------------------------------------------------------- #
# Example usage (executed when run as script)
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    # Build a simple tree
    t = HybridTree()
    t.add_node("root", (0.0, 0.0))
    t.add_node("a", (1.0, 0.0))
    t.add_node("b", (0.0, 1.0))
    t.add_node("c", (1.0, 1.0))

    t.add_edge("root", "a")
    t.add_edge("root", "b")
    t.add_edge("a", "c")
    t.add_edge("b", "c")

    # Run the fused algorithm
    material, prop, dist = hybrid_bandit_router(
        tree=t,
        root="root",
        steps=200,
        likelihood=0.85,
        false_positive=0.05,
        random_seed=42,
    )

    print("Total material (posterior weighted length):", material)
    print("\nPropensity scores per node:")
    for nid, val in prop.items():
        print(f"  {nid}: {val:.4f}")

    print("\nDistances from root:")
    for nid, d in dist.items():
        print(f"  {nid}: {d:.4f}")