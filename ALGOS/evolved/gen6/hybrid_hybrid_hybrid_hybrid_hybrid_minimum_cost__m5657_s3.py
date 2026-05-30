# DARWIN HAMMER — match 5657, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2500_s1.py (gen5)
# parent_b: hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s2.py (gen3)
# born: 2026-05-30T00:04:07Z

"""Hybrid algorithm merging DARWIN HAMMER parents:
- Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2500_s1.py): provides
  Gaussian radial‑basis surrogate modeling, Euclidean distance utilities and a
  pheromone decay system.
- Parent B (hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s2.py): provides
  a minimum‑cost spanning‑tree evaluation, a contextual bandit policy and a
  combined “hybrid_bandit_tree” score.

Mathematical bridge:
Both parents rely on Euclidean distances as a metric.  In Parent A the distance
feeds a Gaussian RBF to produce a similarity weight; in Parent B the same
distance contributes linearly to a tree‑cost term.  The fusion replaces the
plain Euclidean edge length in the tree cost with a *RBF‑weighted* length and
adds a pheromone‑scaled bandit reward.  The resulting hybrid score is

    HybridScore = Σ_e w_RBF(e)·‖p_i‑p_j‖  +  λ·Σ_v d_root(v)
                + Σ_a (bandit_reward(a)·pheromone_factor(a))

where w_RBF(e) is the Gaussian surrogate computed from the feature vectors of
the edge’s endpoints, λ is the path‑weight from Parent B, and the pheromone
factor modulates each bandit action’s contribution.  The code below implements
this unified system with three demonstrative functions.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Sequence, Iterable, Any
import numpy as np
from datetime import datetime, timezone

Vector = Sequence[float]

# ---------- Parent A components (adapted) ----------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel used by the RBF surrogate."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def radial_basis_surrogate_model(
    input_vector: Vector,
    centers: List[Vector],
    widths: List[float]
) -> float:
    """Sum of Gaussian RBFs centered at `centers` with given `widths`."""
    return sum(
        gaussian(euclidean(input_vector, center), 1.0 / width)
        for center, width in zip(centers, widths)
    )

class HybridPheromoneSystem:
    """Simple pheromone decay system (Parent A)."""
    def __init__(self):
        self.pheromones: Dict[Any, Dict[str, Any]] = {}

    def calculate_pheromone_signal(
        self,
        surface_key: Any,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float
    ) -> float:
        now = datetime.now(timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {
                "signal_kind": signal_kind,
                "signal_value": signal_value,
                "half_life_seconds": half_life_seconds,
                "created_time": now,
            }
        else:
            prev = self.pheromones[surface_key]
            elapsed = (now - prev["created_time"]).total_seconds()
            decayed = prev["signal_value"] * math.pow(
                0.5, elapsed / prev["half_life_seconds"]
            )
            # store the new (potentially decayed) value together with the fresh one
            self.pheromones[surface_key] = {
                "signal_kind": signal_kind,
                "signal_value": decayed + signal_value,
                "half_life_seconds": half_life_seconds,
                "created_time": now,
            }
        return self.pheromones[surface_key]["signal_value"]

    def get_factor(self, surface_key: Any) -> float:
        """Return a multiplicative factor derived from the current pheromone."""
        entry = self.pheromones.get(surface_key)
        if not entry:
            return 1.0
        # Normalise to [0,1] assuming reasonable signal ranges
        return 1.0 / (1.0 + math.exp(-entry["signal_value"]))

# ---------- Parent B components (adapted) ----------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass(frozen=True)
class Point:
    x: float
    y: float

class HybridBanditTree:
    """Bandit‑augmented tree cost evaluator (Parent B)."""
    def __init__(self):
        self._policy: Dict[str, List[float]] = {}

    def reset_policy(self) -> None:
        self._policy.clear()

    def _reward(self, action: str) -> float:
        total, n = self._policy.get(action, [0.0, 0.0])
        return total / n if n else 0.0

    def update_policy(self, updates: List[BanditUpdate]) -> None:
        for u in updates:
            stats = self._policy.setdefault(u.action_id, [0.0, 0.0])
            stats[0] += float(u.reward)
            stats[1] += 1.0

    @staticmethod
    def length(a: Point, b: Point) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def tree_cost(
        self,
        nodes: Dict[str, Point],
        edges: List[Tuple[str, str]],
        root: str,
        path_weight: float = 0.2,
        edge_weights: List[float] = None,
    ) -> float:
        """Cost = Σ (edge_weight·length) + path_weight·Σ distance_from_root."""
        adj: Dict[str, List[str]] = {n: [] for n in nodes}
        material = 0.0
        for idx, (a, b) in enumerate(edges):
            adj[a].append(b)
            adj[b].append(a)
            w = edge_weights[idx] if edge_weights else 1.0
            material += w * self.length(nodes[a], nodes[b])

        # BFS to compute root distances
        dist: Dict[str, float] = {root: 0.0}
        stack = [root]
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt not in dist:
                    dist[nxt] = dist[cur] + self.length(nodes[cur], nodes[nxt])
                    stack.append(nxt)

        return material + path_weight * sum(dist.values())

    def hybrid_score(
        self,
        nodes: Dict[str, Point],
        edges: List[Tuple[str, str]],
        root: str,
        path_weight: float,
        edge_weights: List[float],
        updates: List[BanditUpdate],
    ) -> float:
        """Combine tree cost with bandit rewards."""
        self.update_policy(updates)
        tree_score = self.tree_cost(nodes, edges, root, path_weight, edge_weights)
        bandit_score = sum(self._reward(a) for a in self._policy)
        return tree_score + bandit_score

# ---------- Fusion layer ----------
def compute_edge_rbf_weights(
    edges: List[Tuple[str, str]],
    features: Dict[str, Vector],
    centers: List[Vector],
    widths: List[float],
) -> List[float]:
    """
    For each edge (u,v) compute a Gaussian RBF weight based on the concatenated
    feature vector of its endpoints.
    """
    weights = []
    for u, v in edges:
        vec = np.concatenate([features[u], features[v]]).astype(float)
        w = radial_basis_surrogate_model(vec.tolist(), centers, widths)
        weights.append(w)
    return weights

def update_pheromone_and_bandit(
    pheromone_system: HybridPheromoneSystem,
    bandit_tree: HybridBanditTree,
    surface_key: Any,
    signal_kind: str,
    signal_value: float,
    half_life_seconds: float,
    bandit_updates: List[BanditUpdate],
) -> None:
    """
    Decay/refresh pheromone for a given surface and feed the same reward
    information to the bandit policy.
    """
    pheromone_system.calculate_pheromone_signal(
        surface_key, signal_kind, signal_value, half_life_seconds
    )
    bandit_tree.update_policy(bandit_updates)

def hybrid_tree_bandit_pheromone_score(
    nodes: Dict[str, Point],
    edges: List[Tuple[str, str]],
    root: str,
    features: Dict[str, Vector],
    centers: List[Vector],
    widths: List[float],
    path_weight: float,
    pheromone_system: HybridPheromoneSystem,
    bandit_tree: HybridBanditTree,
    bandit_updates: List[BanditUpdate],
    surface_key: Any,
) -> float:
    """
    Full hybrid evaluation:
    1. Compute RBF‑based edge weights.
    2. Update pheromone and bandit policy.
    3. Evaluate tree cost with those weights.
    4. Add bandit reward scaled by pheromone factor.
    """
    # 1. Edge weights via RBF surrogate
    edge_weights = compute_edge_rbf_weights(edges, features, centers, widths)

    # 2. Update pheromone and bandit
    update_pheromone_and_bandit(
        pheromone_system,
        bandit_tree,
        surface_key,
        signal_kind="tree_evaluation",
        signal_value=sum(edge_weights),  # aggregate signal strength
        half_life_seconds=30.0,
        bandit_updates=bandit_updates,
    )

    # 3. Base hybrid score from tree + bandit
    base_score = bandit_tree.hybrid_score(
        nodes,
        edges,
        root,
        path_weight,
        edge_weights,
        bandit_updates,
    )

    # 4. Modulate with pheromone factor (higher pheromone → higher influence)
    pheromone_factor = pheromone_system.get_factor(surface_key)
    return base_score * pheromone_factor

# ---------- Smoke test ----------
if __name__ == "__main__":
    # Simple graph: triangle
    nodes = {
        "A": Point(0.0, 0.0),
        "B": Point(1.0, 0.0),
        "C": Point(0.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"

    # Feature vectors for each node (2‑D for simplicity)
    features = {
        "A": [0.1, 0.2],
        "B": [0.4, 0.2],
        "C": [0.2, 0.5],
    }

    # RBF centers and widths (chosen arbitrarily)
    centers = [
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
    ]
    widths = [0.3, 0.6]

    # Bandit updates (simulated)
    bandit_updates = [
        BanditUpdate(context_id="ctx1", action_id="act1", reward=1.0, propensity=0.5),
        BanditUpdate(context_id="ctx2", action_id="act2", reward=0.2, propensity=0.8),
    ]

    pheromone_system = HybridPheromoneSystem()
    bandit_tree = HybridBanditTree()

    score = hybrid_tree_bandit_pheromone_score(
        nodes=nodes,
        edges=edges,
        root=root,
        features=features,
        centers=centers,
        widths=widths,
        path_weight=0.2,
        pheromone_system=pheromone_system,
        bandit_tree=bandit_tree,
        bandit_updates=bandit_updates,
        surface_key="global_tree",
    )

    print(f"Hybrid score: {score:.4f}")