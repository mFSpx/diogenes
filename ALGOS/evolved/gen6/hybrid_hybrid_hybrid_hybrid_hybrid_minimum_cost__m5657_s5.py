# DARWIN HAMMER — match 5657, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2500_s1.py (gen5)
# parent_b: hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s2.py (gen3)
# born: 2026-05-30T00:04:07Z

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

Vector = Sequence[float]


# ----------------------------------------------------------------------
# Utility functions (Parent A)
# ----------------------------------------------------------------------
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
    widths: List[float],
) -> float:
    """
    Sum of Gaussian RBFs centred at `centers` with given `widths`.

    Widths of zero are replaced by a small epsilon to avoid division‑by‑zero.
    """
    eps = 1e-9
    total = 0.0
    for center, width in zip(centers, widths):
        w = max(width, eps)
        r = euclidean(input_vector, center)
        total += gaussian(r, 1.0 / w)
    return total


# ----------------------------------------------------------------------
# Pheromone system (Parent A)
# ----------------------------------------------------------------------
class HybridPheromoneSystem:
    """Simple pheromone decay system."""

    def __init__(self) -> None:
        self._store: Dict[Any, Dict[str, Any]] = {}

    def calculate_pheromone_signal(
        self,
        surface_key: Any,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        now = datetime.now(timezone.utc)
        entry = self._store.get(surface_key)

        if entry is None:
            self._store[surface_key] = {
                "signal_kind": signal_kind,
                "signal_value": signal_value,
                "half_life_seconds": half_life_seconds,
                "created_time": now,
            }
            return signal_value

        elapsed = (now - entry["created_time"]).total_seconds()
        decay = entry["signal_value"] * math.pow(
            0.5, elapsed / entry["half_life_seconds"]
        )
        new_value = decay + signal_value
        self._store[surface_key] = {
            "signal_kind": signal_kind,
            "signal_value": new_value,
            "half_life_seconds": half_life_seconds,
            "created_time": now,
        }
        return new_value

    def get_factor(self, surface_key: Any) -> float:
        """Return a multiplicative factor derived from the current pheromone."""
        entry = self._store.get(surface_key)
        if not entry:
            return 1.0
        # Logistic normalisation to (0,1)
        return 1.0 / (1.0 + math.exp(-entry["signal_value"]))


# ----------------------------------------------------------------------
# Bandit tree (Parent B)
# ----------------------------------------------------------------------
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
    """Bandit‑augmented tree cost evaluator."""

    def __init__(self) -> None:
        # action_id → [cumulative_reward, count]
        self._policy: Dict[str, List[float]] = {}

    def reset_policy(self) -> None:
        self._policy.clear()

    def _average_reward(self, action: str) -> float:
        total, n = self._policy.get(action, [0.0, 0.0])
        return total / n if n > 0 else 0.0

    def update_policy(self, updates: Iterable[BanditUpdate]) -> None:
        for u in updates:
            stats = self._policy.setdefault(u.action_id, [0.0, 0.0])
            stats[0] += float(u.reward)
            stats[1] += 1.0

    @staticmethod
    def _length(a: Point, b: Point) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _bfs_root_distances(
        self,
        nodes: Dict[str, Point],
        adjacency: Dict[str, List[str]],
        root: str,
    ) -> Dict[str, float]:
        """Breadth‑first search returning distance from `root` to every node."""
        dist: Dict[str, float] = {root: 0.0}
        frontier: List[str] = [root]

        while frontier:
            cur = frontier.pop()
            cur_dist = dist[cur]
            for nxt in adjacency[cur]:
                if nxt not in dist:
                    dist[nxt] = cur_dist + self._length(nodes[cur], nodes[nxt])
                    frontier.append(nxt)
        return dist

    def tree_cost(
        self,
        nodes: Dict[str, Point],
        edges: List[Tuple[str, str]],
        edge_weights: List[float],
        root: str,
        path_weight: float,
    ) -> float:
        """Cost = Σ(edge_weight·length) + path_weight·Σ distance_from_root."""
        adjacency: Dict[str, List[str]] = {n: [] for n in nodes}
        material = 0.0

        for (a, b), w in zip(edges, edge_weights):
            adjacency[a].append(b)
            adjacency[b].append(a)
            material += w * self._length(nodes[a], nodes[b])

        root_dist = self._bfs_root_distances(nodes, adjacency, root)
        return material + path_weight * sum(root_dist.values())

    def bandit_score(self) -> float:
        """Sum of average rewards over all actions."""
        return sum(self._average_reward(a) for a in self._policy)


# ----------------------------------------------------------------------
# Fusion layer – deep integration of the three pillars
# ----------------------------------------------------------------------
class HybridFusion:
    """
    Deeply integrates:
      * Gaussian RBF surrogate (Parent A)
      * Pheromone decay (Parent A)
      * Contextual bandit + tree cost (Parent B)

    The fusion is governed by three hyper‑parameters:
      * `alpha` – baseline proportion of raw Euclidean length (0 ≤ α ≤ 1)
      * `lambda_path` – weight of the root‑distance term
      * `pheromone_scale` – exponent applied to pheromone factor when scaling
        bandit rewards (≥0).  A value of 0 disables scaling.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        lambda_path: float = 0.2,
        pheromone_scale: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must lie in [0,1]")
        self.alpha = alpha
        self.lambda_path = lambda_path
        self.pheromone_scale = pheromone_scale
        self.rng = random.Random(random_state)

        self.pheromone_system = HybridPheromoneSystem()
        self.bandit_tree = HybridBanditTree()

    # ------------------------------------------------------------------
    # Edge‑wise RBF weighting
    # ------------------------------------------------------------------
    def _edge_rbf_weights(
        self,
        edges: List[Tuple[str, str]],
        features: Dict[str, Vector],
        centers: List[Vector],
        widths: List[float],
    ) -> List[float]:
        """
        Compute a Gaussian RBF weight for each edge based on the concatenated
        feature vectors of its endpoints.
        """
        if len(centers) != len(widths):
            raise ValueError("centers and widths must have the same length")
        weights: List[float] = []
        for u, v in edges:
            vec = np.concatenate([np.asarray(features[u]), np.asarray(features[v])])
            w = radial_basis_surrogate_model(vec.tolist(), centers, widths)
            weights.append(w)
        return weights

    # ------------------------------------------------------------------
    # Edge cost blending
    # ------------------------------------------------------------------
    def _blended_edge_costs(
        self,
        edges: List[Tuple[str, str]],
        nodes: Dict[str, Point],
        rbf_weights: List[float],
    ) -> List[float]:
        """
        Produce a list of effective edge costs:
            cost = length * (α + (1‑α)·w_RBF)
        where w_RBF ∈ (0,1] for a Gaussian kernel.
        """
        costs: List[float] = []
        for (a, b), w in zip(edges, rbf_weights):
            length = HybridBanditTree._length(nodes[a], nodes[b])
            blended = self.alpha + (1.0 - self.alpha) * w
            costs.append(length * blended)
        return costs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_pheromone_and_bandit(
        self,
        surface_key: Any,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
        bandit_updates: Iterable[BanditUpdate],
    ) -> None:
        """
        Decay/refresh pheromone for a given surface and feed the same reward
        information to the bandit policy.
        """
        self.pheromone_system.calculate_pheromone_signal(
            surface_key, signal_kind, signal_value, half_life_seconds
        )
        self.bandit_tree.update_policy(bandit_updates)

    def hybrid_score(
        self,
        nodes: Dict[str, Point],
        edges: List[Tuple[str, str]],
        root: str,
        features: Dict[str, Vector],
        centers: List[Vector],
        widths: List[float],
        surface_key: Any,
    ) -> float:
        """
        Compute the fully fused score:

            HybridScore =
                Σ_e blended_edge_cost(e)
                + λ·Σ_v dist_root(v)
                + (pheromone_factor(surface_key) ** pheromone_scale)·BanditScore
        """
        # 1️⃣  RBF weights per edge
        rbf_weights = self._edge_rbf_weights(edges, features, centers, widths)

        # 2️⃣  Blend Euclidean length with RBF weight
        blended_costs = self._blended_edge_costs(edges, nodes, rbf_weights)

        # 3️⃣  Tree cost using blended edge weights
        tree_cost = self.bandit_tree.tree_cost(
            nodes,
            edges,
            blended_costs,
            root,
            self.lambda_path,
        )

        # 4️⃣  Scale bandit reward by pheromone factor
        pheromone_factor = self.pheromone_system.get_factor(surface_key)
        scaled_bandit = (
            pheromone_factor ** self.pheromone_scale
        ) * self.bandit_tree.bandit_score()

        return tree_cost + scaled_bandit

    # ------------------------------------------------------------------
    # Convenience method for a single‑step evaluation
    # ------------------------------------------------------------------
    def evaluate_one_step(
        self,
        nodes: Dict[str, Point],
        edges: List[Tuple[str, str]],
        root: str,
        features: Dict[str, Vector],
        centers: List[Vector],
        widths: List[float],
        surface_key: Any,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
        bandit_updates: Iterable[BanditUpdate],
    ) -> float:
        """
        Perform a full update (pheromone + bandit) followed by a score
        computation.  Returns the hybrid score for the current state.
        """
        self.update_pheromone_and_bandit(
            surface_key,
            signal_kind,
            signal_value,
            half_life_seconds,
            bandit_updates,
        )
        return self.hybrid_score(
            nodes,
            edges,
            root,
            features,
            centers,
            widths,
            surface_key,
        )