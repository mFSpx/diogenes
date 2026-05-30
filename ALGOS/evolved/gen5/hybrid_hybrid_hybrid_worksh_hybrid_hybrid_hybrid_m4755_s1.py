# DARWIN HAMMER — match 4755, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s3.py (gen2)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_minimum_cost__m693_s2.py (gen4)
# born: 2026-05-29T23:57:54Z

"""Hybrid allocation algorithm merging weekday‑based sinusoidal weighting (Parent A)
with a distance‑modulated confidence bandit tree (Parent B).

Mathematical bridge:
- Parent A produces a prior probability vector **w** ∈ ℝⁿ for *n* groups
  via a sinusoidal weekday weight vector.
- Parent B supplies a confidence modifier **c** ∈ ℝⁿ derived from a
  minimum‑cost tree where each group is a node; the confidence for a
  (context, action) pair is
      c_i = 1 / (1 + α·d_i)   where d_i is the root‑to‑node distance and
      α = confidence_alpha.
- The hybrid allocation multiplies the prior and confidence element‑wise,
  normalises the product, and distributes the residual portion of the
  total units accordingly:
      a_i = residual · (w_i·c_i) / Σ_j (w_j·c_j)

The deterministic share is allocated directly, while the residual share
is split using the fused probabilities. This single unified system
preserves the temporal sinusoid of Parent A and the adaptive, distance‑aware
exploration of Parent B.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants (shared)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1

# ----------------------------------------------------------------------
# Parent A utilities
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    return (dt.date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    Sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


# ----------------------------------------------------------------------
# Parent B core – HybridBanditTree
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Point:
    """Immutable 2‑D point."""
    x: float
    y: float


@dataclass(frozen=True)
class BanditUpdate:
    """Observation used to update the hybrid policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float = 1.0


class HybridBanditTree:
    """
    Minimum‑cost tree combined with a contextual bandit.
    The tree provides a distance‑based confidence term that modulates
    prior probabilities supplied by another component (here, the weekday
    weight vector).
    """

    def __init__(
        self,
        nodes: Dict[str, Point],
        edges: List[Tuple[str, str]],
        root: str,
        path_weight: float = 0.2,
        confidence_alpha: float = 1.0,
    ):
        self._validate_graph(nodes, edges, root)
        self.nodes = nodes
        self.root = root
        self.path_weight = path_weight
        self.confidence_alpha = confidence_alpha

        # adjacency list
        self.adj: Dict[str, List[str]] = {n: [] for n in nodes}
        for a, b in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)

        # pre‑compute root‑to‑node distances (in edge count)
        self.root_distances = self._bfs_distances(root)

        # statistics for each (context, action) pair: list of observed rewards
        self._stats: Dict[Tuple[str, str], List[float]] = {}

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_graph(
        nodes: Dict[str, Point],
        edges: List[Tuple[str, str]],
        root: str,
    ) -> None:
        if root not in nodes:
            raise ValueError("Root node not in nodes")
        for a, b in edges:
            if a not in nodes or b not in nodes:
                raise ValueError(f"Edge ({a},{b}) references unknown node")

    def _bfs_distances(self, start: str) -> Dict[str, int]:
        """Breadth‑first search returning hop‑count distance from *start*."""
        dist = {start: 0}
        frontier = [start]
        while frontier:
            node = frontier.pop(0)
            for nbr in self.adj[node]:
                if nbr not in dist:
                    dist[nbr] = dist[node] + 1
                    frontier.append(nbr)
        return dist

    # ------------------------------------------------------------------
    # Confidence term
    # ------------------------------------------------------------------
    def confidence(self, context_id: str, action_id: str) -> float:
        """
        Compute a distance‑modulated confidence value for a (context, action)
        pair. The base term is inversely proportional to the tree distance,
        scaled by ``confidence_alpha``. A small epsilon avoids division by zero.
        """
        d = self.root_distances.get(action_id, 0)
        eps = 1e-9
        base_conf = 1.0 / (1.0 + self.confidence_alpha * d + eps)

        # Optional: incorporate observed reward variance (here we use mean reward)
        key = (context_id, action_id)
        if key in self._stats and self._stats[key]:
            avg_reward = sum(self._stats[key]) / len(self._stats[key])
            # Reward boosts confidence multiplicatively, bounded in [0,2]
            reward_factor = 1.0 + math.tanh(avg_reward)
            return base_conf * reward_factor
        return base_conf

    # ------------------------------------------------------------------
    # Update statistics
    # ------------------------------------------------------------------
    def update(self, observation: BanditUpdate) -> None:
        """Record a reward observation for later confidence adjustment."""
        key = (observation.context_id, observation.action_id)
        self._stats.setdefault(key, []).append(observation.reward)

    # ------------------------------------------------------------------
    # Vectorised confidence for a list of actions
    # ------------------------------------------------------------------
    def confidence_vector(self, context_id: str, actions: Sequence[str]) -> np.ndarray:
        """Return an array of confidence values aligned with *actions*."""
        confs = [self.confidence(context_id, act) for act in actions]
        return np.array(confs, dtype=np.float64)


# ----------------------------------------------------------------------
# Hybrid functions demonstrating the fused system
# ----------------------------------------------------------------------
def fused_prior(confidence_vec: np.ndarray, weight_vec: np.ndarray) -> np.ndarray:
    """
    Element‑wise product of prior weight vector and confidence vector,
    followed by normalisation to a probability simplex.
    """
    if weight_vec.shape != confidence_vec.shape:
        raise ValueError("Shape mismatch between weight and confidence vectors")
    raw = weight_vec * confidence_vec
    total = raw.sum()
    if total == 0:
        # fallback to uniform distribution
        return np.full_like(raw, 1.0 / raw.size)
    return raw / total


def allocate_hybrid_with_bandit(
    *,
    total_units: float,
    date: dt.date,
    tree: HybridBanditTree,
    context_id: str,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
) -> Dict[str, Any]:
    """
    Full allocation pipeline:

    1. Deterministic share = ``total_units * deterministic_target_pct / 100``.
    2. Residual share = total_units − deterministic_share.
    3. Prior weight vector from weekday sinusoid (Parent A).
    4. Confidence vector from tree (Parent B).
    5. Fuse them, normalise, and distribute residual among *groups*.
    6. Return a dict mirroring the schema of Parent A's ``allocate_hybrid``.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")

    deterministic_share = total_units * deterministic_target_pct / 100.0
    residual_share = total_units - deterministic_share

    dow = date.weekday()  # Monday=0 … Sunday=6 (matches Parent A's convention after +1)
    weight_vec = weekday_weight_vector(groups, dow)

    confidence_vec = tree.confidence_vector(context_id, groups)

    fused_prob = fused_prior(confidence_vec, weight_vec)

    residual_allocation = residual_share * fused_prob

    # Assemble output similar to Parent A's schema
    allocation = {
        "date": date.isoformat(),
        "total_units": _pct(total_units),
        "deterministic_share": _pct(deterministic_share),
        "residual_share": _pct(residual_share),
        "groups": {},
        "metadata": {
            "weekday": dow,
            "weight_vector": weight_vec.tolist(),
            "confidence_vector": confidence_vec.tolist(),
            "fused_probability": fused_prob.tolist(),
        },
    }

    for grp, amount in zip(groups, residual_allocation):
        allocation["groups"][grp] = _pct(amount)

    return allocation


def simulate_bandit_updates(
    tree: HybridBanditTree,
    context_id: str,
    actions: Sequence[str],
    n_updates: int = 10,
    reward_range: Tuple[float, float] = (-1.0, 1.0),
) -> None:
    """
    Generate synthetic BanditUpdate observations to populate the tree's
    statistics. This function is useful for demonstration / testing.
    """
    for _ in range(n_updates):
        act = random.choice(actions)
        reward = random.uniform(*reward_range)
        obs = BanditUpdate(
            context_id=context_id,
            action_id=act,
            reward=reward,
            propensity=1.0,
        )
        tree.update(obs)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny tree where each group is a node.
    nodes = {
        "codex": Point(0.0, 0.0),
        "groq": Point(1.0, 0.0),
        "cohere": Point(1.0, 1.0),
        "local_models": Point(0.0, 1.0),
    }
    edges = [
        ("codex", "groq"),
        ("groq", "cohere"),
        ("cohere", "local_models"),
        ("local_models", "codex"),
    ]  # forms a simple cycle; tree logic will ignore the extra edge

    tree = HybridBanditTree(
        nodes=nodes,
        edges=edges,
        root="codex",
        path_weight=0.2,
        confidence_alpha=0.5,
    )

    # Populate with fake observations
    simulate_bandit_updates(tree, context_id="today", actions=GROUPS, n_updates=20)

    today = dt.date.today()
    result = allocate_hybrid_with_bandit(
        total_units=1000.0,
        date=today,
        tree=tree,
        context_id="today",
        deterministic_target_pct=80.0,
        groups=GROUPS,
    )

    # Print a concise representation
    print("Hybrid allocation result:")
    for key, value in result.items():
        if key == "groups":
            print("  groups:")
            for g, amt in value.items():
                print(f"    {g}: {amt}")
        else:
            print(f"  {key}: {value}")