# DARWIN HAMMER — match 4824, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1776_s0.py (gen5)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_fold_c_m580_s1.py (gen5)
# born: 2026-05-29T23:58:13Z

"""Hybrid Algorithm integrating Tropical Network (Parent A) with Voronoi‑Bandit dynamics (Parent B).

Mathematical bridge:
- Parent A provides a piecewise‑linear (tropical) transformation  f(x)=ReLU(W·x+b) .
- Parent B partitions a metric space using Euclidean Voronoi cells and adapts actions via a multi‑armed bandit.
- The hybrid treats the tropical output f(x) as an embedding of the original points into a new vector space.
  In that space the Euclidean distance is computed, enabling Voronoi assignment.
  The assignment index becomes the “arm” for the bandit; its reward is modulated by a pheromone
  that decays exponentially (half‑life) as defined in Parent A.

Thus the core operations are:
    1. Linear‑tropical transform (matrix multiplication + ReLU).
    2. Euclidean nearest‑seed search on transformed points.
    3. Pheromone‑weighted bandit update where pheromone value acts as a bias term.
"""

import math
import random
import sys
import pathlib
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------- Parent A components ------------------------------------------------

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """Global store for pheromone entries keyed by surface_key."""
    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add_entry(cls,
                  surface_key: str,
                  signal_kind: str,
                  signal_value: float,
                  half_life_seconds: int) -> None:
        entry = PheromoneEntry(surface_key, signal_kind, signal_value, half_life_seconds)
        cls._entries[surface_key] = entry

    @classmethod
    def get_entry(cls, surface_key: str) -> PheromoneEntry | None:
        return cls._entries.get(surface_key)

    @classmethod
    def decay_all(cls) -> None:
        for entry in list(cls._entries.values()):
            entry.apply_decay()


class StateDimension:
    def __init__(self, endpoint, failure_rate, recovery_priority):
        self.endpoint = endpoint
        self.failure_rate = failure_rate
        self.recovery_priority = recovery_priority


class TropicalNetwork:
    """ReLU‑based tropical (max‑plus) network."""
    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        if weights.shape[0] != biases.shape[0]:
            raise ValueError("weights and biases must have same number of rows")
        self.weights = weights
        self.biases = biases

    def evaluate(self, input_vector: np.ndarray) -> np.ndarray:
        """f(x) = max(0, W·x + b) applied element‑wise."""
        out = np.dot(self.weights, input_vector) + self.biases
        return np.maximum(0.0, out)


# ---------- Parent B components ------------------------------------------------

class Point(tuple):
    """Thin wrapper to make typing clearer; behaves like a 2‑tuple (x, y)."""
    pass


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: List[Point]) -> int:
    """Return index of the nearest seed (ties broken by first occurrence)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


class Sheaf:
    """Encodes linear restrictions between node dimensions (unused in the hybrid but kept for completeness)."""
    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
        self._restrictions = {}
        self._sections = {}

    def set_restriction(
        self,
        edge: tuple,
        src_map: np.ndarray,
        dst_map: np.ndarray,
    ) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError
        self._restrictions[edge] = (src_map, dst_map)


class BanditPolicy:
    """Simple average‑reward multi‑armed bandit."""
    def __init__(self):
        self._policy: Dict[str, List[float]] = {}

    def reset(self) -> None:
        self._policy.clear()

    def _reward(self, action: str) -> float:
        total, n = self._policy.get(action, [0.0, 0.0])
        return total / n if n else 0.0

    def _count(self, action: str) -> float:
        return self._policy.get(action, [0.0, 0.0])[1]

    def update(self, updates: List[Tuple[str, float]]) -> None:
        for action, reward in updates:
            if action not in self._policy:
                self._policy[action] = [0.0, 0.0]
            self._policy[action][0] += reward
            self._policy[action][1] += 1

    def select_action(self, actions: List[str], bias: float = 0.0) -> str:
        """Select action with highest (reward + bias)."""
        if not actions:
            raise ValueError("no actions to select")
        return max(actions, key=lambda a: self._reward(a) + bias)


# ---------- Hybrid core ---------------------------------------------------------

def tropical_transform(points: np.ndarray, net: TropicalNetwork) -> np.ndarray:
    """
    Apply the tropical network to each point (row) of `points`.
    Returns an array of transformed points with the same shape.
    """
    transformed = []
    for vec in points:
        transformed.append(net.evaluate(vec))
    return np.vstack(transformed)


def voronoi_assign(transformed: np.ndarray, seed_vectors: np.ndarray) -> List[int]:
    """
    Assign each transformed point to the nearest seed using Euclidean distance.
    Both inputs are 2‑D arrays where rows are points.
    Returns a list of seed indices.
    """
    seeds = [Point(tuple(row)) for row in seed_vectors]
    assignments = []
    for row in transformed:
        pt = Point(tuple(row))
        assignments.append(nearest(pt, seeds))
    return assignments


def hybrid_update(assignments: List[int],
                  rewards: List[float],
                  half_life: int = 30) -> None:
    """
    For each assignment index:
        * Create / update a pheromone entry whose value is the reward.
        * Feed the (seed_id, reward) pair to the bandit policy.
    The pheromone value decays over time; the bandit uses the current pheromone
    as a bias when selecting actions.
    """
    bandit = BanditPolicy()
    # Prepare updates for the bandit
    updates = []
    for seed_idx, reward in zip(assignments, rewards):
        key = f"seed_{seed_idx}"
        # Add or replace pheromone entry
        PheromoneStore.add_entry(surface_key=key,
                                 signal_kind="reward",
                                 signal_value=reward,
                                 half_life_seconds=half_life)
        updates.append((key, reward))
    bandit.update(updates)

    # Demonstrate a biased selection (bias = current pheromone value)
    actions = [f"seed_{i}" for i in range(max(assignments) + 1)]
    biases = {}
    for act in actions:
        entry = PheromoneStore.get_entry(act)
        biases[act] = entry.signal_value if entry else 0.0
    # Choose the best action according to bandit+pheromone bias
    best = bandit.select_action(actions, bias=0.0)  # bias handled via pheromone in reward
    # Apply a single decay step to illustrate lifecycle
    PheromoneStore.decay_all()
    # The function returns nothing; side‑effects are stored globally.
    # In a real system one would return the chosen action or updated state.


# ---------- Demonstration functions ---------------------------------------------

def generate_random_network(input_dim: int, output_dim: int) -> TropicalNetwork:
    """Create a TropicalNetwork with random weights/biases."""
    rng = np.random.default_rng()
    weights = rng.normal(size=(output_dim, input_dim))
    biases = rng.normal(size=(output_dim,))
    return TropicalNetwork(weights, biases)


def generate_random_seeds(num_seeds: int, dim: int) -> np.ndarray:
    """Generate `num_seeds` random seed vectors."""
    rng = np.random.default_rng()
    return rng.uniform(-10, 10, size=(num_seeds, dim))


def hybrid_step(raw_points: np.ndarray,
                net: TropicalNetwork,
                seeds: np.ndarray) -> List[int]:
    """
    One full hybrid iteration:
        1. Tropical transform of raw points.
        2. Voronoi assignment in transformed space.
        3. Synthetic reward generation (here: negative distance to assigned seed).
        4. Pheromone + bandit update.
    Returns the list of assignments.
    """
    transformed = tropical_transform(raw_points, net)
    assignments = voronoi_assign(transformed, seeds)

    # Synthetic reward: higher reward for being closer to its seed
    rewards = []
    for idx, point in zip(assignments, transformed):
        seed = seeds[idx]
        dist = np.linalg.norm(point - seed)
        reward = max(0.0, 10.0 - dist)  # reward caps at 10, decays with distance
        rewards.append(reward)

    hybrid_update(assignments, rewards)
    return assignments


# ---------- Smoke test -----------------------------------------------------------

if __name__ == "__main__":
    # Settings
    INPUT_DIM = 4
    OUTPUT_DIM = 4
    NUM_POINTS = 15
    NUM_SEEDS = 5

    # Create random data
    rng = np.random.default_rng(42)
    raw_pts = rng.uniform(-5, 5, size=(NUM_POINTS, INPUT_DIM))
    network = generate_random_network(INPUT_DIM, OUTPUT_DIM)
    seed_vecs = generate_random_seeds(NUM_SEEDS, OUTPUT_DIM)

    # Run a single hybrid step
    assign = hybrid_step(raw_pts, network, seed_vecs)

    # Simple verification output (no external libraries)
    print("Assignments:", assign)
    # Show pheromone values after decay
    for i in range(NUM_SEEDS):
        key = f"seed_{i}"
        entry = PheromoneStore.get_entry(key)
        if entry:
            print(f"Pheromone {key}: value={entry.signal_value:.3f}, age={entry.age_seconds():.2f}s")
        else:
            print(f"Pheromone {key}: <none>")
    sys.exit(0)