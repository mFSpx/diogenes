# DARWIN HAMMER — match 4145, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m594_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_hybrid_m1214_s1.py (gen5)
# born: 2026-05-29T23:53:50Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple
import numpy as np

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = ""  # optional identifier


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))


def count_min_sketch(items: Iterable[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            index = hash((item, d)) % width
            table[d][index] += 1
    return table


def estimate_frequency(sketch: List[List[int]], item: str) -> int:
    estimates = []
    for d, row in enumerate(sketch):
        index = hash((item, d)) % len(row)
        estimates.append(row[index])
    return min(estimates)


def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = -(theta - center) / (width ** 2) * intensity
    return derivative ** 2 / (intensity + eps)


def caputo_fractional_decay(
    values: np.ndarray, order: float = 0.5, dt: float = 1.0
) -> np.ndarray:
    if not 0 < order < 1:
        raise ValueError("order must be in (0,1) for fractional decay")
    n = len(values)
    coeffs = np.array([(-1) ** k * math.comb(order, k) for k in range(n)])
    decayed = np.zeros_like(values, dtype=float)
    for i in range(n):
        decayed[i] = dt ** (-order) * np.sum(coeffs[: i + 1] * values[i :: -1])
    return decayed


def ollivier_ricci_curvature(node_degree: int, avg_neighbor_degree: float) -> float:
    if node_degree == 0:
        return 0.0
    kappa = 1.0 - (avg_neighbor_degree / node_degree)
    return max(-1.0, min(1.0, kappa))


def weighted_sketch_reward(
    sketch: List[List[int]],
    item: str,
    theta: float,
    center: float,
    width: float,
) -> float:
    freq_est = estimate_frequency(sketch, item)
    fisher = fisher_score(theta, center, width)
    reward = fisher * freq_est
    return reward


def decay_and_update_store(
    store: StoreState,
    sketch: List[List[int]],
    items: List[str],
    order: float = 0.5,
) -> StoreState:
    raw = np.array([estimate_frequency(sketch, it) for it in items], dtype=float)
    decayed = caputo_fractional_decay(raw, order=order, dt=store.dt)
    budget = np.full_like(decayed, fill_value=10.0)
    inflow = decayed.tolist()
    outflow = (budget - decayed).clip(min=0).tolist()
    store.update(inflow, outflow)
    return store


def select_labeling_action(
    actions: List[BanditAction],
    sketch: List[List[int]],
    item: str,
    node_degree: int,
    avg_neighbor_degree: float,
    theta: float,
    center: float,
    width: float,
) -> BanditAction:
    curvature = ollivier_ricci_curvature(node_degree, avg_neighbor_degree)
    best_score = -math.inf
    chosen = None
    for act in actions:
        base_reward = weighted_sketch_reward(sketch, item, theta, center, width)
        exp_reward = 0.7 * act.expected_reward + 0.3 * base_reward
        ucb = exp_reward + act.confidence_bound * (1 + curvature)
        if ucb > best_score:
            best_score = ucb
            chosen = BanditAction(
                action_id=act.action_id,
                propensity=act.propensity,
                expected_reward=exp_reward,
                confidence_bound=act.confidence_bound,
            )
    return chosen


def improved_hybrid_algorithm(
    items: List[str],
    theta: float,
    center: float,
    width: float,
    node_degree: int,
    avg_neighbor_degree: float,
    order: float = 0.5,
    dt: float = 1.0,
) -> BanditAction:
    sketch = count_min_sketch(items)
    store = StoreState(dt=dt)
    store = decay_and_update_store(store, sketch, items, order=order)
    actions = [
        BanditAction(action_id="action1", propensity=0.5, expected_reward=10.0, confidence_bound=1.0),
        BanditAction(action_id="action2", propensity=0.3, expected_reward=8.0, confidence_bound=0.8),
    ]
    chosen_action = select_labeling_action(
        actions, sketch, items[0], node_degree, avg_neighbor_degree, theta, center, width
    )
    return chosen_action


# Example usage:
items = ["item1", "item2", "item3"]
theta = 0.5
center = 0.0
width = 1.0
node_degree = 5
avg_neighbor_degree = 3.0
chosen_action = improved_hybrid_algorithm(
    items, theta, center, width, node_degree, avg_neighbor_degree
)
print(chosen_action)