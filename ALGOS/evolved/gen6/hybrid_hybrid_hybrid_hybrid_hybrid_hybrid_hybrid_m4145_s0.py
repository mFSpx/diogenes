# DARWIN HAMMER — match 4145, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m594_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_hybrid_m1214_s1.py (gen5)
# born: 2026-05-29T23:53:50Z

"""
Hybrid Algorithm: Fisher‑Caputo‑Sketch‑Bandit Fusion

Parents:
- hybrid_hybrid_hybrid_fisher_hybrid_hybrid_m594_s1 (Fisher score weighting + Caputo fractional decay)
- hybrid_hybrid_hybrid_sketch_hybrid_hybrid_m1214_s1 (Count‑Min sketch + Bandit‑driven labeling + Ollivier‑Ricci curvature)

Mathematical Bridge:
The Fisher score, originally used to weight similarity in packet routing, is applied as a
multiplicative factor to the estimated frequencies obtained from a Count‑Min sketch.
These weighted frequencies are then evolved in time by a discrete Caputo fractional
derivative, modelling pheromone decay.  The resulting scalar serves as the expected
reward for a contextual bandit that selects a labeling function.  The bandit’s
confidence bound is further modulated by an Ollivier‑Ricci curvature estimate,
capturing graph‑theoretic recovery priority.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (merged from both parents)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Count‑Min sketch utilities (Parent B)
# ----------------------------------------------------------------------
def count_min_sketch(items: Iterable[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Return a Count‑Min sketch table for the given iterable of hashable items."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            index = hash((item, d)) % width
            table[d][index] += 1
    return table


def estimate_frequency(sketch: List[List[int]], item: str) -> int:
    """Estimate the frequency of *item* using the minimum across hash rows."""
    estimates = []
    for d, row in enumerate(sketch):
        index = hash((item, d)) % len(row)
        estimates.append(row[index])
    return min(estimates)


# ----------------------------------------------------------------------
# Fisher score (Parent A)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher score derived from the Gaussian beam intensity."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    # derivative of log‑likelihood for Gaussian w.r.t. theta
    derivative = -(theta - center) / (width ** 2) * intensity
    return derivative ** 2 / (intensity + eps)


# ----------------------------------------------------------------------
# Caputo fractional decay (Parent A)
# ----------------------------------------------------------------------
def caputo_fractional_decay(
    values: np.ndarray, order: float = 0.5, dt: float = 1.0
) -> np.ndarray:
    """
    Apply a simple discrete Caputo fractional derivative to *values*.
    The implementation follows the Grünwald‑Letnikov approximation with
    coefficients derived from the binomial series.
    """
    if not 0 < order < 1:
        raise ValueError("order must be in (0,1) for fractional decay")
    n = len(values)
    coeffs = np.array([(-1) ** k * math.comb(order, k) for k in range(n)])
    # cumulative sum implements the convolution kernel of the Caputo derivative
    decayed = np.zeros_like(values, dtype=float)
    for i in range(n):
        decayed[i] = dt ** (-order) * np.sum(coeffs[: i + 1] * values[i :: -1])
    return decayed


# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature placeholder (Parent B)
# ----------------------------------------------------------------------
def ollivier_ricci_curvature(node_degree: int, avg_neighbor_degree: float) -> float:
    """
    Very rough surrogate for Ollivier‑Ricci curvature on an unweighted graph:
        κ = 1 - (avg_neighbor_degree / node_degree)
    Clipped to [-1, 1].
    """
    if node_degree == 0:
        return 0.0
    kappa = 1.0 - (avg_neighbor_degree / node_degree)
    return max(-1.0, min(1.0, kappa))


# ----------------------------------------------------------------------
# Hybrid core functions (demonstrate the fused mathematics)
# ----------------------------------------------------------------------
def weighted_sketch_reward(
    sketch: List[List[int]],
    item: str,
    theta: float,
    center: float,
    width: float,
) -> float:
    """
    Combine Count‑Min sketch frequency estimate with a Fisher‑score weighting.
    The result is interpreted as a reward signal for the bandit.
    """
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
    """
    Apply Caputo fractional decay to the vector of sketch frequencies for *items*,
    then feed the decayed values as inflow/outflow to the StoreState.
    """
    # Build raw frequency vector
    raw = np.array([estimate_frequency(sketch, it) for it in items], dtype=float)
    # Fractional decay
    decayed = caputo_fractional_decay(raw, order=order, dt=store.dt)
    # Treat decayed values as inflow; outflow is the complement to a fixed budget
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
    """
    Bandit UCB‑style selection where the expected reward is the Fisher‑weighted
    sketch reward, and the confidence bound is scaled by Ollivier‑Ricci curvature.
    """
    curvature = ollivier_ricci_curvature(node_degree, avg_neighbor_degree)
    best_score = -math.inf
    chosen = None
    for act in actions:
        base_reward = weighted_sketch_reward(sketch, item, theta, center, width)
        # Blend action‑specific expected reward with the base reward
        exp_reward = 0.7 * act.expected_reward + 0.3 * base_reward
        # Upper confidence bound (UCB) with curvature modulation
        ucb = exp_reward + act.confidence_bound * (1 + curvature)
        if ucb > best_score:
            best_score = ucb
            chosen = BanditAction(
                action_id=act.action_id,
                propensity=act.propensity,
                expected_reward=exp_reward,
                confidence_bound=act.confidence_bound,
                algorithm=act.algorithm,
            )
    if chosen is None:
        raise RuntimeError("No actions available for selection")
    return chosen


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a synthetic document stream
    docs = [f"doc_{i}" for i in range(20)]
    # Build a Count‑Min sketch over document identifiers
    cms = count_min_sketch(docs, width=32, depth=3)

    # Initialise a StoreState
    store = StoreState(level=5.0, alpha=0.9, beta=0.4, dt=1.0, base=2.0, gain=0.5, limit=8.0)

    # Define a small pool of bandit actions (labeling functions)
    actions = [
        BanditAction(
            action_id="lf_A",
            propensity=0.6,
            expected_reward=1.2,
            confidence_bound=0.3,
            algorithm="labeler_A",
        ),
        BanditAction(
            action_id="lf_B",
            propensity=0.4,
            expected_reward=0.9,
            confidence_bound=0.5,
            algorithm="labeler_B",
        ),
    ]

    # Parameters for Fisher score
    theta, center, width = 0.75, 0.5, 0.2

    # Simulate a single decision step
    target_item = random.choice(docs)
    selected = select_labeling_action(
        actions=actions,
        sketch=cms,
        item=target_item,
        node_degree=5,
        avg_neighbor_degree=3.2,
        theta=theta,
        center=center,
        width=width,
    )
    print(f"Selected action: {selected.action_id}")

    # Update store with fractional decay of frequencies
    store = decay_and_update_store(store, cms, docs, order=0.6)
    print(f"Store level after update: {store.level:.3f}, dance value: {store.dance:.3f}")

    # Demonstrate reward computation directly
    reward = weighted_sketch_reward(cms, target_item, theta, center, width)
    print(f"Fisher‑weighted reward for '{target_item}': {reward:.4f}")