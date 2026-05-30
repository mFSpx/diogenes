# DARWIN HAMMER — match 5663, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m626_s0.py (gen5)
# parent_b: hybrid_hybrid_bandit_router_workshare_allocator_m60_s1.py (gen2)
# born: 2026-05-30T00:03:59Z

"""
Hybrid algorithm merging hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m626_s0.py and hybrid_hybrid_bandit_router_workshare_allocator_m60_s1.py.

The mathematical bridge between the two structures is the use of the Structural Similarity Index (SSIM) to inform the selection of actions in the bandit router.
The SSIM is used to compute the similarity between the expected rewards of different actions, and the action with the highest similarity is selected.
The StoreState instance is used to modulate the deterministic target percentage in the workshare allocation, allowing for adaptive allocation of large language model (LLM) units based on the current state of the honeybee store.
The hybrid sparse expansion utilities are used to expand the action set, and the locality-sensitive hashing is used to project the expanded vectors onto a high-dimensional space.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

# Constants
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)

# SSIM implementation
def compute_ssim(
    x: list[float],
    y: list[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)

# Hybrid sparse expansion utilities
def hybrid_sparse_expansion(values: list[float], m: int, salt: str = "") -> list[float]:
    """Hash‑based sparse expansion of `values` into a vector of length `m`."""
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hash(f"{salt}:{i}:{r}")
            j = h % m
            sign = 1.0 if r % 2 == 0 else -1.0
            out[j] += sign * v
    return out

# Core data structures
@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""

    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""

    context_id: str
    action_id: str
    reward: float
    propensity: float


# Store dynamics – richer state
@dataclass
class StoreState:
    """Encapsulates the honeybee‑style store and its derived control signal."""

    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: list[float], outflow: list[float]) -> tuple[float, float]:
        """
        Apply the store equation and recompute the dance duration.

        Returns
        -------
        new_level, delta
        """
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ (computed lazily)."""
        delta = getattr(self, "_last_delta", 0.0)
        return max(0.0, min(self.limit, self.base + self.gain * delta))

    def _store_last_delta(self, delta: float) -> None:
        self._last_delta = delta

def select_action(store_state: StoreState, actions: list[BanditAction]) -> BanditAction:
    """
    Select the action with the highest similarity to the store state.

    Args
    -------
    store_state: StoreState
    actions: list[BanditAction]

    Returns
    -------
    BanditAction
    """
    similarities = []
    for action in actions:
        similarity = compute_ssim([store_state.level], [action.expected_reward])
        similarities.append(similarity)
    max_similarity_index = np.argmax(similarities)
    return actions[max_similarity_index]

def update_store_state(store_state: StoreState, inflow: list[float], outflow: list[float]) -> StoreState:
    """
    Update the store state based on the inflow and outflow.

    Args
    -------
    store_state: StoreState
    inflow: list[float]
    outflow: list[float]

    Returns
    -------
    StoreState
    """
    new_level, delta = store_state.update(inflow, outflow)
    store_state._store_last_delta(delta)
    return store_state

def hybrid_sparse_expansion_and_select_action(store_state: StoreState, values: list[float], m: int, salt: str = "") -> BanditAction:
    """
    Hybrid sparse expansion and action selection.

    Args
    -------
    store_state: StoreState
    values: list[float]
    m: int
    salt: str

    Returns
    -------
    BanditAction
    """
    expanded_values = hybrid_sparse_expansion(values, m, salt)
    actions = [BanditAction(f"action_{i}", 0.5, value, 0.1, "hybrid") for i, value in enumerate(expanded_values)]
    return select_action(store_state, actions)

if __name__ == "__main__":
    store_state = StoreState()
    values = [0.1, 0.2, 0.3]
    m = 5
    salt = "salt"
    action = hybrid_sparse_expansion_and_select_action(store_state, values, m, salt)
    print(action)