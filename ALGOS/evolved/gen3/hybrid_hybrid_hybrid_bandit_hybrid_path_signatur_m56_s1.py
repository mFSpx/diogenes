# DARWIN HAMMER — match 56, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_workshare_allocator_m60_s0.py (gen2)
# parent_b: hybrid_path_signature_kan_m30_s0.py (gen1)
# born: 2026-05-29T23:25:30Z

"""
This module implements a hybrid algorithm that combines the bandit router from 
hybrid_bandit_router_workshare_allocator_m60_s0.py with the path signature and 
Kolmogorov-Arnold Networks (KAN) algorithms from hybrid_path_signature_kan_m30_s0.py.
The mathematical bridge between these two structures lies in the representation 
of the path signature as a sequence of iterated integrals, which can be approximated 
using the B-spline basis functions employed in KANs. We use the store state from 
the bandit router to modulate the workshare allocation, and integrate the KAN's 
B-spline basis into the path signature computation to leverage the expressive power 
of neural networks to improve the accuracy of the path signature representation.
"""

import numpy as np
import math
import random
import sys
import pathlib

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

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

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
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
        return self.level / self.limit


def lead_lag_transform(path):
    """
    Lead-lag transform: interleave (lead, lag) channels for causality encoding.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def bspline_basis(x, grid, k=3):
    """
    Evaluate B-spline basis functions of order k at positions x.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (
                (x - t[i]) / denom_l * B[:, i]
                if denom_l > 0 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 0 else np.zeros(N)
            )
            B_new[:, i] = term_l + term_r
        B = B_new

    return B


def hybrid_update(store_state: StoreState, path: np.ndarray) -> np.ndarray:
    """
    Update the store state using the path signature.
    
    Parameters
    ----------
    store_state : StoreState
    path : np.ndarray
    
    Returns
    -------
    new_path : np.ndarray
    """
    lead_lag_path = lead_lag_transform(path)
    basis = bspline_basis(np.arange(len(lead_lag_path)), np.arange(len(lead_lag_path)))
    new_path = np.dot(basis, lead_lag_path)
    store_state.update([np.mean(new_path)], [store_state.level])
    return new_path


def hybrid_bandit_action(store_state: StoreState) -> BanditAction:
    """
    Compute the bandit action based on the store state.
    
    Parameters
    ----------
    store_state : StoreState
    
    Returns
    -------
    bandit_action : BanditAction
    """
    action_id = f"action_{store_state.level}"
    propensity = store_state.dance
    expected_reward = np.random.uniform(0.0, 1.0)
    confidence_bound = np.random.uniform(0.0, 1.0)
    algorithm = "hybrid_bandit_router_workshare_allocator_m60_s0"
    return BanditAction(action_id, propensity, expected_reward, confidence_bound, algorithm)


def hybrid_workshare_allocation(store_state: StoreState) -> float:
    """
    Compute the workshare allocation based on the store state.
    
    Parameters
    ----------
    store_state : StoreState
    
    Returns
    -------
    allocation : float
    """
    allocation = store_state.level / store_state.limit
    return allocation


if __name__ == "__main__":
    store_state = StoreState()
    path = np.random.uniform(0.0, 1.0, size=(10, 2))
    new_path = hybrid_update(store_state, path)
    bandit_action = hybrid_bandit_action(store_state)
    allocation = hybrid_workshare_allocation(store_state)
    print(new_path)
    print(bandit_action)
    print(allocation)