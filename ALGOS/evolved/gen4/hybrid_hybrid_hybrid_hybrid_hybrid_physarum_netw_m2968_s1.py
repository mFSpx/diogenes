# DARWIN HAMMER — match 2968, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s0.py (gen3)
# parent_b: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s0.py (gen3)
# born: 2026-05-29T23:47:00Z

"""
This module integrates the hybrid_hybrid_doomsday_cale_hybrid_geometric_pro_m66_s0 and 
hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s1 algorithms into a single hybrid system.
The mathematical bridge is formed by using the geometric product to compute distances and orientations 
between days and seeds, and the flux-based conductance update primitive to update decision scores 
based on the uncertainty of the decision-making process.
"""

import numpy as np
import random
import math
import sys
import pathlib

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    """Flux-based conductance update primitive."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    """Update conductance based on flow and decay."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def hybrid_update(
    context_id: str,
    action_id: str,
    reward: float,
    propensity: float,
    store: float,
    store_decay: float,
    dt: float,
    base_eta: float,
    alpha: float,
    beta: float,
) -> float:
    """Hybrid update that integrates the bandit update and the TTT model."""
    store = store * store_decay
    store += dt * (base_eta * (reward - store)) + dt * (alpha * (reward - reward) + beta * propensity)
    return store


def hybrid_bandit_ttt(
    context_id: str,
    action_id: str,
    reward: float,
    store: float,
    store_decay: float,
    dt: float,
    base_eta: float,
    alpha: float,
    beta: float,
) -> float:
    """Hybrid bandit TTT that integrates the bandit decision and the TTT model."""
    action = BanditAction(
        action_id=action_id,
        propensity=update_conductance(propensity, reward, dt=dt),
        expected_reward=reward,
        confidence_bound=flux(store, 1.0, reward, 0.0, eps=1e-12),
        algorithm="hybrid",
    )
    store = hybrid_update(context_id, action_id, reward, action.propensity, store, store_decay, dt, base_eta, alpha, beta)
    return action, store


def geometric_product_update(
    multivector: Multivector,
    edge_length: float,
    pressure_a: float,
    pressure_b: float,
    eps: float = 1e-12,
) -> Multivector:
    """Update the geometric product based on the flux-based conductance update primitive."""
    conductance = sum(multivector.blades.values())
    new_blades = {}
    for blade, sign in multivector.blades.items():
        new_blade = _multiply_blades(blade, frozenset())
        new_blades[new_blade] = sign * flux(conductance, edge_length, pressure_a, pressure_b, eps)
    return Multivector(new_blades)


def hybrid_geometric_bandit(
    context_id: str,
    action_id: str,
    reward: float,
    multivector: Multivector,
    edge_length: float,
    store_decay: float,
    dt: float,
    base_eta: float,
    alpha: float,
    beta: float,
) -> tuple:
    """Hybrid geometric bandit that integrates the geometric product and the TTT model."""
    action = BanditAction(
        action_id=action_id,
        propensity=update_conductance(multivector.blades[frozenset()][1], reward, dt=dt),
        expected_reward=reward,
        confidence_bound=flux(multivector.blades[frozenset()][1], edge_length, reward, 0.0, eps=1e-12),
        algorithm="hybrid",
    )
    new_multivector = geometric_product_update(multivector, edge_length, reward, 0.0, eps=1e-12)
    store = hybrid_update(context_id, action_id, reward, action.propensity, new_multivector.blades[frozenset()][1], store_decay, dt, base_eta, alpha, beta)
    return action, new_multivector


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""
    def __init__(self, blades: dict):
        self.blades = blades


class BanditAction:
    """Bandit action with propensity, expected reward, confidence bound, and algorithm."""
    def __init__(self, action_id: str, propensity: float, expected_reward: float, confidence_bound: float, algorithm: str):
        self.action_id = action_id
        self.propensity = propensity
        self.expected_reward = expected_reward
        self.confidence_bound = confidence_bound
        self.algorithm = algorithm


def length(a: tuple, b: tuple) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list."""
    lst = list(indices)
    sign = 1
    # Bubble sort; track swaps
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # Duplicate: e_i * e_i = 1, remove both
                lst.pop(j)
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def tree_metrics(
    nodes: dict,
    edges: list,
    root: str,
) -> tuple:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of ne
    """
    # ... (rest of the implementation remains the same)


if __name__ == "__main__":
    # Smoke test
    action_id = "test_action"
    reward = 1.0
    store = 1.0
    store_decay = 0.1
    dt = 1.0
    base_eta = 1.0
    alpha = 1.0
    beta = 1.0
    action, store = hybrid_bandit_ttt("test_context", action_id, reward, store, store_decay, dt, base_eta, alpha, beta)
    print(action.action_id)
    print(store)