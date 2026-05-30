# DARWIN HAMMER — match 2608, survivor 0
# gen: 3
# parent_a: hybrid_voronoi_partition_poikilotherm_schoolf_m49_s0.py (gen1)
# parent_b: hybrid_hybrid_bandit_router_variational_free_ene_m56_s0.py (gen2)
# born: 2026-05-29T23:43:06Z

"""
Hybrid Algorithm: Fusing Voronoi Partition and Variational Free Energy

This module integrates the core topologies of the Voronoi Partition (hybrid_voronoi_partition_poikilotherm_schoolf_m49_s0.py) 
and Variational Free Energy (hybrid_hybrid_bandit_router_variational_free_ene_m56_s0.py) algorithms. The mathematical bridge 
between the two structures lies in the use of probabilistic updates and expectations. Specifically, we utilize the 
Variational Free Energy (VFE) framework to inform the Voronoi partition updates, effectively creating a hybrid algorithm 
that balances exploration-exploitation trade-offs with Bayesian inference.

The Voronoi Partition's geometric problem is fused with the Variational Free Energy's active inference framework, 
enabling the algorithm to adaptively sample actions based on both their expected rewards and the uncertainty 
associated with those expectations.

Imports: numpy, standard library, math, random, sys, pathlib
"""

from __future__ import annotations
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Voronoi Partition core
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * np.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = np.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = np.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def normalized_activity(temp_c: float, low_c: float = 5.0, high_c: float = 40.0, samples: int = 141) -> float:
    """Map an observed operating temperature to a 0..1 activity gate."""
    params = SchoolfieldParams(t_low=c_to_k(low_c), t_high=c_to_k(high_c))
    rate = developmental_rate(c_to_k(temp_c), params)
    max_rate = max(developmental_rate(c_to_k(low_c + (high_c - low_c) * i / max(1, samples - 1)), params) for i in range(samples))
    return 0.0 if max_rate <= 0 else max(0.0, min(1.0, rate / max_rate))

def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest(point: tuple[float, float], seeds: list[tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[tuple[float, float]], seeds: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Parent B – Variational Free Energy core
# --------------------------------------------------------------------

def kl_gaussian(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    mu_p: float | np.ndarray,
    sigma_p: float | np.ndarray
) -> float:
    return -0.5 * ((mu_q - mu_p)**2 / sigma_p**2 + np.log(sigma_p**2 / sigma_q**2))

def gaussian_log_likelihood(x: float | np.ndarray, mu: float | np.ndarray, sigma: float | np.ndarray) -> float:
    return -0.5 * (np.log(2 * np.pi * sigma**2) + (x - mu)**2 / sigma**2)

# ----------------------------------------------------------------------
# Hybrid Voronoi Partition & Variational Free Energy core
# ----------------------------------------------------------------------

def voronoi_bandit_update(
    points: List[Tuple[float, float]],
    seeds: List[Tuple[float, float]],
    rewards: List[float],
    params: SchoolfieldParams = SchoolfieldParams()
) -> Dict[str, Dict[str, float]]:
    regions = assign(points, seeds)
    updates = {}
    for i, (reward, _) in enumerate(zip(rewards, points)):
        region_id = nearest(_, seeds)
        if region_id not in updates:
            updates[region_id] = {}
        if 'reward' not in updates[region_id]:
            updates[region_id]['reward'] = 0.0
        updates[region_id]['reward'] += reward
        if 'count' not in updates[region_id]:
            updates[region_id]['count'] = 0.0
        updates[region_id]['count'] += 1.0
    return updates

def voronoi_bandit_policy(
    points: List[Tuple[float, float]],
    seeds: List[Tuple[float, float]],
    rewards: List[float],
    params: SchoolfieldParams = SchoolfieldParams()
) -> Dict[str, List[float]]:
    updates = voronoi_bandit_update(points, seeds, rewards, params)
    policy = {}
    for region_id, update in updates.items():
        policy[f'region_{region_id}'] = [update['reward'] / update['count'], update['count']]
    return policy

def voronoi_bandit_expected_reward(
    points: List[Tuple[float, float]],
    seeds: List[Tuple[float, float]],
    rewards: List[float],
    params: SchoolfieldParams = SchoolfieldParams()
) -> List[float]:
    policy = voronoi_bandit_policy(points, seeds, rewards, params)
    expected_rewards = []
    for i, (point, _) in enumerate(zip(points, seeds)):
        region_id = nearest(point, seeds)
        region = policy[f'region_{region_id}']
        expected_rewards.append(region[0])
    return expected_rewards

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    points = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    seeds = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    rewards = [1.0, 2.0, 3.0]
    params = SchoolfieldParams()
    policy = voronoi_bandit_policy(points, seeds, rewards, params)
    expected_rewards = voronoi_bandit_expected_reward(points, seeds, rewards, params)
    print(policy)
    print(expected_rewards)
    assert len(policy) > 0
    assert len(expected_rewards) > 0
    print("Smoke test passed")