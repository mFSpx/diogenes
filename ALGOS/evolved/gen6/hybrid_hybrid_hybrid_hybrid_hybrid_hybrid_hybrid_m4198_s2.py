# DARWIN HAMMER — match 4198, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1221_s6.py (gen5)
# parent_b: hybrid_hybrid_hybrid_minimu_semantic_neighbors_m298_s3.py (gen3)
# born: 2026-05-29T23:54:05Z

"""
Hybrid Algorithm: Fusing Hybrid Bandit-Schoolfield-Geometric and Minimum-Cost Semantic Neighbor Systems

This module integrates the Hybrid Bandit-Schoolfield-Geometric Algorithm 
(parent: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1221_s6.py) 
and the Minimum-Cost Semantic Neighbor System 
(parent: hybrid_hybrid_hybrid_minimu_semantic_neighbors_m298_s3.py).

The mathematical bridge between these two systems is established by 
utilizing the semantic similarity function from the semantic neighbor 
system as a modulator for the bandit arm's expected reward. Specifically, 
the temperature-dependent developmental rate `ρ(T)` from the Schoolfield 
model is used to scale the semantic similarity, which in turn affects 
the bandit arm selection.

The core idea is to fuse the geometric projection, thermodynamic scaling, 
and bandit policy with the semantic similarity and Bayesian update rules, 
enabling the algorithm to consider both physical distances and semantic 
relevance of the paths connecting nodes.

Imports:
- numpy for matrix operations
- standard library for basic functions
- math for mathematical operations
- random for random number generation
- sys for system-specific functions
- pathlib for path manipulation
"""

import math
import numpy as np
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List

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
class SchoolfieldParams:
    rho_25: float = 1.0               # rate at 25 °C (298.15 K)
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987

def semantic_similarity(a: List[float], b: List[float]) -> float:
    """Compute the semantic similarity between two vectors."""
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(a, b)) / den

def schoolfield_rate(params: SchoolfieldParams, temperature: float) -> float:
    """Compute the temperature-dependent developmental rate ρ(T)"""
    if temperature < params.t_low:
        return params.rho_25 * math.exp((params.delta_h_low / params.r_cal) * (1/params.t_low - 1/temperature))
    elif temperature > params.t_high:
        return params.rho_25 * math.exp((params.delta_h_high / params.r_cal) * (1/params.t_high - 1/temperature))
    else:
        return params.rho_25 * math.exp((params.delta_h_activation / params.r_cal) * (1/298.15 - 1/temperature))

def hybrid_bandit_semantic_neighbor(context: np.ndarray, 
                                    W: np.ndarray, 
                                    action: BanditAction, 
                                    semantic_vector: List[float], 
                                    temperature: float, 
                                    params: SchoolfieldParams) -> float:
    """Compute the hybrid utility of a bandit arm"""
    # Geometric projection
    score = np.dot(context, W[:, int(action.action_id)])
    
    # Semantic similarity modulation
    similarity = semantic_similarity(context.tolist(), semantic_vector)
    
    # Temperature-dependent developmental rate
    rate = schoolfield_rate(params, temperature)
    
    # Hybrid utility
    utility = rate * (action.expected_reward + score) * similarity
    
    return utility

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Perform Bayesian update on the prior probability."""
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def update_policy(updates: list[BanditUpdate], 
                  context: np.ndarray, 
                  W: np.ndarray, 
                  semantic_vectors: dict[str, List[float]], 
                  temperature: float, 
                  params: SchoolfieldParams) -> None:
    """Accumulate raw rewards for each action and update policy"""
    for u in updates:
        action = BanditAction(u.action_id, u.propensity, 0, 0, "hybrid")
        semantic_vector = semantic_vectors[u.action_id]
        utility = hybrid_bandit_semantic_neighbor(context, W, action, semantic_vector, temperature, params)
        
        # Bayesian update
        prior = 0.5
        likelihood = utility
        marginal = bayes_update(prior, likelihood, 1)
        updated_utility = bayes_update(prior, likelihood, marginal)
        
        # Update policy
        print(f"Updated utility for action {u.action_id}: {updated_utility}")

if __name__ == "__main__":
    # Smoke test
    context = np.array([1, 2, 3])
    W = np.array([[1, 2], [3, 4], [5, 6]])
    action = BanditAction("0", 0.5, 10, 0.1, "hybrid")
    semantic_vector = [1, 2, 3]
    temperature = 298.15
    params = SchoolfieldParams()
    
    utility = hybrid_bandit_semantic_neighbor(context, W, action, semantic_vector, temperature, params)
    print(f"Hybrid utility: {utility}")
    
    updates = [BanditUpdate("context1", "0", 10, 0.5)]
    semantic_vectors = {"0": [1, 2, 3]}
    update_policy(updates, context, W, semantic_vectors, temperature, params)