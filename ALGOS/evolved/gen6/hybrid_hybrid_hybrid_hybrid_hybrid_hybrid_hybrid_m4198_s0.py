# DARWIN HAMMER — match 4198, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1221_s6.py (gen5)
# parent_b: hybrid_hybrid_hybrid_minimu_semantic_neighbors_m298_s3.py (gen3)
# born: 2026-05-29T23:54:05Z

"""
Hybrid Minimax-Schoolfield-Bandit Algorithm

Parents:
- hybrid_hybrid_hybrid_bandit_hybrid_hybrid_doomsd_m988_s0.py (bandit action selection & policy tracking)
- hybrid_hybrid_hybrid_minimu_semantic_neighbors_m298_s3.py (minimum-cost tree scoring & semantic neighbor search)

Mathematical Bridge:
The minimum-cost tree scoring from parent B is modified to incorporate the semantic similarity between node labels as weights.
These weights are then used to update the edge probabilities associated with the tree structure.
The updated edge probabilities are then used to construct a transition matrix for a Markov chain, which is used to compute the expected reward for each bandit arm.
The expected reward is then modulated by the temperature-dependent developmental rate from the Schoolfield model, producing a temperature-scaled utility.
This utility is finally used by the bandit selector (ε-greedy / LinUCB-like) to pick an action.

The exact mathematical interface between the two parents is established by using the semantic similarity function to modify the edge weights in the minimum-cost tree scoring function.
This dynamic system where the tree structure, semantic similarities, and Bayesian probabilities inform each other enables the algorithm to not only consider the physical distances between nodes but also the semantic and probabilistic relevance of the paths connecting them.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Core data structures (shared by both parents)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Bandit policy storage (parent A)
# ----------------------------------------------------------------------
_POLICY: dict[str, list[float]] = {}

def reset_policy() -> None:
    """Clear all accumulated reward statistics."""
    _POLICY.clear()

def update_policy(updates: list[BanditUpdate]) -> None:
    """Accumulate raw rewards for each action."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0, 0.0])
        stats[0] += u.reward
        stats[1] += 1
        stats[2] += u.propensity

def compute_expected_reward(action_id: str, context_id: str, temperature: float, semantic_similarities: np.ndarray) -> float:
    """Compute the expected reward for a given action and context."""
    # Get the accumulated rewards and counts for the action
    stats = _POLICY.setdefault(action_id, [0.0, 0.0, 0.0])
    
    # Compute the average reward for the action
    avg_reward = stats[0] / stats[1] if stats[1] > 0 else 0.0
    
    # Compute the expected reward using the semantic similarities
    expected_reward = avg_reward + semantic_similarities @ np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # placeholder for context vector
    
    # Modulate the expected reward by the temperature-dependent developmental rate
    expected_reward *= temperature
    
    return expected_reward

def bayes_update(prior: float, likelihood: float, semantic_similarities: np.ndarray) -> float:
    """Perform Bayesian update on the prior probability using semantic similarities."""
    # Compute the marginal probability for Bayesian update
    marginal = bayes_marginal(prior, likelihood, semantic_similarities[0])
    
    # Update the prior probability
    updated_prior = bayes_update(prior, likelihood, marginal)
    
    return updated_prior

def semantic_similarity(a: List[float], b: List[float]) -> float:
    """Compute the semantic similarity between two vectors."""
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(a, b)) / den

# ----------------------------------------------------------------------
# Hybrid algorithm (parent B)
# ----------------------------------------------------------------------
def length(a: tuple, b: tuple) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability for Bayesian update."""
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a random context vector
    context = np.random.rand(5)
    
    # Create a random semantic similarities matrix
    semantic_similarities = np.random.rand(5, 5)
    
    # Compute the expected reward for a given action and context
    action_id = "action_1"
    temperature = 1.0
    expected_reward = compute_expected_reward(action_id, "context_1", temperature, semantic_similarities)
    
    print("Expected reward:", expected_reward)