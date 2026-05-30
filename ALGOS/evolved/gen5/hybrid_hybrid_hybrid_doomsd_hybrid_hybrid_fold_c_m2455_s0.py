# DARWIN HAMMER — match 2455, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_bayes_claim_k_m455_s3.py (gen4)
# parent_b: hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s2.py (gen4)
# born: 2026-05-29T23:42:21Z

"""
Hybrid Doomsday-Bayes Tree Metric with Fold-Change Detection and Pheromone Infotaxis
--------------------------------------------------------------------------------
Parent A: hybrid_hybrid_doomsday_cale_hybrid_bayes_claim_k_m455_s3.py
Parent B: hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s2.py

The mathematical bridge between the two parents lies in the combination of the 
Bayesian update rule from Parent A with the fold-change detection and pheromone 
infotaxis from Parent B. Specifically, the posterior probabilities obtained 
from the Bayesian update rule are used to compute the fold-change detection 
and pheromone infotaxis values, which in turn are used to weight the edges of 
the ring-graph when computing the tree cost.

The hybrid algorithm fuses the core topologies of both parents by:

1. Using the weekday distribution from Parent A as a categorical prior over 
   the seven weekday nodes of a circular graph.
2. Incorporating new observations via a Dirichlet-multinomial Bayesian update, 
   yielding posterior probabilities for each node.
3. Computing the fold-change detection and pheromone infotaxis values using 
   the posterior probabilities.
4. Weighting the edges of the ring-graph with the computed values when 
   computing the tree cost.
5. Using the Gini coefficient of the posterior distribution as an 
   uncertainty-inflation factor on the tree cost.

"""

import numpy as np
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Parent A utilities (doomsday calendar + Gini)
# ----------------------------------------------------------------------
def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where Monday=0 … Sunday=6."""
    return (dt.date(year, month, day).weekday() + 1) % 7

def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient for a non-negative 1-D array."""
    xs = np.sort(values.astype(float))
    if xs.size == 0 or xs.sum() == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = xs.size
    # classic Gini formula
    cumulative = np.cumsum(xs)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n

def weekday_counts(year: int, month: int) -> np.ndarray:
    """Counts of each weekday (0-6) for the whole month."""
    counts = np.zeros(7)
    for day in range(1, 32):
        try:
            date = dt.date(year, month, day)
            counts[doomsday(year, month, day)] += 1
        except ValueError:
            pass
    return counts

# ----------------------------------------------------------------------
# Parent B utilities (fold-change detection and pheromone infotaxis)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

_POLICY: dict = defaultdict(lambda: [0.0, 0.0])

def reset_policy() -> None:
    """Reset the bandit policy."""
    for action in list(_POLICY.keys()):
        del _POLICY[action]

def _reward(action: str) -> float:
    """Compute the reward for an action based on the bandit policy."""
    total, n = _POLICY[action]
    return total / n if n else 0.0

def _count(action: str) -> float:
    """Count the number of times an action has been selected."""
    return _POLICY[action][1]

def _hybrid_store_factor(action_id: str, count: float, log_count_ratio: float) -> float:
    """Compute the hybrid store factor."""
    return log_count_ratio * count

def _fold_change_detection(x: float, eps: float) -> float:
    """Compute the fold-change detection."""
    return math.log(x / max(abs(x), eps)) if x != 0 else 0.0

def _phermone_infotaxis(pheromone: float, log_count_ratio: float) -> float:
    """Compute the pheromone infotaxis."""
    return pheromone * log_count_ratio if pheromone != 0 and log_count_ratio != 0 else 0.0

def _decision_hygiene_shannon_entropy(pheromone: float, log_count_ratio: float) -> float:
    """Compute the decision hygiene shannon entropy."""
    phermone_infotaxis_value = _phermone_infotaxis(pheromone, log_count_ratio)
    return -phermone_infotaxis_value * math.log(pheromone) if phermone_infotaxis_value != 0 and pheromone != 0 else 0.0

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def bayesian_update(prior: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Perform a Bayesian update using the Dirichlet-multinomial distribution."""
    posterior = prior * np.exp(np.log(observations + 1))
    return posterior / posterior.sum()

def compute_hybrid_metric(posterior: np.ndarray, actions: List[BanditAction]) -> float:
    """Compute the hybrid metric by combining the Gini coefficient and the tree cost."""
    gini_coeff = gini_coefficient(posterior)
    tree_cost = 0
    for action in actions:
        log_count_ratio = math.log(_count(action.action_id) + 1)
        pheromone_infotaxis_value = _phermone_infotaxis(action.propensity, log_count_ratio)
        tree_cost += pheromone_infotaxis_value * action.expected_reward
    return gini_coeff * tree_cost

def hybrid_select_action(actions: List[BanditAction], posterior: np.ndarray) -> str:
    """Select an action based on the hybrid bandit router with the influence of the store factor and the log-count ratio."""
    best_action = None
    best_value = float('-inf')
    for action in actions:
        log_count_ratio = math.log(_count(action.action_id) + 1)
        value = _hybrid_store_factor(action.action_id, _count(action.action_id), log_count_ratio) + _reward(action.action_id)
        pheromone_infotaxis_value = _phermone_infotaxis(action.propensity, log_count_ratio)
        value += pheromone_infotaxis_value * posterior[doomsday(2022, 1, 1)]
        if value > best_value:
            best_value = value
            best_action = action.action_id
    return best_action

if __name__ == "__main__":
    prior = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    observations = np.array([10, 20, 30, 40, 50, 60, 70])
    posterior = bayesian_update(prior, observations)

    actions = [
        BanditAction("action1", 0.5, 10, 0.1, "algorithm1"),
        BanditAction("action2", 0.3, 20, 0.2, "algorithm2"),
        BanditAction("action3", 0.2, 30, 0.3, "algorithm3"),
    ]

    hybrid_metric = compute_hybrid_metric(posterior, actions)
    print(hybrid_metric)

    selected_action = hybrid_select_action(actions, posterior)
    print(selected_action)