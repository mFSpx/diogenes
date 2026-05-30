# DARWIN HAMMER — match 1428, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_epistemic_certainty_m1153_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_ternar_koopman_operator_m632_s3.py (gen3)
# born: 2026-05-29T23:36:17Z

"""
Hybrid algorithm combining DARWIN HAMMER bandit with RBF surrogate (parent A)
and Koopman operator / Dynamic Mode Decomposition (parent B).

Mathematical bridge:
- The bandit updates the empirical mean rewards with a confidence-weighted
  RBF surrogate.  We will map the Koopman operator matrix **K** into a
  positive definite matrix **W** using the bandit's empirical reward variance.
  **W** is then used to update the surrogate weights, effectively integrating
  the Koopman operator dynamics into the RBF surrogate.
- We also integrate the Koopman framework's polynomial observable lift into
  the RBF surrogate, allowing the surrogate to predict context-aware rewards
  in a higher dimensional space.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Sequence, Callable, Iterable

import json
import re

# ----------------------------------------------------------------------
# Types and global stores (from parent A)
# ----------------------------------------------------------------------
Vector = Sequence[float]

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

# Global mutable state (mirrors parent A)
_POLICY: Dict[str, List[float]] = {}          # action_id -> [total_reward, count]
_STORE: Dict[str, float] = {}                 # arbitrary storage for auxiliary data
_SURROGATE = None                             # will hold an RBFSurrogate instance
_KOOPMAN_MATRIX = None                          # will hold the Koopman operator matrix

# ----------------------------------------------------------------------
# Koopman Operator (from parent B)
# ----------------------------------------------------------------------
def prepare_audit_snapshots(audit_timestamps: List[float], audit_states: List[Vector], prune_schedule: Callable[[float], float]) -> np.ndarray:
    """Builds the audit time-series and lifts it using the Koopman observable."""
    snapshots = np.array([audit_state for audit_state in audit_states])
    lifted_snapshots = np.zeros((snapshots.shape[0], 6))
    lifted_snapshots[:, 0] = snapshots[:, 0]
    for i in range(snapshots.shape[1]):
        for j in range(i+1, snapshots.shape[1]):
            lifted_snapshots[:, 2 + i + j] = snapshots[:, i] * snapshots[:, j]
    lifted_snapshots[:, 3:] = np.square(snapshots[:, 3:])
    for t, timestamp in enumerate(audit_timestamps):
        lifted_snapshots[t] = np.array([prune_schedule(timestamp)] + lifted_snapshots[t, 1:].tolist())
    return lifted_snapshots

def fit_hybrid_koopman(lifted_snapshots: np.ndarray) -> np.ndarray:
    """Runs Dynamic Mode Decomposition (DMD) on the lifted data."""
    return np.linalg.svd(lifted_snapshots)[-1][:, :3]

def hybrid_forecast(koopman_matrix: np.ndarray, current_state: Vector) -> Vector:
    """Propagates the audit state forward using the learned Koopman matrix."""
    return np.dot(koopman_matrix, current_state)

def update_koopman_matrix(koopman_matrix: np.ndarray, new_snapshot: Vector) -> np.ndarray:
    """Updates the Koopman operator matrix using the new snapshot."""
    return np.vstack((koopman_matrix, np.dot(koopman_matrix, new_snapshot)))

def update_bandit_policy(policy: Dict[str, List[float]], update: BanditUpdate) -> Dict[str, List[float]]:
    """Updates the bandit policy using the confidence-weighted RBF surrogate."""
    action_id = update.action_id
    reward = update.reward
    propensity = update.propensity
    variance = np.var([update.reward for update in policy[action_id]])
    weights = np.exp(-np.square(np.linalg.norm(np.array([1, 1, update.context_id]) - policy[action_id][0:3])))
    weights = weights / np.sum(weights)
    new_expected_reward = np.sum([reward * weight for reward, weight in zip([update.reward] * len(weights), weights)])
    policy[action_id][0] = new_expected_reward
    return policy

def update_surrogate_weights(surrogate_weights: np.ndarray, koopman_matrix: np.ndarray, new_snapshot: Vector) -> np.ndarray:
    """Updates the surrogate weights using the Koopman operator matrix."""
    return np.dot(koopman_matrix, surrogate_weights)

def predict_reward(surrogate_weights: np.ndarray, current_state: Vector) -> float:
    """Predicts the context-aware reward using the confidence-weighted RBF surrogate."""
    return np.sum([surrogate_weight * np.exp(-np.square(np.linalg.norm(current_state - policy[0:3]))) for surrogate_weight in surrogate_weights])

def main():
    # Smoke test
    policy = {}
    policy['action_1'] = [0.0, 0.0, 0.0, 0.0]
    policy['action_2'] = [0.0, 0.0, 0.0, 0.0]
    update = BanditUpdate('action_1', 1.0, 0.5)
    policy = update_bandit_policy(policy, update)
    # Create an RBF surrogate
    _SURROGATE = np.zeros((4,))
    # Lift some snapshots
    timestamps = [1.0, 2.0, 3.0]
    states = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    lifted_snapshots = prepare_audit_snapshots(timestamps, states, lambda t: 1.0 - t)
    # Fit the Koopman operator
    koopman_matrix = fit_hybrid_koopman(lifted_snapshots)
    # Update the Koopman matrix
    new_snapshot = np.array([13.0, 14.0, 15.0, 16.0])
    koopman_matrix = update_koopman_matrix(koopman_matrix, new_snapshot)
    # Update the bandit policy
    policy = update_bandit_policy(policy, update)
    # Update the surrogate weights
    surrogate_weights = update_surrogate_weights(koopman_matrix, _SURROGATE, new_snapshot)
    # Predict a reward
    current_state = np.array([17.0, 18.0, 19.0, 20.0])
    reward = predict_reward(surrogate_weights, current_state)
    print(reward)

if __name__ == "__main__":
    main()