# DARWIN HAMMER — match 4708, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2169_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m717_s2.py (gen4)
# born: 2026-05-29T23:57:43Z

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

def hybrid_score_lens(L: np.ndarray, F: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the hybrid score lens score.

    Parameters
    ----------
    L : np.ndarray
        Ternary lens vector of shape (3,)
    F : np.ndarray
        Fusion matrix of shape (3, 9)
    c : np.ndarray
        Regex count vector of shape (9,)

    Returns
    -------
    float
        Hybrid score lens score
    """
    return np.dot(L, np.dot(F, c))


def tropical_score(S: np.ndarray, r: float, action: str) -> float:
    """
    Compute the tropical max-plus score for a given action.

    Parameters
    ----------
    S : np.ndarray
        Count-Min sketch matrix
    r : float
        Bandit reward estimate
    action : str
        Action ID

    Returns
    -------
    float
        Tropical max-plus score
    """
    h = hashlib.sha256(action.encode()).hexdigest()
    idx = int(h, 16) % S.shape[1]
    return np.max(S[:, idx]) + r


def select_hybrid_action(S: np.ndarray, r: float, L: np.ndarray, F: np.ndarray, c: np.ndarray,
                         epsilon: float = 0.1, lambda_val: float = 0.1, alpha: float = 0.1) -> str:
    """
    Choose an action using the hybrid selection criterion.

    Parameters
    ----------
    S : np.ndarray
        Count-Min sketch matrix
    r : float
        Bandit reward estimate
    L : np.ndarray
        Ternary lens vector
    F : np.ndarray
        Fusion matrix
    c : np.ndarray
        Regex count vector
    epsilon : float, optional
        Exploration rate (default: 0.1)
    lambda_val : float, optional
        RLCT estimate (default: 0.1)
    alpha : float, optional
        UCB exploration rate (default: 0.1)

    Returns
    -------
    str
        Action ID
    """
    n = S.shape[0]
    N = np.sum(S, axis=1)
    nhat = np.sum(N > 0)
    ucb = r + np.sqrt(alpha * np.log(n) / N)
    rlct = lambda_val * np.log(nhat)
    score = ucb + rlct + L * hybrid_score_lens(L, F, c)
    scores = np.copy(score)
    scores[N == 0] = -np.inf
    best_idx = np.argmax(scores)
    if np.random.rand() < epsilon:
        return np.random.choice(len(S))
    else:
        return str(best_idx)


def reset_policy() -> None:
    """Clear all stored rewards and counts."""
    global _POLICY
    _POLICY = defaultdict(lambda: {'mean': 0, 'count': 0})


def update_policy(updates: List[Tuple[str, float]]) -> None:
    """
    Update the reward statistics for actions.

    Parameters
    ----------
    updates : List[Tuple[str, float]]
        Each tuple contains (action_id, reward_observed).
    """
    global _POLICY
    for action_id, reward in updates:
        _POLICY[action_id]['mean'] = (_POLICY[action_id]['mean'] * _POLICY[action_id]['count'] + reward) / (_POLICY[action_id]['count'] + 1)
        _POLICY[action_id]['count'] += 1


if __name__ == "__main__":
    # Initialize sketch matrix and fusion matrix
    width = 2000
    depth = 5
    S = np.zeros((depth, width), dtype=np.int64)
    F = np.random.rand(3, 9)

    # Initialize parameters
    r = 0.5
    L = np.array([1, 0, 0])
    c = np.random.rand(9)

    # Test the hybrid score function
    score = hybrid_score_lens(L, F, c)
    print("Hybrid score:", score)

    # Test the tropical score function
    action = "test_action"
    tropical_score = tropical_score(S, r, action)
    print("Tropical score:", tropical_score)

    # Test the hybrid action selection function
    epsilon = 0.1
    lambda_val = 0.1
    alpha = 0.1
    action_id = select_hybrid_action(S, r, L, F, c, epsilon, lambda_val, alpha)
    print("Selected action:", action_id)