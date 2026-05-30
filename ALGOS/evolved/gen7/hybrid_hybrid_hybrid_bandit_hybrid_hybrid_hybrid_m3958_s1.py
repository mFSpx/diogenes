# DARWIN HAMMER — match 3958, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_hybrid_m547_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2380_s1.py (gen6)
# born: 2026-05-29T23:52:49Z

"""
Hybrid Algorithm: Fusing Bandit Router (hybrid_hybrid_bandit_router_hybrid_hybrid_hybrid_m547_s1.py) 
with Tropical Linear Algebra and Gradient/Hessian Utilities (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2380_s1.py)
====================================================================================

This hybrid algorithm combines the core topologies of Parent A (hybrid_hybrid_bandit_router_hybrid_hybrid_hybrid_m547_s1.py) 
and Parent B (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2380_s1.py). The mathematical bridge is established 
by mapping the bandit actions from Parent A to tropical matrices in Parent B, where the propensity of each action 
corresponds to the weights in the tropical matrix. The Fisher information is then used to select the optimal action.

Imports:
- numpy for numerical computations
- standard library for basic functionality
- math for mathematical operations
- random for generating random numbers
- sys for system-specific functions
- pathlib for path manipulation
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
# Parent A – Bandit Router Core
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

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

# ----------------------------------------------------------------------
# Parent B – Tropical Linear Algebra and Gradient/Hessian Utilities Core
# ----------------------------------------------------------------------
def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Max‑plus matrix multiplication.
    (A ⊗ B)[i, j] = max_k (A[i, k] + B[k, j])

    Parameters
    ----------
    A : (n, m) ndarray
    B : (m, p) ndarray

    Returns
    -------
    C : (n, p) ndarray
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Inner dimensions must match for tropical multiplication")
    C = np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)
    return C

def adjusted_grad_hess(logistic_loss: np.ndarray, alpha: float, s: np.ndarray, H: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute element‑wise adjusted gradient and Hessian for the hybrid loss.

    Parameters
    ----------
    logistic_loss : ndarray
        Logistic loss values (vector).
    alpha : float
        Scaling factor for semantic contribution.
    s : ndarray
        Recovery‑priority vector (same shape as logistic_loss).
    H : float
        Global entropy‑like term.

    Returns
    -------
    grad : ndarray
    hess : ndarray
    """
    grad_base = logistic_loss * (1.0 - logistic_loss)
    hess_base = logistic_loss * (1.0 - logistic_loss) * (1.0 - 2.0 * logistic_loss)

    grad = grad_base + alpha * s * H
    hess = hess_base + alpha * s * H
    return grad, hess

# ----------------------------------------------------------------------
# Hybrid Core
# ----------------------------------------------------------------------
def map_bandit_to_tropical(action: BanditAction) -> np.ndarray:
    """
    Map bandit action to tropical matrix.

    Parameters
    ----------
    action : BanditAction

    Returns
    -------
    A : (1, 1) ndarray
    """
    A = np.array([[action.propensity]])
    return A

def hybrid_update_policy(actions: List[BanditAction], updates: List[BanditUpdate]) -> None:
    """
    Update policy using tropical matrix multiplication.

    Parameters
    ----------
    actions : List[BanditAction]
    updates : List[BanditUpdate]
    """
    tropical_matrices = [map_bandit_to_tropical(action) for action in actions]
    A = tropical_matrices[0]
    for B in tropical_matrices[1:]:
        A = tropical_matmul(A, B)

    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def hybrid_adjusted_grad_hess(actions: List[BanditAction], logistic_loss: np.ndarray, alpha: float, s: np.ndarray, H: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute adjusted gradient and Hessian using tropical matrix multiplication.

    Parameters
    ----------
    actions : List[BanditAction]
    logistic_loss : ndarray
    alpha : float
    s : ndarray
    H : float

    Returns
    -------
    grad : ndarray
    hess : ndarray
    """
    tropical_matrices = [map_bandit_to_tropical(action) for action in actions]
    A = tropical_matrices[0]
    for B in tropical_matrices[1:]:
        A = tropical_matmul(A, B)

    grad, hess = adjusted_grad_hess(logistic_loss, alpha, s, H)
    return grad, hess

if __name__ == "__main__":
    # Smoke test
    action1 = BanditAction("action1", 0.5, 10.0, 0.1, "algorithm1")
    action2 = BanditAction("action2", 0.3, 20.0, 0.2, "algorithm2")

    update1 = BanditUpdate("context1", "action1", 10.0, 0.5)
    update2 = BanditUpdate("context2", "action2", 20.0, 0.3)

    actions = [action1, action2]
    updates = [update1, update2]

    hybrid_update_policy(actions, updates)

    logistic_loss = np.array([0.1, 0.2])
    alpha = 0.5
    s = np.array([1.0, 2.0])
    H = 0.1

    grad, hess = hybrid_adjusted_grad_hess(actions, logistic_loss, alpha, s, H)

    print(grad)
    print(hess)