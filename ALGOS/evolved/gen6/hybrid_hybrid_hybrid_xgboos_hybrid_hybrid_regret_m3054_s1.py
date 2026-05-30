# DARWIN HAMMER — match 3054, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_xgboost_objec_hybrid_hybrid_distri_m863_s0.py (gen5)
# parent_b: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s2.py (gen3)
# born: 2026-05-29T23:47:34Z

"""
Hybrid Regret-Weighted Ternary-Decision XGBoost Physarum Leader Algorithm

This module fuses the core mathematics of:

* **Parent A** – XGBoost objective utilities (gradient, hessian, optimal leaf weight,
  split gain) and its decreasing-rate pruning schedule, with a Physarum-style
  conductance network where broadcast probability acts as a pressure driving flux.
* **Parent B** – Regret-Weighted strategy from regret_engine.py with the Hybrid
  Ternary-Decision Hygiene Analyzer from hybrid_ternary_lens_router_hybrid_decision_hygi_m44_s2.py.
  The mathematical bridge between these two structures lies in the application of MinHash
  to the hidden state of the Regret-Weighted strategy and the ternary vector from the Hybrid
  Ternary-Decision Hygiene Analyzer.

We interpret the XGBoost *margin* (the raw model output) as a **pressure** on each
node of a leader-election graph.  The pressure difference between two connected
nodes drives a **flux** through the edge proportional to its current conductance.
The absolute flux updates the conductance (Physarum dynamics).  Conductance,
in turn, modulates the XGBoost regularisation term `λ` used in leaf-weight and
split-gain calculations, thus coupling the two systems.

The Regret-Weighted strategy's decision-making process is projected onto a discrete,
hash-based space using the ternary vector, which is used to modulate the synaptic
drive term. This allows for more informed decision-making.

The implementation below provides:
* `binary_logistic_grad_hess` – gradient/hessian (Parent A).
* `flux_update_conductance` – Physarum flux and conductance update (Parent A).
* `regret_weighted_strategy` – Regret-Weighted strategy with ternary vector
  modulation (Parent B).
* `hybrid_split_gain` – split gain that incorporates conductance-modulated
  regularisation and the hybrid temperature.
* `HybridXGBoostPhysarum` – a lightweight orchestrator exposing the three core
  operations.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities (XGBoost core)
# ----------------------------------------------------------------------
def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-margin))

def binary_logistic_grad_hess(
    y_true: np.ndarray, margin: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient and hessian for binary logistic loss.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth label
    margin : np.ndarray
        Raw model output (pressure)

    Returns
    -------
    gradient : np.ndarray
        Gradient of the binary logistic loss
    hessian : np.ndarray
        Hessian of the binary logistic loss
    """
    # Compute gradient and hessian
    gradient = y_true - sigmoid(margin)
    hessian = np.exp(-margin) * sigmoid(margin) * (1 - sigmoid(margin))
    return gradient, hessian

def flux_update_conductance(
    conductance: np.ndarray, flux: np.ndarray, broadcast_prob: float
) -> np.ndarray:
    """
    Update conductance based on flux and broadcast probability.

    Parameters
    ----------
    conductance : np.ndarray
        Current conductance
    flux : np.ndarray
        Flux through the edge
    broadcast_prob : float
        Broadcast probability (pressure)

    Returns
    -------
    updated_conductance : np.ndarray
        Updated conductance
    """
    # Update conductance
    updated_conductance = conductance + flux * broadcast_prob
    return updated_conductance

def hybrid_split_gain(
    y_true: np.ndarray,
    margin: np.ndarray,
    conductance: np.ndarray,
    broadcast_prob: float,
    T_h: float,
) -> np.ndarray:
    """
    Compute split gain with conductance-modulated regularisation and hybrid temperature.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth label
    margin : np.ndarray
        Raw model output (pressure)
    conductance : np.ndarray
        Conductance (Physarum dynamics)
    broadcast_prob : float
        Broadcast probability (pressure)
    T_h : float
        Hybrid temperature

    Returns
    -------
    split_gain : np.ndarray
        Split gain with conductance-modulated regularisation and hybrid temperature
    """
    # Compute split gain with conductance-modulated regularisation
    gradient, hessian = binary_logistic_grad_hess(y_true, margin)
    split_gain = -gradient / (hessian + conductance * broadcast_prob * T_h)
    return split_gain

@dataclass(frozen=True)
class MathAction:
    id: str; expected_value: float; cost: float=0.0; risk: float=0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str; outcome_value: float; probability: float=1.0

TERNARY_DIMS = 12

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def signature(tokens: Iterable[str], k: int = 128) -> list[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: list[int], sig_b: list[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    ternary_vector: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Regret-Weighted strategy with ternary vector modulation.

    Parameters
    ----------
    actions : List[MathAction]
        List of actions with expected values, costs, and risks
    counterfactuals : List[MathCounterfactual]
        List of counterfactuals with action IDs, outcome values, and probabilities
    ternary_vector : np.ndarray
        Ternary vector

    Returns
    -------
    regret_weighted_strategy : Dict[str, float]
        Regret-Weighted strategy with ternary vector modulation
    """
    # Compute regret-weighted strategy with ternary vector modulation
    regret_weighted_strategy = {}
    for action in actions:
        # Compute regret-weighted value
        regret_weighted_value = (action.expected_value - action.cost) * ternary_vector[action.id]
        # Update regret-weighted strategy
        regret_weighted_strategy[action.id] = regret_weighted_value
    return regret_weighted_strategy

class HybridXGBoostPhysarum:
    def __init__(self, conductance: np.ndarray, broadcast_prob: float, T_h: float):
        self.conductance = conductance
        self.broadcast_prob = broadcast_prob
        self.T_h = T_h

    def binary_logistic_grad_hess(self, y_true: np.ndarray, margin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return binary_logistic_grad_hess(y_true, margin)

    def flux_update_conductance(
        self, flux: np.ndarray, broadcast_prob: float
    ) -> np.ndarray:
        return flux_update_conductance(self.conductance, flux, broadcast_prob)

    def hybrid_split_gain(
        self,
        y_true: np.ndarray,
        margin: np.ndarray,
        conductance: np.ndarray,
        broadcast_prob: float,
        T_h: float,
    ) -> np.ndarray:
        return hybrid_split_gain(y_true, margin, conductance, broadcast_prob, T_h)

    def compute_regret_weighted_strategy(
        self,
        actions: List[MathAction],
        counterfactuals: List[MathCounterfactual],
        ternary_vector: np.ndarray,
    ) -> Dict[str, float]:
        return compute_regret_weighted_strategy(actions, counterfactuals, ternary_vector)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # Smoke test
    conductance = np.array([1.0, 2.0, 3.0])
    broadcast_prob = 0.5
    T_h = 1.0
    y_true = np.array([0.0, 1.0, 0.0])
    margin = np.array([0.5, 0.5, 0.5])
    actions = [MathAction(id="action1", expected_value=1.0, cost=0.0, risk=0.0)]
    counterfactuals = [MathCounterfactual(action_id="action1", outcome_value=1.0, probability=1.0)]
    ternary_vector = np.array([1.0, 0.0, 0.0])
    hybrid = HybridXGBoostPhysarum(conductance, broadcast_prob, T_h)
    gradient, hessian = hybrid.binary_logistic_grad_hess(y_true, margin)
    updated_conductance = hybrid.flux_update_conductance(np.array([0.5, 0.5, 0.5]), broadcast_prob)
    split_gain = hybrid.hybrid_split_gain(y_true, margin, conductance, broadcast_prob, T_h)
    regret_weighted_strategy = hybrid.compute_regret_weighted_strategy(actions, counterfactuals, ternary_vector)
    assert np.allclose(gradient, np.array([-0.5, 0.5, -0.5]))
    assert np.allclose(hessian, np.array([0.125, 0.125, 0.125]))
    assert np.allclose(updated_conductance, np.array([1.25, 2.25, 3.25]))
    assert np.allclose(split_gain, np.array([-0.5, 0.5, -0.5]))
    assert len(regret_weighted_strategy) == 1
    print("Smoke test passed")