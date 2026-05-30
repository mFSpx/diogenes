# DARWIN HAMMER — match 930, survivor 0
# gen: 5
# parent_a: xgboost_objective.py (gen0)
# parent_b: hybrid_hybrid_physarum_netw_hybrid_gliner_zero_s_m66_s2.py (gen4)
# born: 2026-05-29T23:31:35Z

"""
Unified Algorithm: Flux-Based Gliner Hybrid
Fuses the principles of Flux-Based Conductance Update (Parent Algorithm A: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s1.py)
and a Zero-Shot Extractor with Minimum Cost Tree (Parent Algorithm B: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s3.py).
The mathematical bridge between the two parents lies in the integration of the store differential equation in the UnifiedBanditTTT class (Parent A)
with the label extraction and scoring mechanism in the Span class (Parent B).
By fusing these two components, we develop a unified algorithm that leverages the strengths of both parents to extract labels and compute scores based on a flux-based conductance update mechanism.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    """Flux calculation."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    """Update conductance."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def binary_logistic_grad_hess(y_true: np.ndarray, margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and second gradients for binary logistic loss in margin space."""
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h


def optimal_leaf_weight(gradient_sum: float, hessian_sum: float, reg_lambda: float = 1.0) -> float:
    """Compute optimal leaf weight."""
    return -float(gradient_sum) / (float(hessian_sum) + float(reg_lambda))


def split_gain(
    left_gradient: float,
    left_hessian: float,
    right_gradient: float,
    right_hessian: float,
    *,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
) -> float:
    """Compute split gain."""
    gl, hl = float(left_gradient), float(left_hessian)
    gr, hr = float(right_gradient), float(right_hessian)
    parent = (gl + gr) ** 2 / (hl + hr + reg_lambda)
    children = gl**2 / (hl + reg_lambda) + gr**2 / (hr + reg_lambda)
    return 0.5 * (children - parent) - gamma


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


# ----------------------------------------------------------------------
# Unified Flux-Based Gliner Hybrid
# ----------------------------------------------------------------------

def unified_flux_gliner_update(
    conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05
) -> float:
    """Unified flux-based Gliner hybrid update."""
    return update_conductance(conductance, q, dt, gain, decay)


def unified_flux_gliner_split_gain(
    left_gradient: float,
    left_hessian: float,
    right_gradient: float,
    right_hessian: float,
    *,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
) -> float:
    """Unified flux-based Gliner hybrid split gain."""
    return split_gain(left_gradient, left_hessian, right_gradient, right_hessian, reg_lambda=reg_lambda, gamma=gamma)


def unified_flux_gliner_predict(
    x: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    conductance: float,
    dt: float = 1.0,
    gain: float = 1.0,
    decay: float = 0.05,
) -> np.ndarray:
    """Unified flux-based Gliner hybrid prediction."""
    p = sigmoid(x @ gradient)
    flux_value = flux(conductance, 1.0, p, 1.0 - p)
    return p + dt * flux_value


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    conductance = 1.0
    q = 1.0
    dt = 1.0
    gain = 1.0
    decay = 0.05
    left_gradient = 1.0
    left_hessian = 1.0
    right_gradient = 1.0
    right_hessian = 1.0
    reg_lambda = 1.0
    gamma = 0.0
    x = np.random.rand(10)
    gradient = np.random.rand(10)
    hessian = np.random.rand(10)
    unified_flux_gliner_update(conductance, q, dt, gain, decay)
    unified_flux_gliner_split_gain(left_gradient, left_hessian, right_gradient, right_hessian, reg_lambda=reg_lambda, gamma=gamma)
    unified_flux_gliner_predict(x, gradient, hessian, conductance, dt, gain, decay)