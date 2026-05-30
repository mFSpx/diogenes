# DARWIN HAMMER — match 3519, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_xgboost_objec_hybrid_hybrid_hybrid_m1262_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hard_truth_ma_m2396_s2.py (gen3)
# born: 2026-05-29T23:50:30Z

"""Hybrid Algorithm: XGBoost‑Ternary Lens + Endpoint‑Circuit Morphology Fusion

Parents:
- hybrid_hybrid_xgboost_objec_hybrid_hybrid_hybrid_m1262_s0.py (XGBoost loss, gradient/hessian, split gain)
- hybrid_hybrid_hybrid_endpoi_hybrid_hard_truth_ma_m2396_s2.py (EndpointCircuitBreaker, Morphology curvature, global weight W)

Mathematical Bridge:
Both parents expose a scalar that modulates a core optimisation quantity:
    * XGBoost computes leaf weights and split gains from gradient/hessian statistics.
    * The Endpoint‑Circuit side provides a *global weight*   W = H·C   where
          H ∈ [0,1] is the health factor of the circuit breaker and
          C = sphericity·flatness ∈ [0,1] is a morphology curvature.
We fuse them by scaling every XGBoost‑derived quantity with W.  The resulting
leaf weight, split gain and model‑selection score are therefore jointly
optimised with respect to learning dynamics (gradients) *and* system health /
morphology constraints.  The circuit breaker can also veto a split when it
is open, enforcing a hard safety guard on the tree construction process.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A components (XGBoost core)
# ----------------------------------------------------------------------


def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-margin))


def binary_logistic_grad_hess(
    y_true: np.ndarray, margin: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """First and second gradients for binary logistic loss in margin space."""
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h


def optimal_leaf_weight(
    gradient_sum: float, hessian_sum: float, reg_lambda: float = 1.0
) -> float:
    """Optimal leaf weight for XGBoost (unscaled)."""
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
    """XGBoost split gain (unscaled)."""
    gl, hl = float(left_gradient), float(left_hessian)
    gr, hr = float(right_gradient), float(right_hessian)

    parent = (gl + gr) ** 2 / (hl + hr + reg_lambda)
    children = gl ** 2 / (hl + reg_lambda) + gr ** 2 / (hr + reg_lambda)
    gain = 0.5 * (children - parent) - gamma
    return gain


# ----------------------------------------------------------------------
# Parent B components (Endpoint‑Circuit & Morphology)
# ----------------------------------------------------------------------


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def health_factor(self) -> float:
        """Return a health factor H∈[0,1]; 0 if open, otherwise linearly decreasing."""
        if self.open:
            return 0.0
        # Linear mapping: 0 failures → 1.0, failure_threshold‑1 → >0
        return max(0.0, 1.0 - (self.failures / self.failure_threshold))

    def allow(self) -> bool:
        """True if the breaker is closed (i.e., operation allowed)."""
        return not self.open


@dataclass
class Morphology:
    """Simple morphology descriptor."""

    sphericity: float  # ∈[0,1]
    flatness: float    # ∈[0,1]

    def __post_init__(self):
        self.sphericity = max(0.0, min(1.0, float(self.sphericity)))
        self.flatness = max(0.0, min(1.0, float(self.flatness)))

    def curvature(self) -> float:
        """Dimension‑less curvature C = sphericity × flatness."""
        return self.sphericity * self.flatness


# ----------------------------------------------------------------------
# Hybrid operations (bridge)
# ----------------------------------------------------------------------


def global_weight(breaker: EndpointCircuitBreaker, morph: Morphology) -> float:
    """
    Compute the global weight W = H·C that scales XGBoost quantities.
    H is the health factor from the circuit breaker, C is curvature from morphology.
    """
    H = breaker.health_factor()
    C = morph.curvature()
    return H * C


def hybrid_leaf_weight(
    gradient_sum: float,
    hessian_sum: float,
    breaker: EndpointCircuitBreaker,
    morph: Morphology,
    reg_lambda: float = 1.0,
) -> float:
    """
    Scaled leaf weight:  w_hybrid = W · w_opt,
    where w_opt is the standard XGBoost optimal leaf weight.
    """
    w_opt = optimal_leaf_weight(gradient_sum, hessian_sum, reg_lambda)
    W = global_weight(breaker, morph)
    return W * w_opt


def hybrid_split_gain(
    left_gradient: float,
    left_hessian: float,
    right_gradient: float,
    right_hessian: float,
    breaker: EndpointCircuitBreaker,
    morph: Morphology,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
) -> float:
    """
    Compute a split gain that is first evaluated by XGBoost and then
    multiplied by the global weight W.  If the circuit breaker is open,
    the gain is forced to -inf to guarantee the split is rejected.
    """
    if not breaker.allow():
        return -math.inf  # veto by circuit breaker

    base_gain = split_gain(
        left_gradient,
        left_hessian,
        right_gradient,
        right_hessian,
        reg_lambda=reg_lambda,
        gamma=gamma,
    )
    W = global_weight(breaker, morph)
    return W * base_gain


def hybrid_model_score(
    feature_vector: np.ndarray,
    ram_m: float,
    R_max: float,
    breaker: EndpointCircuitBreaker,
    morph: Morphology,
) -> float:
    """
    Hybrid score for a candidate model:
        S_m = W · ||f||₂ · (ram_m / R_max)⁻¹
    If the circuit breaker is open, the score is zero (model cannot be loaded).
    """
    if not breaker.allow():
        return 0.0
    norm_f = np.linalg.norm(feature_vector, ord=2)
    ram_factor = (ram_m / R_max) if R_max != 0 else 0.0
    if ram_factor == 0:
        return 0.0
    W = global_weight(breaker, morph)
    return W * norm_f / ram_factor


# ----------------------------------------------------------------------
# Demonstration utilities
# ----------------------------------------------------------------------


def _dummy_gradients(num_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic gradients and hessians for demo purposes."""
    y = np.random.randint(0, 2, size=num_samples)
    margin = np.random.randn(num_samples)
    g, h = binary_logistic_grad_hess(y, margin)
    return g, h


def demo_hybrid_operations() -> None:
    """Run a quick demonstration of the hybrid functions."""
    # Initialise components
    breaker = EndpointCircuitBreaker(failure_threshold=3)
    morph = Morphology(sphericity=0.8, flatness=0.6)

    # Simulate a few failures to lower health factor
    for _ in range(2):
        breaker.record_failure()

    # Generate dummy gradient/hessian statistics
    grads, hess = _dummy_gradients(20)
    grad_sum = float(np.sum(grads))
    hess_sum = float(np.sum(hess))

    # Hybrid leaf weight
    leaf_w = hybrid_leaf_weight(grad_sum, hess_sum, breaker, morph, reg_lambda=1.0)
    print(f"Hybrid leaf weight: {leaf_w:.4f}")

    # Hybrid split gain (using arbitrary left/right aggregates)
    gain = hybrid_split_gain(
        left_gradient=grad_sum * 0.4,
        left_hessian=hess_sum * 0.4,
        right_gradient=grad_sum * 0.6,
        right_hessian=hess_sum * 0.6,
        breaker=breaker,
        morph=morph,
        reg_lambda=1.0,
        gamma=0.1,
    )
    print(f"Hybrid split gain: {gain:.4f}")

    # Model scoring
    f_vec = np.random.randn(5)
    ram_used = 2.5  # GB
    ram_cap = 8.0   # GB
    score = hybrid_model_score(f_vec, ram_used, ram_cap, breaker, morph)
    print(f"Hybrid model score: {score:.4f}")

    # Show health factor and curvature for reference
    print(f"Health factor H: {breaker.health_factor():.3f}")
    print(f"Curvature C: {morph.curvature():.3f}")
    print(f"Global weight W: {global_weight(breaker, morph):.3f}")


if __name__ == "__main__":
    demo_hybrid_operations()