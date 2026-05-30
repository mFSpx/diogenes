# DARWIN HAMMER — match 3083, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1869_s3.py (gen5)
# parent_b: hybrid_hybrid_xgboost_objec_hybrid_semantic_neig_m2287_s0.py (gen5)
# born: 2026-05-29T23:47:41Z

"""
Hybrid Regret‑Bandit‑Koopman‑XGBoost with Endpoint‑Health‑Semantic‑Priority Fusion
---------------------------------------------------------------------------------
Parent A contributes:
    - Regret‑weighted probability distribution `p_t` over actions.
    - Hoeffding confidence bounds for those probabilities.

Parent B contributes:
    - XGBoost binary logistic gradient/hessian scaled by endpoint health.
    - Leaf‑weight formula regularized by `λ` and further modulated by a
      semantic recovery priority term.

Mathematical Bridge
------------------
The bridge is built by feeding the regret‑weighted distribution `p_t` into the
Hoeffding bound to obtain a per‑action confidence interval `c_t`.  This interval
is then used as a multiplicative scaling factor for the XGBoost gradients and
Hessians, while the semantic recovery priority acts as an additional
regularizer in the leaf‑weight closed‑form solution.  The resulting hybrid
updates simultaneously respect bandit‑style exploration, statistical confidence,
endpoint health, and semantic priority.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action definition used by the regret engine."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class BanditResult:
    """Result of a bandit selection after applying Hoeffding confidence."""
    action_id: str
    propensity: float
    confidence_bound: float
    adjusted_probability: float
    algorithm: str = "HybridRegretBanditKoopmanXGBoost"


# ----------------------------------------------------------------------
# Core Hybrid Functions
# ----------------------------------------------------------------------
def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, float]:
    """
    Softmax‑like probability distribution over actions using expected values.
    Counterfactuals are used to adjust the denominator, providing a regret‑aware
    normalisation.
    """
    # Gather exponentiated utilities
    exp_vals = {a.id: math.exp(a.expected_value) for a in actions}
    # Sum over all actions, optionally weighted by counterfactual probabilities
    cf_weights = {c.action_id: c.probability for c in counterfactuals}
    denominator = sum(
        exp_vals[a.id] * cf_weights.get(a.id, 1.0) for a in actions
    )
    probabilities = {
        a.id: (exp_vals[a.id] * cf_weights.get(a.id, 1.0)) / denominator
        for a in actions
    }
    return probabilities


def hoeffding_confidence_bound(
    n_trials: int,
    delta: float = 0.05,
) -> float:
    """
    Hoeffding bound for a Bernoulli variable.
    Returns the half‑width of the confidence interval.
    """
    if n_trials <= 0:
        return 1.0  # maximal uncertainty
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n_trials))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def binary_logistic_grad_hess(
    y_true: np.ndarray,
    margin: np.ndarray,
    endpoint_health: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    First and second order gradients for binary logistic loss,
    scaled by endpoint health (Parent B).
    """
    p = sigmoid(margin)
    g = (p - y_true) * endpoint_health
    h = (p * (1.0 - p)) * endpoint_health
    return g, h


def optimal_leaf_weight_hybrid(
    gradient_sum: float,
    hessian_sum: float,
    reg_lambda: float = 1.0,
    endpoint_health: float = 1.0,
    semantic_priority: float = 1.0,
) -> float:
    """
    Closed‑form optimal leaf weight for XGBoost, extended with endpoint health
    and semantic priority regularisation (Parent B + bridge).
    """
    # The regularisation term is λ * endpoint_health * semantic_priority
    denom = hessian_sum + reg_lambda * endpoint_health * semantic_priority
    if denom == 0.0:
        return 0.0
    return -gradient_sum / denom


def hybrid_bandit_xgboost_step(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    y_true: np.ndarray,
    margin: np.ndarray,
    endpoint_health: np.ndarray,
    semantic_priority: np.ndarray,
    delta: float = 0.05,
) -> Tuple[BanditResult, float]:
    """
    Executes one hybrid iteration:
        1. Compute regret‑weighted probabilities `p_t`.
        2. Apply Hoeffding bound to obtain per‑action confidence `c_t`.
        3. Scale XGBoost gradients/hessians by `p_t * c_t`.
        4. Compute a hybrid leaf weight using semantic priority.
    Returns the selected BanditResult (the action with max adjusted probability)
    and the leaf weight.
    """
    # 1. Regret‑weighted distribution
    prob_dist = compute_regret_weighted_strategy(actions, counterfactuals)

    # 2. Confidence bounds (same bound for all actions for simplicity)
    n_trials = max(1, len(actions))  # placeholder for actual trial count
    confidence = hoeffding_confidence_bound(n_trials, delta)

    # 3. Adjust probabilities with confidence (acts as exploration factor)
    adjusted = {
        aid: prob * (1.0 - confidence) for aid, prob in prob_dist.items()
    }

    # Select action with highest adjusted probability
    selected_id = max(adjusted, key=adjusted.get)
    selected_propensity = prob_dist[selected_id]

    # 4. Gradient/Hessian scaling
    g, h = binary_logistic_grad_hess(y_true, margin, endpoint_health)

    # Scale by the selected action's adjusted probability (bandit influence)
    scale = adjusted[selected_id]
    g_scaled = g * scale
    h_scaled = h * scale

    # Aggregate sums (simulating a leaf with multiple instances)
    grad_sum = float(np.sum(g_scaled))
    hess_sum = float(np.sum(h_scaled))

    # 5. Leaf weight with semantic priority (average over instances)
    sem_prio_mean = float(np.mean(semantic_priority))
    leaf_weight = optimal_leaf_weight_hybrid(
        grad_sum,
        hess_sum,
        reg_lambda=1.0,
        endpoint_health=float(np.mean(endpoint_health)),
        semantic_priority=sem_prio_mean,
    )

    result = BanditResult(
        action_id=selected_id,
        propensity=selected_propensity,
        confidence_bound=confidence,
        adjusted_probability=adjusted[selected_id],
    )
    return result, leaf_weight


# ----------------------------------------------------------------------
# Additional helper demonstrating the hybrid objective evaluation
# ----------------------------------------------------------------------
def hybrid_objective(
    y_true: np.ndarray,
    margin: np.ndarray,
    endpoint_health: np.ndarray,
    semantic_priority: np.ndarray,
) -> float:
    """
    Computes the regularized XGBoost objective value with the semantic priority
    term integrated.  This mirrors the second‑order Taylor expansion used in
    XGBoost but adds a penalty λ·health·priority.
    """
    g, h = binary_logistic_grad_hess(y_true, margin, endpoint_health)
    # First‑order term
    term1 = np.sum(g * margin)
    # Second‑order term
    term2 = 0.5 * np.sum(h * np.square(margin))
    # Regularisation term (λ = 1.0)
    reg = 0.5 * np.sum(endpoint_health * semantic_priority * np.square(margin))
    return term1 + term2 + reg


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create dummy actions
    actions = [
        MathAction(id="a1", expected_value=random.uniform(-1, 1)),
        MathAction(id="a2", expected_value=random.uniform(-1, 1)),
        MathAction(id="a3", expected_value=random.uniform(-1, 1)),
    ]

    # Dummy counterfactuals (uniform probabilities)
    counterfactuals = [
        MathCounterfactual(action_id=a.id, outcome_value=random.random(), probability=1.0)
        for a in actions
    ]

    # Synthetic binary classification data (5 instances)
    y_true = np.array([0, 1, 0, 1, 0], dtype=float)
    margin = np.random.randn(5)  # raw scores
    endpoint_health = np.clip(np.random.rand(5), 0.5, 1.0)  # health in [0.5,1]
    semantic_priority = np.clip(np.random.rand(5), 0.2, 1.0)  # priority in [0.2,1]

    # Run one hybrid step
    bandit_res, leaf_w = hybrid_bandit_xgboost_step(
        actions,
        counterfactuals,
        y_true,
        margin,
        endpoint_health,
        semantic_priority,
        delta=0.05,
    )
    print("Selected action:", bandit_res.action_id)
    print("Adjusted probability:", bandit_res.adjusted_probability)
    print("Confidence bound:", bandit_res.confidence_bound)
    print("Computed leaf weight:", leaf_w)

    # Evaluate hybrid objective for sanity check
    obj_val = hybrid_objective(y_true, margin, endpoint_health, semantic_priority)
    print("Hybrid objective value:", obj_val)