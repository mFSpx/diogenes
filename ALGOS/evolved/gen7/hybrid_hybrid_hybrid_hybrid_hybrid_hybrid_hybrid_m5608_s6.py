# DARWIN HAMMER — match 5608, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2259_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_h_m901_s1.py (gen5)
# born: 2026-05-30T00:03:20Z

"""Hybrid algorithm combining Fisher‑scoring‑modulated NLMS adaptation with
temperature‑dependent developmental rate (Schoolfield model).

Parents:
- hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1136_s2.py
  (regret‑based strategy providing Fisher scores)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m607_s0.py
  (NLMS adaptive filter)

Mathematical bridge:
The Fisher score vector 𝑠 derived from the regret‑based strategy scales the
NLMS step‑size μ.  The base step‑size is further modulated by the
temperature‑dependent developmental rate ρ(T) from the Schoolfield model.
Thus the effective step‑size is

    μ_eff = μ₀ · ρ(T) · (1 + ‖s‖₂)

and the weight update follows the Normalised LMS rule

    e   = d − wᵀx
    w'  = w + μ_eff · e·x / (ε + ‖x‖₂²)

where ε prevents division by zero.  This fuses the statistical information
(Fisher scores) with the adaptive filtering and thermodynamic scaling in a
single unified system.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action identifier with its expected value and optional cost/risk."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for a given action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Schoolfield developmental rate (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol⁻¹ K⁻¹


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield‑Rollinson poikilotherm rate.
    Returns a temperature‑dependent scaling factor ρ(T).
    """
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = (
        params.rho_25
        * (temp_k / 298.15)
        * math.exp(
            (params.delta_h_activation / params.r_cal)
            * ((1.0 / 298.15) - (1.0 / temp_k))
        )
    )
    low = math.exp(
        (params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k))
    )
    high = math.exp(
        (params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k))
    )
    return numerator / (1.0 + low + high)


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def compute_fisher_score_vector(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> np.ndarray:
    """
    Approximate Fisher score vector s_i = (observed - expected) / variance_i
    for each action i.  For simplicity variance is taken as 1 + |expected|.
    The returned vector length equals the number of distinct actions.
    """
    # Build a mapping from action id to expected value
    expected_map = {a.id: a.expected_value for a in actions}
    # Initialise score list
    scores = []
    for cf in counterfactuals:
        exp_val = expected_map.get(cf.action_id, 0.0)
        variance = 1.0 + abs(exp_val)
        score = (cf.outcome_value - exp_val) / variance
        scores.append(score)
    if not scores:
        return np.zeros(0)
    return np.array(scores, dtype=float)


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Standard NLMS prediction wᵀx."""
    return float(np.dot(weights, x))


def hybrid_nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    desired: float,
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    temperature_c: float,
    base_mu: float = 0.1,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Perform a single hybrid NLMS weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current filter weights (shape (N,)).
    x : np.ndarray
        Input vector (shape (N,)).
    desired : float
        Desired output d.
    actions / counterfactuals : list
        Provide the statistical context for Fisher scoring.
    temperature_c : float
        Ambient temperature in Celsius; converted to Kelvin for the
        developmental_rate function.
    base_mu : float, optional
        Baseline NLMS step size.
    epsilon : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Updated weight vector.
    """
    # 1. Compute Fisher score norm
    s_vec = compute_fisher_score_vector(actions, counterfactuals)
    fisher_norm = np.linalg.norm(s_vec) if s_vec.size else 0.0

    # 2. Temperature‑dependent scaling via Schoolfield model
    temp_k = c_to_k(temperature_c)
    rho_T = developmental_rate(temp_k)

    # 3. Effective step size μ_eff
    mu_eff = base_mu * rho_T * (1.0 + fisher_norm)

    # 4. NLMS error and normalisation term
    y = nlms_predict(weights, x)
    error = desired - y
    norm_factor = epsilon + np.dot(x, x)

    # 5. Weight update
    delta_w = (mu_eff * error / norm_factor) * x
    return weights + delta_w


def hybrid_process_batch(
    init_weights: np.ndarray,
    inputs: np.ndarray,
    desired_outputs: np.ndarray,
    actions_batch: List[List[MathAction]],
    counterfactuals_batch: List[List[MathCounterfactual]],
    temperature_c: float,
) -> np.ndarray:
    """
    Apply `hybrid_nlms_update` sequentially over a batch of samples.

    Parameters
    ----------
    init_weights : np.ndarray
        Starting weight vector.
    inputs : np.ndarray
        Input matrix of shape (M, N) where M is batch size.
    desired_outputs : np.ndarray
        Desired output vector of shape (M,).
    actions_batch / counterfactuals_batch : list of lists
        Per‑sample statistical contexts.
    temperature_c : float
        Ambient temperature (Celsius) applied uniformly to the batch.

    Returns
    -------
    np.ndarray
        Final weight vector after processing the whole batch.
    """
    w = init_weights.copy()
    for i in range(inputs.shape[0]):
        w = hybrid_nlms_update(
            w,
            inputs[i],
            desired_outputs[i],
            actions_batch[i],
            counterfactuals_batch[i],
            temperature_c,
        )
    return w


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Define a small problem
    dim = 5
    init_w = np.zeros(dim, dtype=float)

    # Random inputs and desired outputs
    X = np.random.randn(10, dim)
    d = np.random.randn(10)

    # Create dummy actions / counterfactuals per sample
    actions_batch = []
    counterfactuals_batch = []
    for _ in range(10):
        acts = [
            MathAction(id=f"a{i}", expected_value=random.uniform(-1, 1))
            for i in range(dim)
        ]
        cfs = [
            MathCounterfactual(
                action_id=f"a{i}",
                outcome_value=random.uniform(-1, 1),
                probability=random.random(),
            )
            for i in range(dim)
        ]
        actions_batch.append(acts)
        counterfactuals_batch.append(cfs)

    # Run hybrid processing at 25 °C
    final_w = hybrid_process_batch(
        init_weights=init_w,
        inputs=X,
        desired_outputs=d,
        actions_batch=actions_batch,
        counterfactuals_batch=counterfactuals_batch,
        temperature_c=25.0,
    )

    print("Initial weights:", init_w)
    print("Final weights after hybrid NLMS updates:", final_w)
    # Verify that predictions have changed
    preds_before = X @ init_w
    preds_after = X @ final_w
    mse_before = np.mean((d - preds_before) ** 2)
    mse_after = np.mean((d - preds_after) ** 2)
    print(f"MSE before: {mse_before:.6f}, after: {mse_after:.6f}")