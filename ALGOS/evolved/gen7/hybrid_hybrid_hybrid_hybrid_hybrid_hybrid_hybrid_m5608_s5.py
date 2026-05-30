# DARWIN HAMMER — match 5608, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2259_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_h_m901_s1.py (gen5)
# born: 2026-05-30T00:03:20Z

"""Hybrid Algorithm: Fisher‑Scored NLMS with Temperature‑Dependent Developmental Rate

Parents:
- hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1136_s2.py (regret‑based Fisher scoring)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_h_m901_s1.py (Schoolfield developmental rate & NLMS)

Mathematical Bridge:
The Fisher score vector **g** computed from regret‑based actions is used to
modulate the NLMS adaptation step size μ.  The temperature‑dependent
developmental rate ρ(T) (Schoolfield model) further scales μ, yielding an
effective step size  

    μ_eff = μ₀ · ρ(T) · (1 + ‖g‖₂),

so that both statistical confidence (via Fisher information) and
physiological temperature influence the adaptive filter.  This unified
update rule drives the weight vector **w** of the NLMS filter while
preserving the original linear prediction model."""


import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Parent A – Regret‑based Fisher scoring utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """An action with its expected value and incurred cost."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """A counterfactual outcome for a given action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


def compute_fisher_score(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> np.ndarray:
    """
    Compute a simple Fisher‑score‑like gradient vector.

    For each action a we form the contribution

        g_a = (outcome - expected) / (variance + ε)

    where variance is approximated by (cost² + risk²).  The resulting
    vector is concatenated in the order of ``actions``.
    """
    eps = 1e-8
    scores = []
    cf_map = {cf.action_id: cf for cf in counterfactuals}
    for act in actions:
        cf = cf_map.get(act.id)
        if cf is None:
            # No counterfactual → zero contribution
            scores.append(0.0)
            continue
        variance = act.cost ** 2 + act.risk ** 2 + eps
        g = (cf.outcome_value - act.expected_value) / variance
        scores.append(g)
    return np.array(scores, dtype=float)


# ----------------------------------------------------------------------
# Parent B – Schoolfield developmental rate (temperature) utilities
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


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield‐Rollinson poikilotherm rate primitive.

    Returns the temperature‑dependent developmental rate ρ(T).
    """
    if temp_k <= 0:
        raise ValueError("Temperature must be > 0 K")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)


# ----------------------------------------------------------------------
# NLMS adaptive filter core (Parent B)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction ŷ = wᵀx."""
    return float(np.dot(weights, x))


def nlms_step(
    weights: np.ndarray,
    x: np.ndarray,
    d: float,
    mu: float,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """
    One Normalised Least‑Mean‑Squares (NLMS) update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (shape (N,)).
    x : np.ndarray
        Input vector (shape (N,)).
    d : float
        Desired (target) scalar output.
    mu : float
        Normalised step size (0 < μ ≤ 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    error : float
        Instantaneous prediction error e = d - ŷ.
    """
    y_hat = nlms_predict(weights, x)
    error = d - y_hat
    norm_factor = np.dot(x, x) + eps
    adaptation = (mu / norm_factor) * error * x
    new_weights = weights + adaptation
    return new_weights, error


# ----------------------------------------------------------------------
# Hybrid operation – integrating Fisher scores and temperature scaling
# ----------------------------------------------------------------------
def hybrid_adapt(
    weights: np.ndarray,
    x: np.ndarray,
    d: float,
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    temperature_c: float,
    base_mu: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """
    Perform a single hybrid adaptation step.

    1. Compute the Fisher‑score vector g from actions/counterfactuals.
    2. Compute its ℓ₂‑norm ‖g‖₂.
    3. Convert temperature to Kelvin and obtain the developmental rate ρ(T).
    4. Form the effective NLMS step size
           μ_eff = μ₀ · ρ(T) · (1 + ‖g‖₂).
    5. Apply the NLMS update with μ_eff.

    The returned tuple mirrors ``nlms_step``: (updated_weights, error).
    """
    # 1‑2. Fisher information contribution
    g = compute_fisher_score(actions, counterfactuals)
    g_norm = np.linalg.norm(g)

    # 3‑4. Temperature scaling
    temp_k = temperature_c + 273.15
    rho_t = developmental_rate(temp_k)
    mu_eff = base_mu * rho_t * (1.0 + g_norm)

    # Clamp μ_eff to the NLMS stability interval (0, 2]
    mu_eff = max(min(mu_eff, 2.0), 1e-6)

    # 5. NLMS adaptation
    new_weights, error = nlms_step(weights, x, d, mu_eff)
    return new_weights, error


# ----------------------------------------------------------------------
# Additional utility: simple Structural Similarity (SSIM) proxy
# ----------------------------------------------------------------------
def simple_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute a lightweight SSIM‑like similarity between two vectors.

    This proxy uses mean, variance and covariance:
        ssim = ((2μ_a μ_b + C1)(2σ_ab + C2)) / ((μ_a²+μ_b² + C1)(σ_a²+σ_b² + C2))
    where C1, C2 are small constants.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_a = np.mean(a)
    mu_b = np.mean(b)
    sigma_a2 = np.var(a)
    sigma_b2 = np.var(b)
    sigma_ab = np.mean((a - mu_a) * (b - mu_b))

    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a2 + sigma_b2 + C2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Dummy NLMS setup
    dim = 5
    w = np.zeros(dim, dtype=float)               # initial weights
    x = np.random.randn(dim)                     # random input vector
    d = 0.7                                        # desired scalar output

    # Dummy regret‑based actions
    actions = [
        MathAction(id=f"a{i}", expected_value=random.uniform(-1, 1),
                   cost=random.uniform(0, 0.2), risk=random.uniform(0, 0.1))
        for i in range(dim)
    ]
    # Counterfactual outcomes (simulated)
    counterfactuals = [
        MathCounterfactual(action_id=act.id,
                           outcome_value=act.expected_value + random.gauss(0, 0.05))
        for act in actions
    ]

    # Temperature in Celsius
    temp_c = 22.0

    # Perform hybrid adaptation
    w_new, err = hybrid_adapt(w, x, d, actions, counterfactuals, temp_c)

    # Display results
    print("Initial weights:", w)
    print("Updated weights:", w_new)
    print("Prediction error:", err)

    # Demonstrate SSIM proxy between raw input and filtered output
    y_pred = nlms_predict(w_new, x)
    ssim_val = simple_ssim(np.array([d]), np.array([y_pred]))
    print("SSIM between target and prediction (proxy):", ssim_val)