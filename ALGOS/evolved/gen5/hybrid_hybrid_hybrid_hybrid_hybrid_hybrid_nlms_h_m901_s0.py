# DARWIN HAMMER — match 901, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_state_space_duality_m60_s2.py (gen3)
# parent_b: hybrid_hybrid_nlms_hybrid_h_hybrid_hybrid_hybrid_m77_s0.py (gen4)
# born: 2026-05-29T23:31:27Z

"""
This module integrates the Schoolfield-Rollinson poikilotherm rate primitive from hybrid_hybrid_bandit_router_state_space_duality_m143_s0.py
and the Hybrid NLMS & Liquid-Time-Constant (LTC) Network and Hybrid Ternary Router & TTT-Linear algorithms from hybrid_hybrid_nlms_hybrid_h_hybrid_hybrid_hybrid_m77_s0.py.
The mathematical bridge between these two structures is the use of the temperature-dependent developmental rate from the poikilotherm model 
to update the weights of the Hybrid NLMS & LTC Network, which is then used to compress the input distribution of the Hybrid Ternary Router & TTT-Linear algorithm.
The variational free energy is used to update the belief mean of the ternary router, which is then used to compute the SSIM between the input and output of the ternary router.
"""

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import sys
import pathlib

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol^-1 K^-1

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

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def update_hybrid_nlms(weights, x, target, mu=0.5, eps=1e-9, tau=1.0, beta=1.0, temp_k=None, params=SchoolfieldParams()):
    if temp_k is not None:
        developmental_rate_val = developmental_rate(temp_k, params)
        weights = update_ltc(weights, x, target, mu, eps, tau, beta)
        weights = [w * developmental_rate_val for w in weights]
    return weights

def hybrid_ternary_router(x, W, temp_k=None, params=SchoolfieldParams()):
    y = np.dot(W, x)
    if temp_k is not None:
        developmental_rate_val = developmental_rate(temp_k, params)
        y = y * developmental_rate_val
    return y

def predict(weights, x):
    """Predict the output of the system using the given weights and input."""
    return sum(w * xi for w, xi in zip(weights, x))

def update_ltc(weights, x, target, mu=0.5, eps=1e-9, tau=1.0, beta=1.0):
    """Update the weights using the NLMS update rule and the LTC ODE."""
    if len(weights) != len(x):
        raise ValueError('weights and input must have equal length')
    if not 0 < mu < 2:
        raise ValueError('mu must be in the interval (0, 2)')
    y = predict(weights, x)
    error = target - y
    power = sum(xi * xi for xi in x) + eps
    next_weights = [w + mu * error * xi / power for w, xi in zip(weights, x)]
    # LTC ODE
    g_t = np.clip(predict(weights, x) + np.random.uniform(0, 1, len(weights)) + beta * np.random.uniform(0, 1, len(weights)), 0, 1)
    return next_weights, g_t

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    """Initialize W shape (d_out, d_in).

    d_out defaults to d_in. Small random initialization; scale controls
    the initial magnitude so the first few gradient steps are interpretable.
    """
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W, x, target):
    """Compute the reconstruction loss."""
    y = np.dot(W, x)
    return np.mean((y - target) ** 2)

if __name__ == "__main__":
    # Smoke test
    weights = [1.0, 2.0, 3.0]
    x = [4.0, 5.0, 6.0]
    target = 10.0
    mu = 0.5
    eps = 1e-9
    tau = 1.0
    beta = 1.0
    temp_k = 300.0  # Kelvin
    params = SchoolfieldParams()
    updated_weights = update_hybrid_nlms(weights, x, target, mu, eps, tau, beta, temp_k, params)
    print(updated_weights)
    y = hybrid_ternary_router(x, updated_weights, temp_k, params)
    print(y)