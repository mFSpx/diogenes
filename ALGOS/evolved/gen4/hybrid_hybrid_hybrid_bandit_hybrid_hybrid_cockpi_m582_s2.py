# DARWIN HAMMER — match 582, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_endpoi_m134_s2.py (gen3)
# parent_b: hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s4.py (gen2)
# born: 2026-05-29T23:29:53Z

"""
Hybrid module fusing the DARWIN HAMMER parents:
- hybrid_hybrid_bandit_router_hybrid_hybrid_endpoi_m134_s2.py (gen 3)
- hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s4.py (gen 2)

The mathematical bridge lies in the interaction between the bandit algorithm's 
developmental rate and the cockpit's trust-weighted style vectors. We embed 
the bandit's temperature-dependent rate into the cockpit's velocity field, 
scaling the velocity with the bandit's normalized activity.

The hybrid system integrates the bandit's developmental rate equation into 
the cockpit's rectified-flow core, modulating the velocity field with the 
bandit's activity level. This yields a temperature-dependent, trust-weighted 
style target.

The module provides three representative hybrid functions:
* `hybrid_style_target` – compute the trust-weighted style target.
* `hybrid_style_loss`   – MSE loss against the target.
* `hybrid_euler_step`   – Euler integration toward the target.
"""

import numpy as np
import math
import random
from dataclasses import dataclass

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

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def normalized_activity(temp_c: float, low_c: float = 5.0, high_c: float = 40.0) -> float:
    params = SchoolfieldParams()
    temp_k = c_to_k(temp_c)
    rate = developmental_rate(temp_k, params)
    return rate / (rate + 1.0)

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int, total_displayed: int) -> float:
    if total_displayed <= 0:
        return 1.0
    return max(0.0, min(1.0, displayed_ok / total_displayed))

def hybrid_style_target(v0: np.ndarray, v1: np.ndarray, temp_c: float, claims_with_evidence: int, total_claims_emitted: int, displayed_ok: int, unknown_displayed_as_ok: int, total_displayed: int) -> np.ndarray:
    activity = normalized_activity(temp_c)
    trust_factor = anti_slop_ratio(claims_with_evidence, total_claims_emitted) * cockpit_honesty(displayed_ok, unknown_displayed_as_ok, total_displayed)
    velocity = v1 - v0
    return v0 + trust_factor * activity * velocity

def hybrid_style_loss(v_target: np.ndarray, v_pred: np.ndarray) -> float:
    return np.mean((v_target - v_pred) ** 2)

def hybrid_euler_step(v0: np.ndarray, v_target: np.ndarray, step_size: float) -> np.ndarray:
    return v0 + step_size * (v_target - v0)

if __name__ == "__main__":
    v0 = np.array([1.0, 2.0, 3.0])
    v1 = np.array([4.0, 5.0, 6.0])
    temp_c = 25.0
    claims_with_evidence = 10
    total_claims_emitted = 20
    displayed_ok = 15
    unknown_displayed_as_ok = 2
    total_displayed = 20
    step_size = 0.1
    
    v_target = hybrid_style_target(v0, v1, temp_c, claims_with_evidence, total_claims_emitted, displayed_ok, unknown_displayed_as_ok, total_displayed)
    v_pred = hybrid_euler_step(v0, v_target, step_size)
    loss = hybrid_style_loss(v_target, v_pred)
    
    print("v_target:", v_target)
    print("v_pred:", v_pred)
    print("loss:", loss)