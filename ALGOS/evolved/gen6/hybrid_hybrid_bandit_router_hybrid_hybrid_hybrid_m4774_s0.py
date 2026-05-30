# DARWIN HAMMER — match 4774, survivor 0
# gen: 6
# parent_a: hybrid_bandit_router_poikilotherm_schoolf_m20_s3.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_rbf_surrogate_m1813_s0.py (gen5)
# born: 2026-05-29T23:57:57Z

"""
Module hybrid_fusion: A hybrid algorithm fusing the structures of 'hybrid_bandit_router_poikilotherm_schoolf_m20_s3.py' 
and 'hybrid_hybrid_hybrid_hybrid_hybrid_rbf_surrogate_m1813_s0.py'. The mathematical bridge lies in the integration of 
the certainty measures from the radial basis function surrogate model with the Schoolfield temperature model, 
where the developmental rate is used as a factor to update the confidence in labeling function results.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

R_CAL = 1.987  # cal mol^-1 K^-1
K25 = 298.15  # reference temperature (25 °C) in Kelvin

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL

@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")

    def as_dict(self) -> Dict[str, str | int | Tuple[str, ...]]:
        return {
            "label": self.label,
            "confidence_bps": self.confidence_bps,
            "authority_class": self.authority_class,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "generated_at": self.generated_at
        }

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
    """Clear all stored reward statistics."""
    _POLICY.clear()

def _reward(a: str) -> float:
    """Empirical mean reward for action *a* (0 if never tried)."""
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def update_policy(updates: List[BanditUpdate]) -> None:
    """In‑place update of the global policy with a batch of observations."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Full Schoolfield rate as a function of absolute temperature."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / K25) * math.exp(
        -params.delta_h_activation / (params.r_cal * temp_k)
    )
    denominator = 1 + math.exp(
        (params.delta_h_low / (params.r_cal * temp_k)) * ((temp_k - params.t_low) / (params.t_high - temp_k))
    ) + math.exp(
        (params.delta_h_high / (params.r_cal * temp_k)) * ((params.t_high - temp_k) / (temp_k - params.t_low))
    )
    return numerator / denominator

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate")
        m[col], m[pivot] = m[pivot], m[col]
        pivot_val = m[col][col]
        m[col] = [v / pivot_val for v in m[col]]
        for row in range(n):
            if row != col:
                m[row] = [row_val - m[row][col] * col_val for row_val, col_val in zip(m[row], m[col])]
    return [row[-1] for row in m]

def update_certainty_flags(flags: List[CertaintyFlag], temp_k: float, params: SchoolfieldParams) -> None:
    for flag in flags:
        rate = developmental_rate(temp_k, params)
        flag.confidence_bps = int(flag.confidence_bps * rate)

def get_action_confidence(action: BanditAction, temp_k: float, params: SchoolfieldParams) -> float:
    rate = developmental_rate(temp_k, params)
    return action.propensity * rate

def get_flag_confidence(flag: CertaintyFlag, temp_k: float, params: SchoolfieldParams) -> float:
    rate = developmental_rate(temp_k, params)
    return flag.confidence_bps * rate

if __name__ == "__main__":
    params = SchoolfieldParams()
    temp_k = c_to_k(25)
    flag = CertaintyFlag("FACT", 1000, "High", "rationale", ("ref1", "ref2"))
    action = BanditAction("action1", 0.5, 10, 0.1, "algo1")
    update_certainty_flags([flag], temp_k, params)
    confidence = get_action_confidence(action, temp_k, params)
    flag_confidence = get_flag_confidence(flag, temp_k, params)
    assert confidence >= 0
    assert flag_confidence >= 0