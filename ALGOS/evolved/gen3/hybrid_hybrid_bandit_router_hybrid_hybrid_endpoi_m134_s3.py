# DARWIN HAMMER — match 134, survivor 3
# gen: 3
# parent_a: hybrid_bandit_router_poikilotherm_schoolf_m20_s0.py (gen1)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_krampus_brain_m24_s2.py (gen2)
# born: 2026-05-29T23:25:54Z

import math
import random
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Immutable description of a bandit arm."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Immutable record of a single interaction."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters of the Schoolfield temperature‑performance curve."""
    rho_25: float = 1.0                # rate at 25 °C (in arbitrary units)
    delta_h_activation: float = 12_000.0  # J mol⁻¹
    t_low: float = 283.15              # K  (≈10 °C)
    t_high: float = 307.15             # K  (≈34 °C)
    delta_h_low: float = -45_000.0     # J mol⁻¹
    delta_h_high: float = 65_000.0     # J mol⁻¹
    r_cal: float = 1.987               # cal mol⁻¹ K⁻¹ (≈8.314 J mol⁻¹ K⁻¹)


@dataclass
class EndpointCircuitBreaker:
    """Mutable circuit‑breaker tracking failures."""
    failure_threshold: int = 3
    failures: int = 0

    def record_failure(self) -> None:
        """Increment failure count, capped at the threshold."""
        self.failures = min(self.failures + 1, self.failure_threshold)

    def reset(self) -> None:
        """Reset failure counter."""
        self.failures = 0


# ----------------------------------------------------------------------
# Global policy store (simple in‑memory bandit statistics)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])  # [cumulative_reward, count]

def reset_policy() -> None:
    """Clear all learned statistics."""
    _POLICY.clear()


def update_policy(updates: List[BanditUpdate]) -> None:
    """In‑place update of the global policy using a batch of observations."""
    for u in updates:
        stats = _POLICY[u.action_id]
        stats[0] += float(u.reward)
        stats[1] += 1.0


def _reward(action_id: str) -> float:
    """Return the empirical mean reward for an action (0 if unseen)."""
    total, n = _POLICY.get(action_id, (0.0, 0.0))
    return total / n if n else 0.0


# ----------------------------------------------------------------------
# Temperature utilities
# ----------------------------------------------------------------------
def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def _safe_exp(x: float, *, clip: float = 700.0) -> float:
    """Exponentiation with overflow protection (clip to a safe range)."""
    return math.exp(max(min(x, clip), -clip))


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield model for temperature‑dependent developmental rate.
    Returns a non‑negative rate; values outside the biologically plausible range
    are smoothly tapered to avoid overflow.
    """
    if temp_k <= 0:
        raise ValueError("Temperature must be positive Kelvin.")
    if params.rho_25 < 0:
        raise ValueError("rho_25 must be non‑negative.")

    # Core exponent terms (clipped to avoid overflow)
    act_exp = (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    low_exp = (params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k))
    high_exp = (params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k))

    numerator = params.rho_25 * (temp_k / 298.15) * _safe_exp(act_exp)
    denominator = 1.0 + _safe_exp(low_exp) + _safe_exp(high_exp)

    rate = numerator / denominator
    return max(rate, 0.0)  # enforce non‑negativity


def normalized_activity(temp_c: float,
                        low_c: float = 5.0,
                        high_c: float = 40.0) -> float:
    """
    Normalised activity ∈ (0, 1) derived from the Schoolfield curve.
    The activity is further bounded to the user‑specified temperature window.
    """
    # Clamp temperature to the biologically relevant window before conversion
    temp_c_clamped = max(min(temp_c, high_c), low_c)
    temp_k = c_to_k(temp_c_clamped)
    rate = developmental_rate(temp_k)
    # Logistic‑like normalisation that never reaches exactly 0 or 1
    return rate / (rate + 1.0)


# ----------------------------------------------------------------------
# Health and curvature utilities
# ----------------------------------------------------------------------
def health_score(failures: int, threshold: int, recovery_priority: float) -> float:
    """
    Base health metric (0 ≤ score ≤ 1). Values are clipped to stay within bounds.
    """
    if threshold <= 0:
        raise ValueError("Threshold must be positive.")
    base = (1.0 - failures / threshold) * (1.0 - recovery_priority)
    return max(min(base, 1.0), 0.0)


def curvature_score(morph_curvature: float, health: float) -> float:
    """
    Morphology‑aware curvature score. The tanh term maps curvature
    (which may be any real number) into a smooth weight ∈ (0, 1).
    """
    weight = 0.5 + 0.5 * math.tanh(morph_curvature)
    return health * weight


# ----------------------------------------------------------------------
# Integrated (temperature‑aware) metrics
# ----------------------------------------------------------------------
def hybrid_health_score(failures: int,
                        threshold: int,
                        recovery_priority: float,
                        temp_c: float) -> float:
    """
    Blend the base health score with temperature‑dependent activity.
    The activity acts as a soft‑switch: at low activity the health leans
    towards the base value, while at high activity the health is amplified.
    """
    base = health_score(failures, threshold, recovery_priority)
    activity = normalized_activity(temp_c)
    # Linear interpolation between base and an activity‑scaled variant
    return (1.0 - activity) * base + activity * (base * activity)


def hybrid_curvature_score(morph_curvature: float,
                           failures: int,
                           threshold: int,
                           recovery_priority: float,
                           temp_c: float) -> float:
    """
    Compute curvature score using the temperature‑aware health metric.
    No additional activity factor is applied to avoid double‑scaling.
    """
    health = hybrid_health_score(failures, threshold, recovery_priority, temp_c)
    return curvature_score(morph_curvature, health)


def hybrid_brain_map(endpoint: str,
                     breaker: EndpointCircuitBreaker,
                     temp_c: float,
                     recovery_priority: float = 0.5,
                     morph_curvature: float = 1.0) -> float:
    """
    High‑level interface used by the surrounding system.
    Returns a single scalar that can be interpreted as a reliability
    or “brain” score for the given endpoint.
    """
    # In a realistic deployment `breaker` would be updated elsewhere.
    health = hybrid_health_score(breaker.failures,
                                 breaker.failure_threshold,
                                 recovery_priority,
                                 temp_c)
    return curvature_score(morph_curvature, health)


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simulated scenario
    breaker = EndpointCircuitBreaker(failure_threshold=3)
    # Record a few random failures to illustrate state changes
    for _ in range(2):
        if random.random() < 0.3:
            breaker.record_failure()

    temperature_c = 20.0
    score = hybrid_brain_map("example_endpoint", breaker, temperature_c)
    print(f"Hybrid brain score at {temperature_c} °C: {score:.4f}")