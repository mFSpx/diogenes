# DARWIN HAMMER — match 4845, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hard_t_m322_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_poikilotherm__m2665_s0.py (gen6)
# born: 2026-05-29T23:58:19Z

"""Hybrid Algorithm Fusion of:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hard_t_m322_s1.py (circuit‑breaker + morphology)
- hybrid_hybrid_hybrid_hybrid_hybrid_poikilotherm__m2665_s0.py (temperature‑dependent Schoolfield rate + pheromone system)

Mathematical Bridge
------------------
The bridge is built on two shared scalar quantities:
1. **Failure rate** `f ∈ [0,1]` from the circuit‑breaker, which scales any
   downstream signal.
2. **Temperature‑dependent rate** `r(T)` from the Schoolfield model, which
   modulates decay constants.

We fuse them by:
* Multiplying the raw pheromone signal `S` by `(1‑f)` – failures dampen the signal.
* Adjusting the pheromone half‑life `τ` with `τ' = τ / r(T)` – higher biological
  rates accelerate decay.
* Combining the morphologic sphericity `σ` with the scaled pheromone value to
  produce a unified risk score `R = σ * (1‑f) * S * r(T)`.

The module therefore implements a unified system that can be used for
resource‑aware scheduling, risk assessment, or adaptive text‑model routing."""
import sys
import math
import random
import pathlib
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Parent A – Circuit breaker and morphology utilities
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Simple circuit‑breaker tracking consecutive failures."""
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint may receive work)."""
        return not self.open

    def failure_rate(self) -> float:
        """Normalized failure rate in [0,1]."""
        return min(self.failures / self.failure_threshold, 1.0)


@dataclass
class Morphology:
    """Geometric description of an endpoint."""
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of geometric mean to maximal dimension, ∈ (0,1]."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    gm = (length * width * height) ** (1.0 / 3.0)
    return gm / max(length, width, height)


# ----------------------------------------------------------------------
# Parent B – Schoolfield temperature rate and pheromone system
# ----------------------------------------------------------------------
R_CAL = 1.987  # cal mol^-1 K^-1
K25 = 298.15   # reference temperature (K)


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL


def schoolfield_rate(T: float, p: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Temperature‑dependent rate using the Schoolfield equation.

    r(T) = rho_25 *
           exp[-ΔH/(R) * (1/T - 1/K25)] /
           (1 + exp[ΔH_low/R * (1/t_low - 1/T)] + exp[ΔH_high/R * (1/t_high - 1/T)])
    """
    if T <= 0:
        raise ValueError("Temperature must be positive Kelvin")
    num = p.rho_25 * math.exp(-p.delta_h_activation / p.r_cal * (1.0 / T - 1.0 / K25))
    den = (1.0
           + math.exp(p.delta_h_low / p.r_cal * (1.0 / p.t_low - 1.0 / T))
           + math.exp(p.delta_h_high / p.r_cal * (1.0 / p.t_high - 1.0 / T)))
    return num / den


class HybridPheromoneSystem:
    """Manages pheromone signals with temperature‑adjusted decay."""
    def __init__(self):
        # Mapping: surface_key -> dict with current value and half‑life (seconds)
        self._store: dict[str, dict] = {}

    def inject(self, surface_key: str, signal_value: float, half_life_seconds: float) -> None:
        """Create or replace a pheromone entry."""
        if half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be positive")
        self._store[surface_key] = {
            "value": float(signal_value),
            "half_life": float(half_life_seconds),
            "t0": datetime.now(timezone.utc).timestamp()
        }

    def _elapsed(self, t0: float) -> float:
        return datetime.now(timezone.utc).timestamp() - t0

    def decay_factor(self, half_life: float, elapsed: float) -> float:
        """Standard exponential decay factor based on half‑life."""
        return 0.5 ** (elapsed / half_life)

    def get_signal(self,
                   surface_key: str,
                   failure_rate: float,
                   temperature_K: float) -> float:
        """Return the current signal after applying:
        1. exponential decay,
        2. temperature‑dependent acceleration,
        3. circuit‑breaker damping.
        """
        if surface_key not in self._store:
            raise KeyError(f"No pheromone for key {surface_key}")

        entry = self._store[surface_key]
        elapsed = self._elapsed(entry["t0"])
        # Temperature accelerates decay: effective half‑life = half_life / r(T)
        r_T = schoolfield_rate(temperature_K)
        effective_half_life = entry["half_life"] / max(r_T, 1e-9)
        decay = self.decay_factor(effective_half_life, elapsed)
        # Dampen by failure rate
        damped = entry["value"] * decay * (1.0 - failure_rate)
        return damped


# ----------------------------------------------------------------------
# Hybrid Functions Demonstrating the Unified System
# ----------------------------------------------------------------------
def compute_morphology_score(morph: Morphology) -> float:
    """Score based solely on shape – higher sphericity yields higher score."""
    sigma = sphericity_index(morph.length, morph.width, morph.height)
    # Incorporate mass as a simple linear factor (normalised)
    mass_factor = math.tanh(morph.mass / 1000.0)  # mass in kg, saturates near 1
    return sigma * mass_factor


def compute_temperature_adjusted_pheromone(system: HybridPheromoneSystem,
                                           key: str,
                                           breaker: EndpointCircuitBreaker,
                                           temperature_K: float) -> float:
    """Wraps HybridPheromoneSystem.get_signal with the circuit‑breaker state."""
    fr = breaker.failure_rate()
    return system.get_signal(key, failure_rate=fr, temperature_K=temperature_K)


def unified_risk_score(morph: Morphology,
                       breaker: EndpointCircuitBreaker,
                       pheromone_system: HybridPheromoneSystem,
                       pheromone_key: str,
                       temperature_K: float) -> float:
    """Combined risk/utility score:
    R = (morphology_score) * (scaled pheromone signal) * r(T)
    where r(T) is the raw Schoolfield rate (acts as a temperature sensitivity).
    """
    morph_score = compute_morphology_score(morph)
    pheromone_signal = compute_temperature_adjusted_pheromone(
        pheromone_system, pheromone_key, breaker, temperature_K
    )
    temp_rate = schoolfield_rate(temperature_K)
    return morph_score * pheromone_signal * temp_rate


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a circuit breaker that has experienced one failure
    cb = EndpointCircuitBreaker(failure_threshold=3)
    cb.record_failure()          # failures = 1 → failure_rate = 1/3

    # Define a simple morphology
    morph = Morphology(length=2.0, width=1.5, height=1.0, mass=850.0)

    # Initialise pheromone system and inject a signal
    pher = HybridPheromoneSystem()
    pher.inject(surface_key="endpoint_A", signal_value=5.0, half_life_seconds=120.0)

    # Choose a temperature (e.g., 295 K)
    T_K = 295.0

    # Compute the unified risk score
    score = unified_risk_score(
        morph=morph,
        breaker=cb,
        pheromone_system=pher,
        pheromone_key="endpoint_A",
        temperature_K=T_K
    )

    print(f"Unified risk score at {T_K:.1f} K: {score:.6f}")

    # Verify that circuit breaker still allows work (failure_rate < 1)
    assert cb.allow(), "Circuit breaker incorrectly opened"

    # Ensure the pheromone signal is a finite number
    sig = compute_temperature_adjusted_pheromone(pher, "endpoint_A", cb, T_K)
    assert math.isfinite(sig), "Pheromone signal is not finite"

    # Basic sanity check on morphology score
    morph_score = compute_morphology_score(morph)
    assert 0.0 < morph_score <= 1.0, "Morphology score out of expected range"

    print("Smoke test completed successfully.")