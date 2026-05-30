# DARWIN HAMMER — match 4702, survivor 2
# gen: 5
# parent_a: poikilotherm_schoolfield.py (gen0)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m1168_s3.py (gen4)
# born: 2026-05-29T23:57:41Z

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Callable, Optional


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
R_CAL = 1.987  # cal·mol⁻¹·K⁻¹
K25 = 298.15   # reference temperature (K)


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameter set for the Schoolfield‐Rollinson poikilotherm rate model."""
    rho_25: float = 1.0                 # rate at 25 °C (K)
    delta_h_activation: float = 12_000.0  # activation enthalpy (cal·mol⁻¹)
    t_low: float = 283.15               # low‑temperature cutoff (K)
    t_high: float = 307.15              # high‑temperature cutoff (K)
    delta_h_low: float = -45_000.0      # low‑temperature enthalpy (cal·mol⁻¹)
    delta_h_high: float = 65_000.0      # high‑temperature enthalpy (cal·mol⁻¹)
    r_cal: float = R_CAL                # gas constant (cal·mol⁻¹·K⁻¹)


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class EngineEndpoint:
    """Placeholder for endpoint metadata; can be extended with weighting factors."""
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    weight: float = 1.0   # optional scalar that modulates the final score


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


# ----------------------------------------------------------------------
# Core model – cached for efficiency and deeper integration
# ----------------------------------------------------------------------
class SchoolfieldModel:
    """
    Encapsulates the Schoolfield rate equation with a cached maximal rate.
    The cache avoids recomputing the expensive maximum on every call and
    provides a vectorised interface for bulk temperature evaluation.
    """

    def __init__(
        self,
        params: SchoolfieldParams = SchoolfieldParams(),
        temp_range_c: Tuple[float, float] = (5.0, 40.0),
        samples: int = 401,
    ):
        self.params = params
        self._temp_grid_k = np.linspace(
            c_to_k(temp_range_c[0]), c_to_k(temp_range_c[1]), max(2, samples)
        )
        self._rate_grid = self._raw_rate(self._temp_grid_k)
        self._max_rate = float(np.max(self._rate_grid))
        if self._max_rate <= 0.0:
            raise ValueError("Maximum developmental rate must be positive.")

    def _raw_rate(self, temp_k: np.ndarray) -> np.ndarray:
        """Un‑normalised Schoolfield rate for an array of Kelvin temperatures."""
        p = self.params
        # Guard against overflow/underflow by clipping exponent arguments
        exp_arg_act = (p.delta_h_activation / p.r_cal) * ((1.0 / K25) - (1.0 / temp_k))
        exp_arg_low = (p.delta_h_low / p.r_cal) * ((1.0 / p.t_low) - (1.0 / temp_k))
        exp_arg_high = (p.delta_h_high / p.r_cal) * ((1.0 / p.t_high) - (1.0 / temp_k))

        # Use np.exp on clipped values for numerical stability
        exp_act = np.exp(np.clip(exp_arg_act, -700, 700))
        low = np.exp(np.clip(exp_arg_low, -700, 700))
        high = np.exp(np.clip(exp_arg_high, -700, 700))

        numerator = p.rho_25 * (temp_k / K25) * exp_act
        denominator = 1.0 + low + high
        return numerator / denominator

    def rate(self, temp_k: float) -> float:
        """Return the normalised developmental rate for a single Kelvin temperature."""
        if temp_k <= 0.0:
            raise ValueError("Temperature must be positive Kelvin.")
        raw = self._raw_rate(np.array([temp_k]))[0]
        return raw / self._max_rate

    def rate_vector(self, temp_k: Sequence[float]) -> np.ndarray:
        """Vectorised normalised rate for an iterable of Kelvin temperatures."""
        temp_arr = np.asarray(temp_k, dtype=float)
        if np.any(temp_arr <= 0):
            raise ValueError("All temperatures must be positive Kelvin.")
        raw = self._raw_rate(temp_arr)
        return raw / self._max_rate


# ----------------------------------------------------------------------
# Morphology‑based indices
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity (unitless, 0 < ≤ 1)."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive.")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness ratio (unitless, larger → flatter)."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive.")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology,
    b: float = 1.0 / 3.0,
    k: float = 0.35,
    neck_lever: float = 1.0,
) -> float:
    """Estimated right‑ing time based on morphology (arbitrary units)."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("Mass and neck lever must be positive.")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalised recovery priority in the range [0, 1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Hybrid SSM – deeper integration of endpoint metadata
# ----------------------------------------------------------------------
class HybridTropicalSSM:
    """
    Provides sequential and parallel evaluation of the hybrid state‑space model.
    The model now incorporates a per‑endpoint weighting factor, caches the
    temperature‑activity mapping, and exploits NumPy for bulk computation.
    """

    def __init__(
        self,
        temp_range_c: Tuple[float, float] = (5.0, 40.0),
        samples: int = 401,
        activity_params: Optional[SchoolfieldParams] = None,
    ):
        self.activity_model = SchoolfieldModel(
            params=activity_params or SchoolfieldParams(),
            temp_range_c=temp_range_c,
            samples=samples,
        )

    def _step(
        self,
        temp_c: float,
        morphology: Morphology,
        endpoint: EngineEndpoint,
    ) -> float:
        """Single‑step evaluation returning a weighted activity‑recovery product."""
        activity = self.activity_model.rate(c_to_k(temp_c))
        recovery = recovery_priority(morphology)
        return activity * recovery * endpoint.weight

    def sequential(
        self,
        temp_c_values: Sequence[float],
        morphology: Morphology,
        endpoint: EngineEndpoint,
    ) -> List[float]:
        """Process a list of temperatures using the same morphology/endpoint."""
        temps = np.asarray(temp_c_values, dtype=float)
        activities = self.activity_model.rate_vector(c_to_k(temps))
        recovery = recovery_priority(morphology)
        weighted = activities * recovery * endpoint.weight
        return weighted.tolist()

    def parallel(
        self,
        temp_c_values: Sequence[float],
        morphologies: Sequence[Morphology],
        endpoints: Sequence[EngineEndpoint],
    ) -> List[float]:
        """
        Process heterogeneous inputs. Lengths of the three sequences must match.
        Vectorisation is limited by differing morphology parameters, so we loop
        but keep the temperature conversion vectorised.
        """
        if not (len(temp_c_values) == len(morphologies) == len(endpoints)):
            raise ValueError("All input sequences must have the same length.")
        temps_k = c_to_k(np.asarray(temp_c_values, dtype=float))
        activities = self.activity_model.rate_vector(temps_k)

        results = []
        for act, morph, ep in zip(activities, morphologies, endpoints):
            rec = recovery_priority(morph)
            results.append(act * rec * ep.weight)
        return results


# ----------------------------------------------------------------------
# Demo / simple sanity check (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example objects
    demo_morph = Morphology(length=1.2, width=0.8, height=0.6, mass=2.5)
    demo_endpoint = EngineEndpoint(
        engine_id="E42",
        channel="alpha",
        residency="zone‑1",
        runtime="short",
        resource_class="premium",
        weight=1.2,
    )

    # Initialise the hybrid model
    ssm = HybridTropicalSSM()

    # Single step
    print("Single step:", ssm._step(27.0, demo_morph, demo_endpoint))

    # Sequential processing
    temps_seq = [20.0, 25.0, 30.0, 35.0]
    print("Sequential:", ssm.sequential(temps_seq, demo_morph, demo_endpoint))

    # Parallel processing with varying morphologies/endpoints
    morphs = [
        demo_morph,
        Morphology(length=0.9, width=0.9, height=0.5, mass=1.8),
        Morphology(length=1.5, width=1.0, height=0.7, mass=3.0),
    ]
    endpoints = [
        demo_endpoint,
        EngineEndpoint(
            engine_id="E43",
            channel="beta",
            residency="zone‑2",
            runtime="long",
            resource_class="standard",
            weight=0.9,
        ),
        EngineEndpoint(
            engine_id="E44",
            channel="gamma",
            residency="zone‑3",
            runtime="medium",
            resource_class="basic",
            weight=1.0,
        ),
    ]
    temps_par = [22.0, 28.0, 33.0]
    print("Parallel:", ssm.parallel(temps_par, morphs, endpoints))