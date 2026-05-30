# DARWIN HAMMER — match 5248, survivor 2
# gen: 5
# parent_a: hybrid_poikilotherm_schoolf_hybrid_hybrid_hybrid_m1080_s0.py (gen4)
# parent_b: hybrid_infotaxis_hybrid_semantic_neig_m739_s0.py (gen3)
# born: 2026-05-30T00:00:57Z

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import numpy as np

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
R_CAL = 1.987  # cal·mol⁻¹·K⁻¹
KELVIN_REF = 298.15  # 25 °C in Kelvin
LN2 = math.log(2.0)


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    """Full Schoolfield parameters (activation and low/high temperature deactivation)."""
    rho_25: float = 1.0               # Base rate at 25 °C
    delta_h_activation: float = 12_000.0   # Activation enthalpy (cal·mol⁻¹)
    t_low: float = 283.15            # Low‑temperature breakpoint (K)
    delta_h_low: float = -45_000.0   # Low‑temperature deactivation enthalpy (cal·mol⁻¹)
    t_high: float = 307.15           # High‑temperature breakpoint (K)
    delta_h_high: float = 65_000.0   # High‑temperature deactivation enthalpy (cal·mol⁻¹)
    r_cal: float = R_CAL


@dataclass(frozen=True)
class Morphology:
    """Geometric description of the organism."""
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _validate_positive(*values: float) -> None:
    for v in values:
        if v <= 0.0:
            raise ValueError("All provided dimensions and masses must be positive.")


def sphericity_index(length: float, width: float, height: float) -> float:
    """Dimensionless measure of how close a shape is to a sphere."""
    _validate_positive(length, width, height)
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Dimensionless measure of flatness (larger → flatter)."""
    _validate_positive(length, width, height)
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology,
                        b: float = 1.0 / 3.0,
                        k: float = 0.35,
                        neck_lever: float = 1.0) -> float:
    """A proxy for the time needed to right the organism after overturning."""
    _validate_positive(m.mass, neck_lever)
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def logistic_normalizer(x: float,
                        x0: float = 5.0,
                        k: float = 1.0) -> float:
    """Map any real x to (0,1) using a logistic curve centered at x0."""
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def recovery_priority(m: Morphology,
                      max_index: float = 10.0,
                      logistic_center: float = 5.0,
                      logistic_slope: float = 1.0) -> float:
    """
    Convert the righting‑time index into a normalized priority in [0,1].
    Uses a logistic mapping to avoid hard clipping and to provide a smoother
    transition for extreme morphologies.
    """
    rti = righting_time_index(m)
    # Scale to a comparable range before logistic mapping
    scaled = rti / max_index
    return logistic_normalizer(scaled, logistic_center, logistic_slope)


def _decay_factor(half_life: float, elapsed: float) -> float:
    """Exponential decay factor based on half‑life."""
    if half_life <= 0.0:
        raise ValueError("half_life must be positive")
    if elapsed < 0.0:
        raise ValueError("elapsed time cannot be negative")
    return math.exp(-LN2 * elapsed / half_life)


def calculate_pheromone_signal(signal_value: float,
                               half_life_seconds: float,
                               elapsed_seconds: float,
                               morphology: Morphology,
                               signal_kind: str = "default",
                               kind_weights: Optional[Dict[str, float]] = None) -> float:
    """
    Produce a pheromone signal that is (a) decayed over time,
    (b) weighted by signal kind, and
    (c) modulated by the organism's recovery priority.
    """
    if signal_value < 0.0:
        raise ValueError("signal_value must be non‑negative")
    decay = _decay_factor(half_life_seconds, elapsed_seconds)

    # Default weighting: all kinds equal unless overridden
    weight = 1.0
    if kind_weights is not None:
        weight = kind_weights.get(signal_kind, 1.0)

    priority = recovery_priority(morphology)
    return signal_value * decay * weight * priority


def _schoolfield_rate(params: SchoolfieldParams,
                     temperature: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the classic Schoolfield rate without any pheromone modulation.
    Supports scalar or NumPy array temperatures.
    """
    T = np.asarray(temperature, dtype=float)
    if np.any(T <= 0.0):
        raise ValueError("Temperature must be positive in Kelvin.")

    # Activation term
    act = np.exp(
        (params.delta_h_activation / params.r_cal) *
        (1.0 / KELVIN_REF - 1.0 / T)
    )

    # Low‑temperature deactivation term
    low = 1.0 + np.exp(
        (params.delta_h_low / params.r_cal) *
        (1.0 / params.t_low - 1.0 / T)
    )

    # High‑temperature deactivation term
    high = 1.0 + np.exp(
        (params.delta_h_high / params.r_cal) *
        (1.0 / params.t_high - 1.0 / T)
    )

    return params.rho_25 * act / (low * high)


def temperature_dependent_rate(params: SchoolfieldParams,
                               temperature: Union[float, np.ndarray],
                               pheromone_signal: float) -> Union[float, np.ndarray]:
    """
    Combine the full Schoolfield rate with a pheromone scaling factor.
    The pheromone signal is assumed dimensionless and applied multiplicatively.
    """
    base_rate = _schoolfield_rate(params, temperature)
    return base_rate * pheromone_signal


def hybrid_operation(morphology: Morphology,
                     schoolfield_params: SchoolfieldParams,
                     temperature: Union[float, np.ndarray],
                     signal_value: float,
                     half_life_seconds: float,
                     elapsed_seconds: float = 0.0,
                     signal_kind: str = "default",
                     kind_weights: Optional[Dict[str, float]] = None) -> Union[float, np.ndarray]:
    """
    End‑to‑end hybrid computation:
    1. Build a pheromone signal that respects decay, kind weighting, and morphology.
    2. Feed the signal into the full Schoolfield temperature‑dependent rate.
    """
    pheromone = calculate_pheromone_signal(
        signal_value=signal_value,
        half_life_seconds=half_life_seconds,
        elapsed_seconds=elapsed_seconds,
        morphology=morphology,
        signal_kind=signal_kind,
        kind_weights=kind_weights,
    )
    return temperature_dependent_rate(schoolfield_params, temperature, pheromone)


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a representative organism
    morph = Morphology(length=1.0, width=2.0, height=3.0, mass=10.0)

    # Schoolfield parameters (could be tuned per species)
    sf_params = SchoolfieldParams()

    # Environmental temperature (scalar or vector)
    temp = 300.0  # Kelvin

    # Pheromone characteristics
    sig_val = 1.0
    half_life = 10.0          # seconds
    elapsed = 2.0             # seconds since release
    kind = "foraging"
    weights = {"foraging": 1.2, "alarm": 0.8}

    rate = hybrid_operation(
        morphology=morph,
        schoolfield_params=sf_params,
        temperature=temp,
        signal_value=sig_val,
        half_life_seconds=half_life,
        elapsed_seconds=elapsed,
        signal_kind=kind,
        kind_weights=weights,
    )
    print(f"Hybrid rate (scalar temperature): {rate}")

    # Vectorised example
    temps = np.linspace(280, 320, 5)  # K
    rates_vec = hybrid_operation(
        morphology=morph,
        schoolfield_params=sf_params,
        temperature=temps,
        signal_value=sig_val,
        half_life_seconds=half_life,
        elapsed_seconds=elapsed,
        signal_kind=kind,
        kind_weights=weights,
    )
    print("Hybrid rates (vector temperature):", rates_vec)