# DARWIN HAMMER — match 2126, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_fisher_locali_hybrid_caputo_fracti_m1185_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m558_s0.py (gen5)
# born: 2026-05-29T23:40:53Z

"""Hybrid Fisher‑Caputo‑Morphology‑Schoolfield Model Scoring

This module merges the core mathematics of two parent algorithms:

* **Parent A** – Provides a Fisher‑information measure of a text string and a
  fractional decay kernel κₐ(t)=t^{α‑1}/Γ(α).
* **Parent B** – Supplies a sphericity index for 3‑D morphology and the
  Schoolfield temperature‑rate equation.

The mathematical bridge is a *hybrid score* that treats the Fisher
information of a model’s identifier as an “informational activation energy”.
It is combined multiplicatively with a geometry factor (sphericity) and a
temperature‑dependent rate (Schoolfield).  The whole product is then
weighted by the fractional decay kernel to model memory fading over time.

The resulting system can be used to rank models in a pool, taking into
account information content, shape, thermal performance and temporal decay.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities
# ----------------------------------------------------------------------
def gamma_func(z: float) -> float:
    """Accurate Gamma using log‑Gamma to avoid overflow."""
    if z <= 0:
        raise ValueError("Gamma argument must be positive")
    return math.exp(math.lgamma(z))


def fractional_decay(alpha: float, t: float) -> float:
    """
    Fractional decay kernel κ_α(t) = t^{α‑1} / Γ(α).

    Parameters
    ----------
    alpha : float
        Order of the fractional operator (0 < α ≤ 1).
    t : float
        Positive time instant.

    Returns
    -------
    float
        Decay factor.
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    if t <= 0:
        raise ValueError("time t must be positive")
    return t ** (alpha - 1) / gamma_func(alpha)


def fisher_information(text: str, eps: float = 1e-12) -> float:
    """
    Compute a scalar Fisher information for a string based on its
    character‑frequency distribution.

    Returns a non‑negative scalar; larger values indicate a more
    “informative” (i.e. less uniform) text.
    """
    if not text:
        return eps

    # histogram of byte values
    bytes_arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    hist, _ = np.histogram(bytes_arr, bins=256, range=(0, 256), density=False)
    p = hist.astype(np.float64) / hist.sum() + eps  # avoid zero probabilities

    # discrete gradient (central differences)
    grad = np.gradient(p)

    fisher = np.sum((grad ** 2) / p)
    return max(fisher, eps)


# ----------------------------------------------------------------------
# Parent B structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor of a model."""
    name: str
    ram_mb: int
    tier: str


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity: ratio of the volume‑based sphere radius to the mean edge."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / ((length + width + height) / 3.0)


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal·K⁻¹·mol⁻¹


def schoolfield_rate(params: SchoolfieldParams, temperature: np.ndarray) -> np.ndarray:
    """
    Compute the Schoolfield temperature‑rate curve for an array of temperatures.

    Returns an array of the same shape as `temperature` containing the
    dimensionless rate relative to the reference temperature 25 °C (298.15 K).
    """
    T = temperature.astype(np.float64)
    R = params.r_cal * 4.184  # convert cal to Joule

    # Activation term
    act = np.exp(-params.delta_h_activation / R * (1.0 / T - 1.0 / 298.15))

    # Low‑temperature inhibition
    low = np.exp(params.delta_h_low / R * (1.0 / params.t_low - 1.0 / T))

    # High‑temperature inhibition
    high = np.exp(params.delta_h_high / R * (1.0 / params.t_high - 1.0 / T))

    rate = params.rho_25 * act * low * high
    return rate


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_information_factor(model_name: str) -> float:
    """
    Convert the Fisher information of a model identifier into a dimensionless
    factor in (0, ∞).  The factor is normalised by the maximum possible Fisher
    information for a 256‑byte alphabet (empirically approximated).
    """
    raw = fisher_information(model_name)
    # Empirical upper bound for a 1‑KB random string (very high variance)
    max_estimate = 1e4
    return raw / max_estimate + 1e-6  # avoid exact zero


def compute_geometry_factor(morph: Morphology) -> float:
    """
    Geometry factor derived from sphericity and mass.
    Higher sphericity (more sphere‑like) and lower mass increase the factor.
    """
    sph = sphericity_index(morph.length, morph.width, morph.height)
    # Normalise sphericity to (0,1] and combine with inverse mass
    mass_factor = 1.0 / (morph.mass + 1e-6)
    return sph * mass_factor


def compute_temperature_factor(params: SchoolfieldParams, temperature: float) -> float:
    """
    Temperature factor for a single temperature value.  The Schoolfield rate
    is evaluated and then normalised to its maximum over a plausible range
    (273 K–323 K) to keep the factor in (0, 1].
    """
    temps = np.array([temperature])
    rate = schoolfield_rate(params, temps)[0]

    # Pre‑compute the maximal rate over the range for normalisation
    temp_grid = np.linspace(273.15, 323.15, 200)
    max_rate = schoolfield_rate(params, temp_grid).max()
    return rate / (max_rate + 1e-12)


def hybrid_model_score(
    model_name: str,
    morph: Morphology,
    params: SchoolfieldParams,
    temperature: float,
    alpha: float = 0.8,
    time: float = 1.0,
) -> float:
    """
    Unified hybrid score.

    S = κ_α(t) · I_fisher · G_geom · T_temp

    where
        κ_α(t)      – fractional decay kernel,
        I_fisher    – normalised Fisher information of the model name,
        G_geom      – geometry factor (sphericity & mass),
        T_temp      – temperature factor (Schoolfield).

    The score grows with information content, compact shape, favourable
    temperature, and decays with elapsed time according to the fractional
    kernel.
    """
    decay = fractional_decay(alpha, time)
    info = compute_information_factor(model_name)
    geom = compute_geometry_factor(morph)
    temp = compute_temperature_factor(params, temperature)

    return decay * info * geom * temp


def rank_models(models: List[Tuple[ModelTier, Morphology]], 
                params: SchoolfieldParams,
                temperature: float,
                alpha: float = 0.8,
                time: float = 1.0) -> List[Tuple[ModelTier, float]]:
    """
    Evaluate a pool of models and return them sorted by descending hybrid score.
    """
    scored = []
    for tier, morph in models:
        score = hybrid_model_score(
            model_name=tier.name,
            morph=morph,
            params=params,
            temperature=temperature,
            alpha=alpha,
            time=time,
        )
        scored.append((tier, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a synthetic pool of models
    random.seed(42)
    pool: List[Tuple[ModelTier, Morphology]] = []
    for i in range(5):
        name = f"model_{i}_{''.join(random.choices('ABCDEF', k=3))}"
        tier = ModelTier(name=name, ram_mb=random.randint(256, 2048), tier="standard")
        morph = Morphology(
            length=random.uniform(0.5, 2.5),
            width=random.uniform(0.5, 2.5),
            height=random.uniform(0.5, 2.5),
            mass=random.uniform(10.0, 100.0),
        )
        pool.append((tier, morph))

    # Parameters for the temperature model
    sf_params = SchoolfieldParams()

    # Example temperature (298 K ≈ 25 °C)
    temp_K = 298.15

    ranked = rank_models(pool, sf_params, temperature=temp_K, alpha=0.9, time=2.0)

    print("Ranked models (highest score first):")
    for tier, score in ranked:
        print(f"{tier.name:20s} | RAM: {tier.ram_mb:4d} MB | Score: {score:.6e}")