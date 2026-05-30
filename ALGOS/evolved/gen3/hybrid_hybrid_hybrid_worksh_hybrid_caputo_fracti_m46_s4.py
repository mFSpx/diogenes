# DARWIN HAMMER — match 46, survivor 4
# gen: 3
# parent_a: hybrid_hybrid_workshare_all_liquid_time_constant_m67_s2.py (gen2)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py (gen1)
# born: 2026-05-29T23:26:38Z

import math
import numpy as np
from datetime import date
from typing import List, Dict, Tuple, Sequence

# ---------------------------------------------------------------------------
# Constants & Helpers
# ---------------------------------------------------------------------------

GROUPS = ("codex", "groq", "cohere", "local_models")


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


# ---------------------------------------------------------------------------
# Lanczos coefficients for Gamma approximation (used by caputo kernel)
# ---------------------------------------------------------------------------

_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857
])


def _gamma(z: float) -> float:
    """Lanczos approximation of the Gamma function."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * _gamma(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x


# ---------------------------------------------------------------------------
# Caputo kernel utilities (fractional memory)
# ---------------------------------------------------------------------------


def caputo_kernel(alpha: float, t: np.ndarray) -> np.ndarray:
    """
    Compute the raw (unnormalized) Caputo kernel values for a vector of time indices.
    The kernel is t^{alpha-1} / Gamma(alpha).
    """
    if alpha <= 0:
        raise ValueError("Fractional order alpha must be positive.")
    # avoid t=0 when alpha<1 (singularity) by starting at 1
    t = np.where(t == 0, 1e-12, t)
    return t ** (alpha - 1) / _gamma(alpha)


def normalized_caputo_weights(alpha: float, length: int, scale: float = 1.0) -> np.ndarray:
    """
    Produce a normalized weight vector of given length.
    ``scale`` allows a dynamic modulation of the fractional order (e.g. via LTC).
    """
    effective_alpha = max(alpha * scale, 1e-6)  # keep >0
    times = np.arange(1, length + 1)  # start at 1 to avoid zero‑power issues
    raw = caputo_kernel(effective_alpha, times)
    total = raw.sum()
    if total == 0:
        return np.full_like(raw, 1.0 / length)
    return raw / total


def fractional_weighted_sum(weights: np.ndarray, history: np.ndarray) -> float:
    """Apply the weights to a numeric history vector."""
    if weights.shape != history.shape:
        raise ValueError("Weights and history must have the same shape.")
    return float(np.dot(weights, history))


# ---------------------------------------------------------------------------
# LTC (Liquid Time‑Constant) utilities
# ---------------------------------------------------------------------------


def init_hybrid_ltc() -> Tuple[float, float]:
    """
    Initialise LTC parameters for a single‑dimensional day‑of‑week input.

    Returns
    -------
    tau_max : float
        Upper bound for the effective time constant.
    llm_base : float
        Baseline LLM share before modulation.
    """
    tau_max = 1.0          # maximum effective time constant
    llm_base = 0.5         # baseline LLM share
    return tau_max, llm_base


def ltc_modulation_factor(tau_sys: float, tau_max: float, day_of_week: float) -> float:
    """
    Compute a smooth modulation factor based on the effective time constant
    and the normalized day of week.  The factor varies between 0 and 1.

    Parameters
    ----------
    tau_sys : float
        Current effective time constant.
    tau_max : float
        Maximum allowed time constant.
    day_of_week : float
        Normalised day of week in [0, 1].

    Returns
    -------
    float
        Modulation factor.
    """
    # Linear interpolation blended with a sinusoidal term for richer dynamics
    base = tau_sys / tau_max
    sinusoid = 0.5 * (1 + math.sin(2 * math.pi * day_of_week - math.pi / 2))
    return base * (0.6 + 0.4 * sinusoid)  # keep within [0,1]


def hybrid_allocate_by_dates(
    tau_sys: float,
    llm_base: float,
    tau_max: float,
    dates: Sequence[date]
) -> List[Dict[str, float]]:
    """
    Compute per‑day, per‑group allocations using the LTC‑modulated LLM share.

    The allocation for each group is the same, but the overall magnitude varies
    with the day of week and the LTC dynamics.
    """
    allocations: List[Dict[str, float]] = []
    for d in dates:
        # Normalise weekday (Monday=0 … Sunday=6) to [0,1]
        day_norm = (d.weekday()) / 6.0
        factor = ltc_modulation_factor(tau_sys, tau_max, day_norm)
        llm_units = llm_base * factor
        allocation = {g: _pct(llm_units) for g in GROUPS}
        allocations.append(allocation)
    return allocations


# ---------------------------------------------------------------------------
# Fractional‑memory tree cost utilities
# ---------------------------------------------------------------------------


def edge_length(edge: Sequence[float]) -> float:
    """Euclidean length of a tree edge."""
    return float(np.linalg.norm(edge))


def incremental_fractional_tree_cost(
    alpha: float,
    material: float,
    path_weight: float,
    edges: Sequence[Sequence[float]],
    distances: Sequence[float],
    tau_sys: float,
    tau_max: float
) -> float:
    """
    Build the tree edge‑by‑edge, update distances, and evaluate the hybrid cost.

    The LTC time constant scales the fractional order, providing a tighter
    coupling between the two parent systems.
    """
    # Build history: edge lengths followed by explicit distances
    edge_lengths = np.array([edge_length(e) for e in edges], dtype=float)
    distance_arr = np.array(distances, dtype=float)
    history = np.concatenate([edge_lengths, distance_arr])

    if history.size == 0:
        raise ValueError("History for fractional cost cannot be empty.")

    # Normalised Caputo weights, modulated by LTC
    scale = tau_sys / tau_max
    weights = normalized_caputo_weights(alpha, len(history), scale=scale)

    weighted_sum = fractional_weighted_sum(weights, history)
    cost = material + path_weight * weighted_sum
    return float(cost)


def fractional_ssm_step(alpha: float, x: Sequence[float], u: float, tau_sys: float, tau_max: float) -> float:
    """
    Generic state‑space update that also uses the same LTC‑modulated Caputo weighting.
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.size == 0:
        raise ValueError("State vector x cannot be empty.")
    scale = tau_sys / tau_max
    weights = normalized_caputo_weights(alpha, x_arr.size, scale=scale)
    weighted_sum = fractional_weighted_sum(weights, x_arr)
    return weighted_sum + u


# ---------------------------------------------------------------------------
# Integrated hybrid system
# ---------------------------------------------------------------------------


def hybrid_system(
    tau_sys: float,
    alpha: float,
    material: float,
    path_weight: float,
    edges: Sequence[Sequence[float]],
    distances: Sequence[float],
    dates: Sequence[date]
) -> Tuple[List[Dict[str, float]], float]:
    """
    Combine the LTC module and the fractional‑memory tree cost module.

    Returns
    -------
    allocations : list of dict
        Per‑day, per‑group LLM allocations.
    fractional_cost : float
        The hybrid tree cost incorporating LTC‑scaled fractional memory.
    """
    tau_max, llm_base = init_hybrid_ltc()
    allocations = hybrid_allocate_by_dates(tau_sys, llm_base, tau_max, dates)
    fractional_cost = incremental_fractional_tree_cost(
        alpha, material, path_weight, edges, distances, tau_sys, tau_max
    )
    return allocations, fractional_cost


# ---------------------------------------------------------------------------
# Smoke test (executed only when run as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test parameters
    alpha = 0.5                     # fractional order
    material = 1.0                  # material length
    path_weight = 0.5               # path weight
    edges = [[1, 2], [2, 3], [3, 4]]
    distances = [1.0, 2.0, 3.0]
    dates = [date(2022, 7, 1), date(2022, 7, 2), date(2022, 7, 3)]

    # Effective time constant (could be produced by a learned gating function)
    tau_sys = 0.8

    allocations, fractional_cost = hybrid_system(
        tau_sys, alpha, material, path_weight, edges, distances, dates
    )

    print("Allocations:")
    for a in allocations:
        print(a)
    print("Fractional cost:", _pct(fractional_cost))