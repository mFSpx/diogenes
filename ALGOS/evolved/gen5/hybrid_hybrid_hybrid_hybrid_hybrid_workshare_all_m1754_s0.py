# DARWIN HAMMER — match 1754, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_bandit_router_m999_s0.py (gen4)
# parent_b: hybrid_workshare_allocator_doomsday_calendar_m14_s0.py (gen1)
# born: 2026-05-29T23:38:46Z

"""
Hybrid Algorithm: hybrid_hybrid_workshare_temperature_fusion.py

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_bandit_router_m999_s0.py (temperature‑constrained
  optimization via the Schoolfield model and a composite cost with reconstruction,
  SSIM, and variational free‑energy terms).
- hybrid_workshare_allocator_doomsday_calendar_m14_s0.py (deterministic work‑share
  allocation modulated by the day‑of‑week derived from the Doomsday calendar).

Mathematical Bridge:
Both parents manipulate scalar quantities that influence a global objective:
* The Schoolfield model yields a temperature‑dependent developmental rate `r(T)`.
* The work‑share allocator distributes a total resource `U` into deterministic
  and LLM‑driven portions, where the LLM portion can be further weighted by the
  day‑of‑week factor `d/7`.

We treat the deterministic target percentage `p_det` as a controllable variable
that is *regularized* by the temperature model: higher biologically plausible
temperatures encourage a larger deterministic share, while extreme temperatures
penalize it. The unified cost function is

    J(T, p_det) = w_r·E_rec + w_s·(1‑SSIM) + w_f·F_FE
                  + w_a·|p_det - p_opt(T)|²
                  + w_d·|U_llm·(d/7) - U_target|²

where `p_opt(T)` is a monotonic mapping from temperature to an optimal deterministic
percentage (derived from the normalized Schoolfield rate) and `U_target` is a
user‑defined desired LLM allocation for the given day. Minimising `J` jointly
over temperature `T` and deterministic share `p_det` yields a hybrid solution
that respects both physical plausibility and workload distribution constraints.
"""

import json
import sys
import random
import math
from pathlib import Path
from datetime import datetime, timezone, date
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Constants and utility functions (Parent A)
# ----------------------------------------------------------------------
R_CAL = 1.987  # cal·mol⁻¹·K⁻¹
K25 = 298.15  # reference temperature (25 °C) in Kelvin

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z‑format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0   # cal·mol⁻¹
    t_low: float = 283.15                  # K
    t_high: float = 307.15                 # K
    delta_h_low: float = -45_000.0         # cal·mol⁻¹
    delta_h_high: float = 65_000.0         # cal·mol⁻¹
    r_cal: float = R_CAL

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Full Schoolfield rate as a function of absolute temperature."""
    if temp_k <= 0:
        raise ValueError("Temperature must be positive Kelvin")
    # Helper for Arrhenius term
    def arrhenius(delta_h: float) -> float:
        return math.exp(-delta_h / (params.r_cal * temp_k))
    # Temperature scaling terms
    low_term = (1.0 + math.exp((params.delta_h_low / params.r_cal) *
               (1.0 / params.t_low - 1.0 / temp_k))) ** -1
    high_term = (1.0 + math.exp((params.delta_h_high / params.r_cal) *
                (1.0 / params.t_high - 1.0 / temp_k))) ** -1
    rate = params.rho_25 * arrhenius(params.delta_h_activation) * low_term * high_term
    return rate

# ----------------------------------------------------------------------
# Allocation utilities (Parent B)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    return round(float(value), 6)

def allocate_workshare(*,
                      total_units: float,
                      deterministic_target_pct: float = 90.0,
                      groups: tuple[str, ...] = GROUPS) -> dict:
    """Basic deterministic/LLM split and equal LLM distribution across groups."""
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")
    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    per_group = llm_units / len(groups)
    lanes = [
        {
            "group": group,
            "llm_units": _pct(per_group),
            "llm_share_pct": _pct(100.0 / len(groups)),
            "proof_required": True,
        }
        for group in groups
    ]
    return {
        "total_units": _pct(total_units),
        "deterministic_target_pct": _pct(deterministic_target_pct),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "lanes": lanes,
    }

def doomsday(year: int, month: int, day: int) -> int:
    """Return day of week as 0=Sunday … 6=Saturday (John Conway's Doomsday algorithm)."""
    return (date(year, month, day).weekday() + 1) % 7

def allocate_workshare_by_day(*,
                              total_units: float,
                              year: int,
                              month: int,
                              day: int,
                              deterministic_target_pct: float = 90.0) -> dict:
    """Allocate workshare and weight the LLM portion by the day‑of‑week factor."""
    day_of_week = doomsday(year, month, day)
    base = allocate_workshare(total_units=total_units,
                              deterministic_target_pct=deterministic_target_pct)
    # Scale LLM units by the fraction of the week represented by the day.
    base["day_of_week"] = day_of_week
    base["day_of_week_llm_units"] = _pct(base["llm_units"] * (day_of_week / 7.0))
    return base

# ----------------------------------------------------------------------
# Hybrid core (new)
# ----------------------------------------------------------------------
def optimal_deterministic_pct(temp_k: float,
                              params: SchoolfieldParams = SchoolfieldParams(),
                              min_pct: float = 50.0,
                              max_pct: float = 95.0) -> float:
    """
    Map a temperature to an *optimal* deterministic target percentage.
    The mapping uses a normalized Schoolfield rate: higher rates → higher deterministic share.
    """
    rate = developmental_rate(temp_k, params)
    # Normalise rate to [0,1] using the rate at the low and high bounds.
    rate_low = developmental_rate(params.t_low, params)
    rate_high = developmental_rate(params.t_high, params)
    norm = (rate - rate_low) / (rate_high - rate_low + 1e-12)
    norm = np.clip(norm, 0.0, 1.0)
    return min_pct + norm * (max_pct - min_pct)

def hybrid_cost(temp_k: float,
                total_units: float,
                day_of_week: int,
                deterministic_target_pct: float,
                params: SchoolfieldParams = SchoolfieldParams(),
                weights: dict | None = None) -> float:
    """
    Unified cost J(T, p_det) = reconstruction + SSIM + free‑energy + allocation penalties.
    For demonstration, reconstruction_error, SSIM and free‑energy are synthetic but
    retain the mathematical form of the parent A cost terms.
    """
    if weights is None:
        weights = {
            "reconstruction": 1.0,
            "ssim": 1.0,
            "free_energy": 1.0,
            "alloc_det": 1.0,
            "alloc_llm_day": 1.0,
        }

    # ---- Synthetic reconstruction error (quadratic distance from a reference) ----
    # Assume a reference value of 0.5 for a normalized metric.
    recon_metric = random.random()  # placeholder for a real metric
    reconstruction_error = (recon_metric - 0.5) ** 2

    # ---- Synthetic SSIM term (higher is better) ----
    # Model SSIM as a sigmoid of temperature (hotter → more structural similarity).
    ssim = 1.0 / (1.0 + math.exp(-(temp_k - 295.0) / 5.0))

    # ---- Variational free‑energy term (negative log‑likelihood of the rate) ----
    rate = developmental_rate(temp_k, params)
    free_energy = -math.log(rate + 1e-12)

    # ---- Allocation deterministic penalty ----
    p_opt = optimal_deterministic_pct(temp_k, params)
    alloc_det_penalty = (deterministic_target_pct - p_opt) ** 2

    # ---- Allocation LLM‑day penalty ----
    # Desired LLM units for the given day (user can set a target; we use 30% of total as example)
    desired_llm_day = 0.30 * total_units
    llm_units = total_units * (100.0 - deterministic_target_pct) / 100.0
    llm_day_units = llm_units * (day_of_week / 7.0)
    alloc_llm_day_penalty = (llm_day_units - desired_llm_day) ** 2

    # Weighted sum
    J = (weights["reconstruction"] * reconstruction_error +
         weights["ssim"] * (1.0 - ssim) +
         weights["free_energy"] * free_energy +
         weights["alloc_det"] * alloc_det_penalty +
         weights["alloc_llm_day"] * alloc_llm_day_penalty)
    return J

def hybrid_gradient_step(temp_k: float,
                         deterministic_target_pct: float,
                         total_units: float,
                         day_of_week: int,
                         params: SchoolfieldParams,
                         lr: float = 0.1,
                         weights: dict | None = None) -> tuple[float, float]:
    """
    Perform a single gradient descent step on (T, p_det) using finite differences.
    Returns updated (temp_k, deterministic_target_pct).
    """
    eps = 1e-4

    # Base cost
    base = hybrid_cost(temp_k, total_units, day_of_week,
                       deterministic_target_pct, params, weights)

    # Gradient w.r.t temperature
    cost_plus = hybrid_cost(temp_k + eps, total_units, day_of_week,
                            deterministic_target_pct, params, weights)
    grad_T = (cost_plus - base) / eps

    # Gradient w.r.t deterministic percentage
    cost_plus = hybrid_cost(temp_k, total_units, day_of_week,
                            deterministic_target_pct + eps, params, weights)
    grad_p = (cost_plus - base) / eps

    # Update (simple SGD with clipping to feasible ranges)
    new_T = temp_k - lr * grad_T
    new_T = max(params.t_low, min(params.t_high, new_T))

    new_p = deterministic_target_pct - lr * grad_p
    new_p = max(0.0, min(100.0, new_p))

    return new_T, new_p

def hybrid_optimize(initial_temp_c: float,
                    total_units: float,
                    year: int,
                    month: int,
                    day: int,
                    deterministic_target_pct: float = 90.0,
                    iterations: int = 50,
                    lr: float = 0.5,
                    params: SchoolfieldParams = SchoolfieldParams(),
                    weights: dict | None = None) -> dict:
    """
    Jointly optimise temperature (in Celsius) and deterministic target percentage.
    Returns a dictionary summarising the final state and the allocation for the given day.
    """
    temp_k = c_to_k(initial_temp_c)
    day_of_week = doomsday(year, month, day)

    for _ in range(iterations):
        temp_k, deterministic_target_pct = hybrid_gradient_step(
            temp_k,
            deterministic_target_pct,
            total_units,
            day_of_week,
            params,
            lr=lr,
            weights=weights,
        )

    final_temp_c = temp_k - 273.15
    allocation = allocate_workshare_by_day(
        total_units=total_units,
        year=year,
        month=month,
        day=day,
        deterministic_target_pct=deterministic_target_pct,
    )
    result = {
        "final_temperature_c": round(final_temp_c, 3),
        "final_deterministic_pct": round(deterministic_target_pct, 3),
        "day_of_week": day_of_week,
        "final_cost": round(hybrid_cost(temp_k, total_units, day_of_week,
                                        deterministic_target_pct, params, weights), 6),
        "allocation": allocation,
    }
    return result

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example parameters
    init_temp_c = 22.0          # starting guess (°C)
    total_units = 1_000.0       # abstract work units
    year, month, day = 2026, 5, 29

    # Run optimisation
    outcome = hybrid_optimize(
        initial_temp_c=init_temp_c,
        total_units=total_units,
        year=year,
        month=month,
        day=day,
        deterministic_target_pct=85.0,
        iterations=30,
        lr=0.3,
    )

    # Pretty‑print the result (no external libraries)
    print(json.dumps(outcome, indent=2))