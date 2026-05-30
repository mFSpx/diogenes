# DARWIN HAMMER — match 67, survivor 1
# gen: 2
# parent_a: hybrid_workshare_allocator_doomsday_calendar_m14_s0.py (gen1)
# parent_b: liquid_time_constant.py (gen0)
# born: 2026-05-29T23:24:20Z

"""
This module is a fusion of hybrid_workshare_allocator_doomsday_calendar_m14_s0.py and liquid_time_constant.py.
The mathematical bridge between the two structures is the concept of time-dependent allocation, 
where the workshare allocator distributes work units among different groups based on the day of the week, 
and the liquid time constant network allocates its state based on the input and the current state.
Here, we combine the two by allocating work units based on the liquid time constant network's state, 
which is determined by the day of the week and the current input.
"""

import numpy as np
import math
import random
import sys
from datetime import date
from pathlib import Path

GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    return round(float(value), 6)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise sigmoid σ(x) = 1 / (1 + exp(-x))."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def ltc_f(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Network function f(x, I, t, θ) = σ(W @ [x; I] + b).

    Parameters
    ----------
    x : shape (hidden_dim,)   — current hidden state
    I : shape (input_dim,)    — current external input
    W : shape (hidden_dim, hidden_dim + input_dim)
    b : shape (hidden_dim,)

    Returns
    -------
    f_val : shape (hidden_dim,)   values in (0, 1)
    """
    concat = np.concatenate([x, I], axis=0)  # (hidden_dim + input_dim,)
    return sigmoid(W @ concat + b)

def doomsday(year: int, month: int, day: int) -> int:
    return (date(year, month, day).weekday() + 1) % 7

def allocate_workshare_by_day(*, total_units: float, year: int, month: int, day: int, deterministic_target_pct: float = 90.0) -> dict[str, float]:
    day_of_week = doomsday(year, month, day)
    allocation = allocate_workshare(total_units=total_units, deterministic_target_pct=deterministic_target_pct)
    allocation["day_of_week"] = day_of_week
    allocation["day_of_week_llm_units"] = allocation["llm_units"] * (day_of_week / 7)
    return allocation

def allocate_workshare(*, total_units: float, deterministic_target_pct: float = 90.0, groups: tuple[str, ...] = GROUPS) -> dict[str, float]:
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

def ltc_step(
    x: np.ndarray,
    I: np.ndarray,
    params: dict,
    dt: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Advance the LTC state by one Euler step.

    ODE:  dx/dt = -[1/τ + f] · x + f · A

    Parameters
    ----------
    x      : shape (hidden_dim,)  — current state
    I      : shape (input_dim,)   — current input
    params : dict with keys
        "W"   : shape (hidden_dim, hidden_dim + input_dim)
        "b"   : shape (hidden_dim,)
        "tau" : float  — base membrane time constant (scalar, shared here)
        "A"   : shape (hidden_dim,)  — attractor target
    dt     : Euler integration step size

    Returns
    -------
    x_new   : shape (hidden_dim,)  — updated state
    tau_sys : float  — mean effective liquid time constant over neurons
    """
    W = params["W"]
    b = params["b"]
    tau = float(params["tau"])
    A = params["A"]

    f_val = ltc_f(x, I, W, b)                          # (hidden_dim,)

    dx_dt = -(1.0 / tau + f_val) * x + f_val * A       # (hidden_dim,)

    x_new = x + dt * dx_dt                              # Euler step

    tau_sys_vec = tau / (1.0 + tau * f_val)             # (hidden_dim,)
    tau_sys = float(np.mean(tau_sys_vec))

    return x_new, tau_sys

def init_ltc(
    hidden_dim: int,
    input_dim: int,
    tau: float = 1.0,
    seed: int = 0,
) -> dict:
    """Initialise LTC parameters.

    Parameters
    ----------
    hidden_dim : number of hidden (neural) units
    input_dim  : dimensionality of the external input
    tau        : base membrane time constant (positive scalar)
    seed       : numpy RNG seed for reproducibility

    Returns
    -------
    params : dict with keys "W", "b", "tau", "A"
    """
    np.random.seed(seed)
    W = np.random.rand(hidden_dim, hidden_dim + input_dim)
    b = np.random.rand(hidden_dim)
    A = np.random.rand(hidden_dim)
    return {"W": W, "b": b, "tau": tau, "A": A}

def ltc_workflow(
    total_units: float,
    year: int,
    month: int,
    day: int,
    deterministic_target_pct: float = 90.0,
    hidden_dim: int = 10,
    input_dim: int = 5,
    tau: float = 1.0,
    seed: int = 0,
    dt: float = 0.1,
    num_steps: int = 10,
) -> None:
    params = init_ltc(hidden_dim, input_dim, tau, seed)
    x = np.zeros(hidden_dim)
    I = np.random.rand(input_dim)
    allocation = allocate_workshare_by_day(total_units=total_units, year=year, month=month, day=day, deterministic_target_pct=deterministic_target_pct)
    print(allocation)
    for _ in range(num_steps):
        x, tau_sys = ltc_step(x, I, params, dt=dt)
        print(f"x: {x}, tau_sys: {tau_sys}")

if __name__ == "__main__":
    ltc_workflow(total_units=100.0, year=2024, month=1, day=1)