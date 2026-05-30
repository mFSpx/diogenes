# DARWIN HAMMER — match 67, survivor 0
# gen: 2
# parent_a: hybrid_workshare_allocator_doomsday_calendar_m14_s0.py (gen1)
# parent_b: liquid_time_constant.py (gen0)
# born: 2026-05-29T23:24:20Z

"""
This module combines the concepts of workshare allocation and liquid time-constant networks.
The mathematical bridge between the two structures is the idea of adaptive allocation, where 
the workshare allocator distributes work units based on the day of the week, and the liquid time-constant 
network adapts its time constant based on the input. Here, we combine the two by allocating work units 
based on the day of the week, which is determined by the doomsday calendar algorithm, and using the 
liquid time-constant network to adapt the allocation based on the input.
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
    concat = np.concatenate([x, I], axis=0)
    return sigmoid(W @ concat + b)

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

def doomsday(year: int, month: int, day: int) -> int:
    return (date(year, month, day).weekday() + 1) % 7

def allocate_workshare_by_day(*, total_units: float, year: int, month: int, day: int, deterministic_target_pct: float = 90.0) -> dict[str, float]:
    day_of_week = doomsday(year, month, day)
    allocation = allocate_workshare(total_units=total_units, deterministic_target_pct=deterministic_target_pct)
    allocation["day_of_week"] = day_of_week
    allocation["day_of_week_llm_units"] = allocation["llm_units"] * (day_of_week / 7)
    return allocation

def ltc_step(
    x: np.ndarray,
    I: np.ndarray,
    params: dict,
    dt: float = 0.1,
) -> tuple[np.ndarray, float]:
    W = params["W"]
    b = params["b"]
    tau = float(params["tau"])
    A = params["A"]

    f_val = ltc_f(x, I, W, b)

    dx_dt = -(1.0 / tau + f_val) * x + f_val * A

    x_new = x + dt * dx_dt

    tau_sys_vec = tau / (1.0 + tau * f_val)
    tau_sys = float(np.mean(tau_sys_vec))

    return x_new, tau_sys

def init_ltc(
    hidden_dim: int,
    input_dim: int,
    tau: float = 1.0,
    seed: int = 0,
) -> dict:
    np.random.seed(seed)
    W = np.random.rand(hidden_dim, hidden_dim + input_dim)
    b = np.random.rand(hidden_dim)
    A = np.random.rand(hidden_dim)
    return {
        "W": W,
        "b": b,
        "tau": tau,
        "A": A,
    }

def hybrid_allocate_workshare(
    *, 
    total_units: float, 
    year: int, 
    month: int, 
    day: int, 
    deterministic_target_pct: float = 90.0, 
    hidden_dim: int = 10, 
    input_dim: int = 5,
) -> dict[str, float]:
    day_of_week = doomsday(year, month, day)
    allocation = allocate_workshare(total_units=total_units, deterministic_target_pct=deterministic_target_pct)
    params = init_ltc(hidden_dim, input_dim)
    x = np.random.rand(hidden_dim)
    I = np.random.rand(input_dim)
    x_new, tau_sys = ltc_step(x, I, params)
    allocation["day_of_week"] = day_of_week
    allocation["day_of_week_llm_units"] = allocation["llm_units"] * (day_of_week / 7)
    allocation["effective_time_constant"] = tau_sys
    return allocation

def summarize_savings_by_day(*, total_units: float, year: int, month: int, day: int, deterministic_target_pct: float = 90.0) -> dict[str, float]:
    allocation = allocate_workshare_by_day(total_units=total_units, year=year, month=month, day=day, deterministic_target_pct=deterministic_target_pct)
    return {
        "day_of_week": allocation["day_of_week"],
        "day_of_week_llm_units": allocation["day_of_week_llm_units"],
        "baseline_llm_units": allocation["llm_units"],
        "token_savings_pct": _pct((allocation["total_units"] - allocation["llm_units"]) / allocation["total_units"] * 100.0),
    }

def simulate_hybrid_system(*, total_units: float, year: int, month: int, day: int, deterministic_target_pct: float = 90.0, hidden_dim: int = 10, input_dim: int = 5) -> tuple[dict[str, float], dict[str, float]]:
    allocation = hybrid_allocate_workshare(total_units=total_units, year=year, month=month, day=day, deterministic_target_pct=deterministic_target_pct, hidden_dim=hidden_dim, input_dim=input_dim)
    savings = summarize_savings_by_day(total_units=total_units, year=year, month=month, day=day, deterministic_target_pct=deterministic_target_pct)
    return allocation, savings

if __name__ == "__main__":
    allocation = allocate_workshare(total_units=100.0)
    print(allocation)
    day_allocation = allocate_workshare_by_day(total_units=100.0, year=2024, month=1, day=1)
    print(day_allocation)
    savings = summarize_savings_by_day(total_units=100.0, year=2024, month=1, day=1)
    print(savings)
    hybrid_allocation, hybrid_savings = simulate_hybrid_system(total_units=100.0, year=2024, month=1, day=1)
    print(hybrid_allocation)
    print(hybrid_savings)