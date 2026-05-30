# DARWIN HAMMER — match 67, survivor 2
# gen: 2
# parent_a: hybrid_workshare_allocator_doomsday_calendar_m14_s0.py (gen1)
# parent_b: liquid_time_constant.py (gen0)
# born: 2026-05-29T23:24:20Z

"""
Hybrid Allocation-LTC Module
===========================

Parents:
- **workshare_allocator / doomsday_calendar** (PARENT ALGORITHM A)
- **Liquid Time‑Constant Networks** (PARENT ALGORITHM B)

Mathematical Bridge
-------------------
Algorithm A distributes a deterministic portion of *total_units* and
shares the remaining *LLM* portion among a set of groups.  The share can be
further modulated by the day‑of‑week (0‑6) obtained from the Doomsday
calendar.

Algorithm B evolves a hidden state **x(t)** whose *effective* time constant

    τ_sys(t) = τ / (1 + τ·f(x(t), I(t)))

depends on the current input **I(t)** via a learned gating function *f*.

The hybrid treats each calendar day as a discrete time step *t*.  The
day‑of‑week (scaled to [0, 1]) is fed to the LTC as the external input
**I(t)**.  The resulting scalar *τ_sys(t)* is used to scale the LLM
allocation for that day:

    llm_units(t) = llm_base · (τ_sys(t) / τ_max)

where *llm_base* is the LLM portion defined by the deterministic target
percentage and *τ_max* is the maximum τ_sys observed over the processed
date sequence.  This creates a mathematically coupled system in which the
temporal dynamics of the LTC directly reshape the resource allocation
schedule.

The module therefore fuses:
1. The deterministic/LLM split and group‑wise division of Algorithm A.
2. The input‑dependent effective time constant of Algorithm B as a
   multiplicative factor on the LLM share of each day.

The three public functions demonstrate the hybrid operation:
- `init_hybrid_ltc` – initialise LTC parameters for a single‑dimensional
  day‑of‑week input.
- `hybrid_allocate_by_dates` – compute per‑day, per‑group allocations
  using the LTC‑modulated LLM share.
- `summarize_hybrid_savings` – aggregate baseline vs. LTC‑modulated
  allocations and report a savings percentage.
"""

import math
import random
import sys
from datetime import date
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Constants & Helpers (from Parent A)
# ---------------------------------------------------------------------------

GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def _deterministic_split(total_units: float, deterministic_target_pct: float) -> tuple[float, float]:
    """Return (deterministic_units, llm_units) according to the target percentage."""
    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    return _pct(deterministic_units), _pct(llm_units)

def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where Monday=0 … Sunday=6 (same as Python's weekday)."""
    return date(year, month, day).weekday()

# ---------------------------------------------------------------------------
# Liquid Time‑Constant Network Primitives (from Parent B)
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element‑wise sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def ltc_f(x: np.ndarray, I: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gating function f(x, I) = σ(W·[x;I] + b)."""
    concat = np.concatenate([x, I], axis=0)
    return sigmoid(W @ concat + b)

def ltc_step(
    x: np.ndarray,
    I: np.ndarray,
    params: dict,
    dt: float = 0.1,
) -> tuple[np.ndarray, float]:
    """One Euler integration step of the LTC ODE."""
    W = params["W"]
    b = params["b"]
    tau = float(params["tau"])
    A = params["A"]

    f_val = ltc_f(x, I, W, b)               # (hidden_dim,)
    dx_dt = -(1.0 / tau + f_val) * x + f_val * A
    x_new = x + dt * dx_dt

    tau_sys_vec = tau / (1.0 + tau * f_val)  # (hidden_dim,)
    tau_sys = float(np.mean(tau_sys_vec))

    return x_new, tau_sys

def ltc_forward(I_seq: np.ndarray, params: dict, x0: np.ndarray | None = None, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Run the LTC over a full input sequence."""
    T, _ = I_seq.shape
    hidden_dim = params["A"].shape[0]
    x = np.zeros(hidden_dim) if x0 is None else np.array(x0, dtype=float)

    X = np.empty((T, hidden_dim))
    tau_sys_seq = np.empty(T)

    for t in range(T):
        x, tau_sys = ltc_step(x, I_seq[t], params, dt=dt)
        X[t] = x
        tau_sys_seq[t] = tau_sys

    return X, tau_sys_seq

def init_ltc(hidden_dim: int, input_dim: int, tau: float = 1.0, seed: int = 0) -> dict:
    """Randomly initialise LTC parameters."""
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.5, size=(hidden_dim, hidden_dim + input_dim))
    b = rng.normal(scale=0.1, size=(hidden_dim,))
    A = rng.normal(scale=0.5, size=(hidden_dim,))
    return {"W": W, "b": b, "tau": float(tau), "A": A}

# ---------------------------------------------------------------------------
# Hybrid Functions
# ---------------------------------------------------------------------------

def init_hybrid_ltc(
    hidden_dim: int = 4,
    input_dim: int = 1,
    tau: float = 1.0,
    seed: int = 0,
) -> dict:
    """
    Initialise LTC parameters tailored for the hybrid system.

    The LTC expects a single scalar input: the normalized day‑of‑week
    (0 → Monday, 6 → Sunday) divided by 6 to lie in [0, 1].
    """
    return init_ltc(hidden_dim=hidden_dim, input_dim=input_dim, tau=tau, seed=seed)

def hybrid_allocate_by_dates(
    total_units: float,
    dates: list[tuple[int, int, int]],
    deterministic_target_pct: float = 90.0,
    params_ltc: dict | None = None,
    dt: float = 0.1,
) -> list[dict]:
    """
    Compute per‑day, per‑group allocations where the LLM share is modulated
    by the LTC's effective time constant.

    Parameters
    ----------
    total_units : total resource pool for each day (float, >0)
    dates       : list of (year, month, day) tuples
    deterministic_target_pct : percentage of total_units kept deterministic
    params_ltc  : LTC parameters; if None, defaults are created
    dt          : Euler integration step size for the LTC

    Returns
    -------
    allocations : list of dicts, one per input date, each containing:
        - "date": (y,m,d)
        - "day_of_week": 0‑6
        - "tau_sys": effective time constant for that day
        - "deterministic_units"
        - "llm_units_modulated"
        - "lanes": list of per‑group dicts with allocated LLM units
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0 <= deterministic_target_pct <= 100):
        raise ValueError("deterministic_target_pct must be in [0,100]")
    if not dates:
        raise ValueError("at least one date required")

    # 1️⃣ Split deterministic / LLM portion (same for every day)
    deterministic_units, llm_base = _deterministic_split(total_units, deterministic_target_pct)

    # 2️⃣ Prepare LTC input sequence (normalized day_of_week)
    dow_list = [doomsday(y, m, d) for (y, m, d) in dates]
    I_seq = np.array([[dow / 6.0] for dow in dow_list], dtype=float)  # shape (T,1)

    # 3️⃣ Initialise LTC if not supplied
    if params_ltc is None:
        params_ltc = init_hybrid_ltc(hidden_dim=4, input_dim=1, tau=1.0, seed=42)

    # 4️⃣ Run LTC forward to obtain τ_sys per day
    _, tau_sys_seq = ltc_forward(I_seq, params_ltc, dt=dt)

    tau_max = float(np.max(tau_sys_seq)) if tau_sys_seq.size > 0 else 1.0

    allocations = []
    for idx, (y, m, d) in enumerate(dates):
        day_of_week = dow_list[idx]
        tau_sys = tau_sys_seq[idx]

        # Scale LLM units by relative τ_sys (preserve total deterministic part)
        llm_units_mod = llm_base * (tau_sys / tau_max)

        per_group = llm_units_mod / len(GROUPS)

        lanes = [
            {
                "group": grp,
                "llm_units": _pct(per_group),
                "llm_share_pct": _pct(100.0 / len(GROUPS)),
                "proof_required": True,
            }
            for grp in GROUPS
        ]

        allocations.append({
            "date": (y, m, d),
            "day_of_week": day_of_week,
            "tau_sys": _pct(tau_sys),
            "deterministic_units": deterministic_units,
            "llm_units_modulated": _pct(llm_units_mod),
            "lanes": lanes,
        })

    return allocations

def summarize_hybrid_savings(
    dates: list[tuple[int, int, int]],
    total_units: float,
    deterministic_target_pct: float = 90.0,
    params_ltc: dict | None = None,
    dt: float = 0.1,
) -> dict:
    """
    Produce a high‑level summary comparing baseline (no LTC modulation)
    with the hybrid allocation.

    Returns a dictionary with:
        - "baseline_llm_units": LLM units per day without modulation
        - "average_modulated_llm_units": mean of LTC‑scaled LLM units
        - "overall_savings_pct": percentage reduction in LLM usage
        - "average_tau_sys": mean effective time constant
    """
    # Baseline LLM portion (same each day)
    _, llm_baseline = _deterministic_split(total_units, deterministic_target_pct)

    # Hybrid allocations
    hybrid_allocs = hybrid_allocate_by_dates(
        total_units=total_units,
        dates=dates,
        deterministic_target_pct=deterministic_target_pct,
        params_ltc=params_ltc,
        dt=dt,
    )

    modulated_llm_vals = [alloc["llm_units_modulated"] for alloc in hybrid_allocs]
    avg_modulated = float(np.mean(modulated_llm_vals)) if modulated_llm_vals else 0.0
    avg_tau = float(np.mean([alloc["tau_sys"] for alloc in hybrid_allocs])) if hybrid_allocs else 0.0

    savings_pct = _pct((llm_baseline - avg_modulated) / llm_baseline * 100.0) if llm_baseline != 0 else 0.0

    return {
        "baseline_llm_units": _pct(llm_baseline),
        "average_modulated_llm_units": _pct(avg_modulated),
        "overall_savings_pct": savings_pct,
        "average_tau_sys": _pct(avg_tau),
    }

# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example dates spanning a week
    sample_dates = [
        (2024, 1, 1),  # Monday
        (2024, 1, 2),
        (2024, 1, 3),
        (2024, 1, 4),
        (2024, 1, 5),
        (2024, 1, 6),
        (2024, 1, 7),  # Sunday
    ]

    total = 100.0
    det_pct = 85.0

    # Run hybrid allocation
    allocs = hybrid_allocate_by_dates(
        total_units=total,
        dates=sample_dates,
        deterministic_target_pct=det_pct,
    )
    for a in allocs:
        print(f"Date {a['date']} (dow={a['day_of_week']}), τ_sys={a['tau_sys']}, "
              f"LLM_mod={a['llm_units_modulated']}, lanes={a['lanes'][0]['llm_units']} each")

    # Summarize savings
    summary = summarize_hybrid_savings(
        dates=sample_dates,
        total_units=total,
        deterministic_target_pct=det_pct,
    )
    print("\nSummary:", summary)