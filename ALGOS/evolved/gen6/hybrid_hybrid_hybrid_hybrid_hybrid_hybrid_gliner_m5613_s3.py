# DARWIN HAMMER — match 5613, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_hybrid_m2104_s1.py (gen5)
# parent_b: hybrid_hybrid_gliner_zero_s_hybrid_doomsday_cale_m49_s2.py (gen2)
# born: 2026-05-30T00:03:26Z

"""Hybrid algorithm merging:
- Parent A: temperature‑dependent Schoolfield developmental rate and text‑based cognitive risk.
- Parent B: vectorised weekday computation (Sakamoto) and Gini inequality coefficient.

Mathematical bridge:
We treat a series of daily temperatures as a vector **r** = developmental_rate(T_k). The same
days are mapped to weekday indices **w** via Sakamoto's algorithm, allowing a weekday‑weight
vector **α** (e.g., higher weight on weekdays). The inequality of the rate distribution is
captured by the Gini coefficient **G(r)**. The cognitive risk extracted from a text fragment
produces a scalar **R_text**. The final hybrid score combines these quantities:

    S_hybrid = R_text * (1 + G(r)) * α·r̂

where **r̂** is the normalized rate vector and **α·r̂** is a weekday‑weighted average. This
fusion integrates the thermodynamic kinetics of Parent A with the statistical‑temporal tools
of Parent B into a single coherent system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import re
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Schoolfield developmental rate and cognitive risk utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield equation for temperature‑dependent developmental rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = (
        params.rho_25
        * (temp_k / 298.15)
        * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    )
    denominator = (
        1
        + math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
        + math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    )
    return numerator / denominator


def compute_cognitive_risk(
    text: str,
    temp_k: float,
    params: SchoolfieldParams = SchoolfieldParams(),
    learning_rate: float = 0.1,
) -> float:
    """
    Estimate a “cognitive risk” score from textual cues and ambient temperature.
    Evidence‑related words lower risk, planning‑related words raise it.
    The temperature‑scaled developmental rate modulates the magnitude.
    """
    evidence_re = re.compile(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
        r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
        re.I,
    )
    planning_re = re.compile(
        r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
        r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
        re.I,
    )

    evidence_cnt = len(evidence_re.findall(text))
    planning_cnt = len(planning_re.findall(text))

    # Feature vector: [evidence, planning]
    fv = np.array([evidence_cnt, planning_cnt], dtype=np.float64)

    # Positive and negative weight vectors (symmetric for illustration)
    w_pos = np.array([1.0, -1.0])   # evidence reduces risk, planning increases it
    w_neg = np.array([-1.0, 1.0])   # opposite polarity when planning dominates

    # Choose weight set based on which count is larger
    w = w_pos if evidence_cnt >= planning_cnt else w_neg

    # Linear combination
    base_risk = float(w @ fv) * learning_rate

    # Temperature modulation via developmental rate (higher rate ⇒ higher metabolic activity ⇒ higher risk)
    rate = developmental_rate(temp_k, params)
    risk = base_risk * (1.0 + rate)

    return max(risk, 0.0)  # risk cannot be negative


# ----------------------------------------------------------------------
# Parent B – Weekday calculation and Gini coefficient
# ----------------------------------------------------------------------
def weekday_sakamoto(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """
    Vectorised Tomohiko Sakamoto algorithm returning weekday indices
    (0 = Sunday … 6 = Saturday).
    """
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)

    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a 1‑D non‑negative array.
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    x = np.sort(values.astype(np.float64))
    if x.size == 0 or np.isclose(x.sum(), 0.0):
        return 0.0
    if np.any(x < 0):
        raise ValueError("values must be non‑negative")
    n = x.size
    i = np.arange(1, n + 1)
    gini = (2.0 * np.sum(i * x)) / (n * np.sum(x)) - (n + 1) / n
    return float(gini)


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_developmental_profile(
    temps_celsius: List[float],
    dates: List[datetime],
    params: SchoolfieldParams = SchoolfieldParams(),
) -> Dict[str, Any]:
    """
    Given parallel temperature and date lists, compute:
    - Kelvin temperatures
    - Developmental rates r_i
    - Weekday indices w_i
    - Gini coefficient G(r)
    - Weekday‑weight vector α (higher weight for weekdays, lower for weekend)
    Returns a dictionary with all intermediate results.
    """
    if len(temps_celsius) != len(dates):
        raise ValueError("temps_celsius and dates must have the same length")

    # Convert to numpy arrays
    temps_k = np.array([c_to_k(t) for t in temps_celsius], dtype=np.float64)
    rates = np.vectorize(developmental_rate, excluded=[1])(temps_k, params)

    # Extract year/month/day vectors for Sakamoto
    years = np.array([d.year for d in dates], dtype=np.int64)
    months = np.array([d.month for d in dates], dtype=np.int64)
    days = np.array([d.day for d in dates], dtype=np.int64)

    weekdays = weekday_sakamoto(years, months, days)  # 0=Sun … 6=Sat

    # Weekday weighting: Mon‑Fri = 1.0, Sat‑Sun = 0.5
    weekday_weights = np.where(weekdays <= 5, 1.0, 0.5)

    gini = gini_coefficient(rates)

    profile = {
        "temps_k": temps_k,
        "rates": rates,
        "weekdays": weekdays,
        "weekday_weights": weekday_weights,
        "gini": gini,
    }
    return profile


def hybrid_cognitive_risk(
    text: str,
    temps_celsius: List[float],
    dates: List[datetime],
    params: SchoolfieldParams = SchoolfieldParams(),
    learning_rate: float = 0.1,
) -> float:
    """
    Compute a risk score that blends textual cognitive risk with the
    temperature‑driven developmental profile.
    """
    profile = hybrid_developmental_profile(temps_celsius, dates, params)

    # Normalise rates to obtain a probability‑like vector
    rates = profile["rates"]
    if rates.sum() == 0:
        norm_rates = np.zeros_like(rates)
    else:
        norm_rates = rates / rates.sum()

    # Weekday‑weighted average of normalised rates
    weighted_avg_rate = float(np.dot(profile["weekday_weights"], norm_rates))

    # Base textual risk at the *mean* temperature (more stable than per‑day)
    mean_temp_k = float(np.mean(profile["temps_k"]))
    base_risk = compute_cognitive_risk(text, mean_temp_k, params, learning_rate)

    # Fuse: increase risk proportionally to inequality (Gini) and to the weekday‑weighted average
    hybrid_score = base_risk * (1.0 + profile["gini"]) * (1.0 + weighted_avg_rate)

    return hybrid_score


def hybrid_summary(
    text: str,
    temps_celsius: List[float],
    dates: List[datetime],
) -> Dict[str, Any]:
    """
    Produce a human‑readable summary dictionary containing all intermediate
    quantities and the final hybrid score.
    """
    profile = hybrid_developmental_profile(temps_celsius, dates)
    score = hybrid_cognitive_risk(text, temps_celsius, dates)

    summary = {
        "text_snippet": text[:100],
        "mean_temperature_K": float(np.mean(profile["temps_k"])),
        "mean_developmental_rate": float(np.mean(profile["rates"])),
        "gini_developmental_rate": profile["gini"],
        "weekday_weighted_rate": float(np.dot(profile["weekday_weights"], profile["rates"]) / profile["weekday_weights"].sum()),
        "hybrid_score": score,
    }
    return summary


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example temperatures (°C) for a week
    temps_c = [15.0, 17.2, 20.5, 22.0, 19.3, 16.8, 14.1]

    # Corresponding dates (UTC)
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    dates = [today - np.timedelta64(i, "D") for i in range(6, -1, -1)]  # 7 days back to today

    # Sample text containing evidence and planning cues
    sample_text = (
        "The team verified the source and logged the proof. "
        "We need to plan the next steps, create a checklist, and schedule a roadmap."
    )

    # Run hybrid functions
    profile = hybrid_developmental_profile(temps_c, dates)
    print("Developmental rates:", profile["rates"])
    print("Weekday indices:", profile["weekdays"])
    print("Gini coefficient:", profile["gini"])

    risk = hybrid_cognitive_risk(sample_text, temps_c, dates)
    print("Hybrid cognitive risk score:", risk)

    summary = hybrid_summary(sample_text, temps_c, dates)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")