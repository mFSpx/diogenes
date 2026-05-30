# DARWIN HAMMER — match 5613, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_hybrid_m2104_s1.py (gen5)
# parent_b: hybrid_hybrid_gliner_zero_s_hybrid_doomsday_cale_m49_s2.py (gen2)
# born: 2026-05-30T00:03:26Z

"""Hybrid Algorithm combining Schoolfield developmental dynamics (Parent A) with
statistical inequality (Gini) and temporal weighting (weekday) from Parent B.

Mathematical Bridge
-------------------
- **Temperature‑driven developmental rate** `ρ(T)` from the Schoolfield model
  (Parent A) provides a scalar factor that modulates any downstream risk metric.
- **Keyword frequency distribution** extracted from a text yields a non‑negative
  vector `k = [k₁,…,kₙ]`.  The Gini coefficient `G(k)` (Parent B) quantifies the
  inequality of evidence vs. planning tokens, acting as a *diversity penalty*.
- **Weekday weighting** `w(d)` (Parent B) assigns a multiplicative factor based on
  the calendar day (higher on weekends, lower on weekdays).

The fused risk score for a given `text`, temperature `T` (°C) and `date` is


R = ρ(T) * (1 + α·E) * (1 + β·w(d)) * (1 - γ·G(k))


where  
`E` = evidence‑planning balance (from regex counts in Parent A),  
`α,β,γ` are tunable hyper‑parameters (default 0.5, 0.2, 0.8).

The code below implements the three core functions required for this hybrid
system and provides a small smoke test.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import re
import hashlib
from datetime import datetime, timezone
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

def developmental_rate(temp_k: float,
                       params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield model (Parent A). Returns the temperature‑dependent developmental rate.
    """
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    num = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    denom = (1.0
             + math.exp((params.delta_h_low / params.r_cal) *
                        ((1.0 / params.t_low) - (1.0 / temp_k)))
             + math.exp((params.delta_h_high / params.r_cal) *
                        ((1.0 / params.t_high) - (1.0 / temp_k))))
    return num / denom

def _keyword_counts(text: str) -> Tuple[int, int]:
    """Count evidence‑related and planning‑related tokens (Parent A)."""
    evidence_re = re.compile(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|"
        r"receipt|sha256|screenshot|record|log|document|proof|fact|facts|check|"
        r"checked|audit)\b", re.I)
    planning_re = re.compile(
        r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|"
        r"prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|"
        r"test|smoke)\b", re.I)
    ev = len(evidence_re.findall(text))
    pl = len(planning_re.findall(text))
    return ev, pl

def compute_cognitive_balance(text: str) -> float:
    """
    Returns a signed balance `E = (evidence - planning) / (evidence + planning + ε)`.
    Positive values indicate evidence‑heavy text.
    """
    ev, pl = _keyword_counts(text)
    eps = 1e-9
    return (ev - pl) / (ev + pl + eps)

# ----------------------------------------------------------------------
# Parent B – Gini coefficient, weekday weighting and hashing utilities
# ----------------------------------------------------------------------

def weekday_factor(date: datetime) -> float:
    """
    Weekday weighting factor.
    Monday‑Friday → 0.0, Saturday → 0.2, Sunday → 0.4 (simple example).
    """
    wd = date.weekday()  # Monday=0 … Sunday=6
    if wd >= 5:
        return 0.2 * (wd - 4)  # 0.2 for Saturday, 0.4 for Sunday
    return 0.0

def gini_coefficient(values: np.ndarray) -> float:
    """
    Gini coefficient of a 1‑D non‑negative array (Parent B).
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    x = np.sort(values.astype(np.float64))
    if x.size == 0 or np.isclose(x.sum(), 0.0):
        return 0.0
    if np.any(x < 0):
        raise ValueError("values must be non‑negative")
    n = x.size
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))

def text_keyword_vector(text: str,
                        vocab: List[str]) -> np.ndarray:
    """
    Build a frequency vector over `vocab` from `text`.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    freq = np.zeros(len(vocab), dtype=np.int64)
    vocab_index = {w: i for i, w in enumerate(vocab)}
    for w in words:
        if w in vocab_index:
            freq[vocab_index[w]] += 1
    return freq

def sha256_text(text: str) -> str:
    """SHA‑256 hash of the supplied text."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

# ----------------------------------------------------------------------
# Hybrid core – three functions demonstrating the fused operation
# ----------------------------------------------------------------------

def hybrid_developmental_risk(text: str,
                              temp_c: float,
                              date: datetime,
                              vocab: List[str] | None = None,
                              alpha: float = 0.5,
                              beta: float = 0.2,
                              gamma: float = 0.8) -> float:
    """
    Compute the hybrid risk score R as described in the module docstring.
    """
    # 1. Temperature factor via Schoolfield model
    temp_k = c_to_k(temp_c)
    rho = developmental_rate(temp_k)

    # 2. Cognitive evidence‑planning balance
    balance = compute_cognitive_balance(text)  # E in [‑1, 1]
    balance_factor = 1.0 + alpha * balance

    # 3. Weekday weighting
    w = weekday_factor(date)
    weekday_factor_val = 1.0 + beta * w

    # 4. Gini penalty based on keyword frequency distribution
    if vocab is None:
        vocab = ["evidence", "verify", "confirm", "plan", "checklist", "schedule"]
    freq_vec = text_keyword_vector(text, vocab)
    gini = gini_coefficient(freq_vec.astype(np.float64))
    gini_factor = 1.0 - gamma * gini  # reduces risk when distribution is uniform

    # Composite risk
    risk = rho * balance_factor * weekday_factor_val * gini_factor
    return max(risk, 0.0)  # risk cannot be negative

def generate_hybrid_span(text: str,
                         temp_c: float,
                         date: datetime,
                         label: str = "HybridRisk") -> Tuple[Dict[str, Any], str]:
    """
    Produce a dictionary mimicking Parent B's `Span` enriched with the hybrid risk.
    Returns (span_dict, sha256_hash_of_text).
    """
    start = 0
    end = len(text)
    risk = hybrid_developmental_risk(text, temp_c, date)
    score = min(risk / 10.0, 1.0)  # normalise to [0,1] for illustration
    span = {
        "start": start,
        "end": end,
        "text": text,
        "label": label,
        "score": score,
        "backend": "hybrid_engine",
        "risk": risk,
    }
    h = sha256_text(text)
    return span, h

def batch_hybrid_analysis(samples: List[Tuple[str, float, datetime]]) -> List[Dict[str, Any]]:
    """
    Process a batch of (text, temperature°C, date) tuples and return a list of
    hybrid span dictionaries.
    """
    results = []
    for text, temp_c, date in samples:
        span, _ = generate_hybrid_span(text, temp_c, date)
        results.append(span)
    return results

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple demonstration that runs without external data
    demo_text = (
        "The plan includes several steps. We have verified the evidence and "
        "recorded the logs. A checklist will be used to ensure completeness."
    )
    demo_temp_c = 25.0  # moderate temperature
    demo_date = datetime.now(timezone.utc)

    risk = hybrid_developmental_risk(demo_text, demo_temp_c, demo_date)
    print(f"Hybrid risk score: {risk:.4f}")

    span, h = generate_hybrid_span(demo_text, demo_temp_c, demo_date)
    print("Generated Span:", span)
    print("SHA‑256 of text:", h)

    batch = [
        (demo_text, demo_temp_c, demo_date),
        ("Evidence confirmed. No planning needed.", 30.0, demo_date.replace(day=demo_date.day - 1)),
        ("Schedule the test and verify the source.", 15.0, demo_date.replace(day=demo_date.day - 2)),
    ]
    batch_results = batch_hybrid_analysis(batch)
    print("\nBatch results:")
    for i, res in enumerate(batch_results, 1):
        print(f"{i}: score={res['score']:.3f}, risk={res['risk']:.3f}")