# DARWIN HAMMER — match 2568, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_caputo_hybrid_fisher_locali_m637_s1.py (gen4)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_worksh_m146_s1.py (gen3)
# born: 2026-05-29T23:42:56Z

"""Hybrid Fisher‑Caputo Chrono Allocation
====================================

This module fuses the *Hybrid Caputo‑Geometric Minimum‑Cost Tree* (parent A) 
with the *Hybrid Fisher‑Chrono* algorithm (parent B).  

Both parents rely on a Gaussian beam model:

* In the Chrono‑Caputo part the Gaussian value is used as a weight for the 
  Caputo fractional‑derivative kernel.
* In the Fisher‑Chrono part the Gaussian beam’s derivative yields the Fisher 
  information score.

The **mathematical bridge** is therefore the Gaussian beam `g(θ)`.  
We reuse `g(θ)` to (i) weight the Caputo kernel `φ(t;α)`, (ii) build the Fisher 
score `I(θ)`, and (iii) modulate the weekday‑allocation vector.  The final 
allocation for a set of groups is obtained by element‑wise multiplication of
the three contributions followed by stochastic normalisation.

The implementation provides three core functions that demonstrate the hybrid
operation:
    * ``caputo_weights`` – Lanczos‑approximated Caputo kernel.
    * ``fisher_score``   – Fisher information derived from the Gaussian beam.
    * ``hybrid_fisher_caputo_allocation`` – unified allocation across groups.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, date, timezone

import numpy as np

# ----------------------------------------------------------------------
# Lanczos approximation – needed for the Caputo kernel (parent A)
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of Γ(z) for z>0."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    z -= 1.0
    x = _LANCZOS_C / (z + np.arange(_LANCZOS_G) + 1.0)
    # the denominator of the rational approximation
    series = np.sum(x) + 1.0
    t = z + _LANCZOS_G - 0.5
    return math.sqrt(2 * math.pi) * t**(z + 0.5) * math.exp(-t) * series

def caputo_weights(t: float, alpha: float) -> float:
    """Caputo fractional derivative kernel φ(t;α) ≈ t^{‑α} / Γ(1‑α)."""
    if t <= 0:
        raise ValueError("time argument must be positive")
    return t ** (-alpha) / gamma_lanczos(1.0 - alpha)

# ----------------------------------------------------------------------
# Gaussian beam – common to both parents
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Standard Gaussian beam."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

# ----------------------------------------------------------------------
# Fisher information – derived from the Gaussian beam (parent B)
# ----------------------------------------------------------------------
def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Weekday weight vector – sinusoidal rotation (parent B)
# ----------------------------------------------------------------------
def weekday_weight_vector(dow: int, groups: int) -> np.ndarray:
    """Row‑stochastic vector that rotates sinusoidally with the day of week."""
    base_angles = np.arange(groups) * (2.0 * math.pi) / groups
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    return (raw / raw.sum()).astype(np.float64)

def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where Monday=0, Sunday=6 (mod 7)."""
    return (date(year, month, day).weekday() + 1) % 7

# ----------------------------------------------------------------------
# Hybrid Chrono‑Caputo scoring for a list of timestamped candidates
# ----------------------------------------------------------------------
def hybrid_chrono_caputo(
    candidates: list[dict[str, str]],
    center: float,
    width: float,
    alpha: float
) -> np.ndarray:
    """
    For each candidate compute:
        score = GaussianBeam(timestamp) * CaputoWeight(timestamp, α)

    Returns a NumPy array of scores.
    """
    scores = []
    for cand in candidates:
        ts = datetime.fromisoformat(cand["timestamp"]).timestamp()
        g = gaussian_beam(ts, center, width)
        c = caputo_weights(ts, alpha)
        scores.append(g * c)
    return np.array(scores, dtype=np.float64)

# ----------------------------------------------------------------------
# Unified allocation that merges Fisher, Caputo and weekday weighting
# ----------------------------------------------------------------------
def hybrid_fisher_caputo_allocation(
    total_units: float,
    on_date: date,
    alpha: float,
    width: float = 1.0,
    center: float = 0.0,
    groups: int = 4
) -> dict:
    """
    Allocate ``total_units`` across ``groups`` using a product of three
    components:

        * weekday_weight_vector   – sinusoidal day‑of‑week modulation,
        * fisher_score            – information density per group,
        * caputo_weights           – fractional‑derivative kernel evaluated at
          a synthetic “angle” equal to the group index.

    The final allocation vector is normalised to sum to ``total_units``.
    Returns a mapping ``group_id -> allocated_amount``.
    """
    dow = doomsday(on_date.year, on_date.month, on_date.day)
    wk_vec = weekday_weight_vector(dow, groups)

    # angles are simply the integer group indices; they act as a surrogate
    # for a continuous variable in the Gaussian/Fisher/Caputo formulas.
    angles = np.arange(groups, dtype=np.float64)

    fisher_vec = np.array([fisher_score(theta, center, width) for theta in angles])
    caputo_vec = np.array([caputo_weights(theta + 1.0, alpha) for theta in angles])  # +1 avoids t=0

    # Element‑wise product forms the raw importance metric
    raw = wk_vec * fisher_vec * caputo_vec

    # Guard against pathological zero‑vector
    if raw.sum() == 0:
        normalized = np.full(groups, 1.0 / groups)
    else:
        normalized = raw / raw.sum()

    allocation = (normalized * total_units).tolist()
    return {f"group_{i}": allocation[i] for i in range(groups)}

# ----------------------------------------------------------------------
# Additional helper that demonstrates matrix‑style combination of the three
# contributions for an arbitrary number of items.
# ----------------------------------------------------------------------
def hybrid_matrix_combination(
    items: np.ndarray,
    alpha: float,
    width: float,
    center: float
) -> np.ndarray:
    """
    Given a 1‑D array ``items`` (interpreted as angles or timestamps),
    return a column vector where each row equals:

        G(item) * Φ(item;α) * I(item)

    with G = Gaussian beam, Φ = Caputo kernel, I = Fisher information.
    """
    g = np.vectorize(gaussian_beam)(items, center, width)
    phi = np.vectorize(caputo_weights)(items, alpha)
    fisher = np.vectorize(fisher_score)(items, center, width)
    return (g * phi * fisher).reshape(-1, 1)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ---- Test hybrid_chrono_caputo ----
    dummy_candidates = [
        {"id": "a", "timestamp": "2023-01-01T12:00:00"},
        {"id": "b", "timestamp": "2023-01-02T15:30:00"},
        {"id": "c", "timestamp": "2023-01-03T09:45:00"},
    ]
    scores = hybrid_chrono_caputo(dummy_candidates, center=0.0, width=1e9, alpha=0.5)
    print("Chrono‑Caputo scores:", scores)

    # ---- Test hybrid_fisher_caputo_allocation ----
    today = date.today()
    alloc = hybrid_fisher_caputo_allocation(
        total_units=1000.0,
        on_date=today,
        alpha=0.4,
        width=5.0,
        center=2.0,
        groups=5,
    )
    print(f"Allocation for {today.isoformat()}:", alloc)

    # ---- Test hybrid_matrix_combination ----
    angles = np.linspace(0.1, 5.0, 8)
    mat = hybrid_matrix_combination(angles, alpha=0.3, width=1.0, center=0.0)
    print("Hybrid matrix combination:\n", mat)