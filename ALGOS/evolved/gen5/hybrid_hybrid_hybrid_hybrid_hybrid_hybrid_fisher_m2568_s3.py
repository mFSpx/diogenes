# DARWIN HAMMER — match 2568, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_caputo_hybrid_fisher_locali_m637_s1.py (gen4)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_worksh_m146_s1.py (gen3)
# born: 2026-05-29T23:42:56Z

"""
Hybrid Chrono‑Caputo‑Fisher Allocation
====================================

This module fuses the two parent algorithms:

* **Hybrid Caputo‑Geometric Minimum‑Cost Tree (HCG‑MCT)** – provides the
  Caputo fractional‑derivative kernel via a Lanczos‑approximated Γ function.
* **Hybrid Chrono‑Fisher / Weekday Allocation** – supplies a Gaussian‑beam
  model, its Fisher‑information score (derivative of the Gaussian) and a
  sinusoidal weekday weight vector.

**Mathematical bridge**

Both parents rely on the same Gaussian beam `G(θ) = exp(-½((θ−c)/w)²)`.  
The HCG‑MCT multiplies this beam by a Caputo kernel `C(t;α)=t^{‑α}/Γ(1‑α)`.  
The Fisher score in the second parent is `I(θ)= (∂_θ G)² / G`.  

By defining a *combined* weight


W(θ;α) = G(θ)·C(θ;α)


its derivative is


∂_θ W = (∂_θ G)·C + G·(∂_θ C)


and the associated Fisher‑information becomes


I_W(θ;α) = (∂_θ W)² / W .


The hybrid algorithm therefore:
1. Computes `W` for each timestamp (Caputo‑Gaussian weighting).
2. Derives `I_W` to obtain a Fisher‑like importance per angle/group.
3. Mixes `I_W` with the weekday sinusoidal weight vector to allocate a
   total resource across groups.

The implementation below provides three public functions that demonstrate
this fused behaviour and a smoke‑test in the `__main__` block.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, date
import numpy as np

# ----------------------------------------------------------------------
# Lanczos approximation of the Gamma function (used by the Caputo kernel)
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
    """Lanczos approximation of Γ(z) for z > 0."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    z -= 1
    x = _LANCZOS_C / (z + np.arange(_LANCZOS_G + 1))
    # The first coefficient is 1.0, the rest are stored in _LANCZOS_C[1:].
    # The series sum can be written as a dot product.
    series = 1.0 + np.dot(x[1:], np.ones(_LANCZOS_G))
    t = z + _LANCZOS_G - 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * series

# ----------------------------------------------------------------------
# Core mathematical building blocks
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Standard Gaussian beam."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def caputo_weights(t: float, alpha: float) -> float:
    """Caputo fractional‑derivative kernel φ(t;α) ≈ t^{‑α} / Γ(1‑α)."""
    if t <= 0:
        raise ValueError("time argument must be positive for Caputo kernel")
    return t ** (-alpha) / gamma_lanczos(1 - alpha)

def combined_weight(theta: float, center: float, width: float, alpha: float) -> float:
    """Gaussian beam multiplied by the Caputo kernel."""
    return gaussian_beam(theta, center, width) * caputo_weights(theta, alpha)

def derivative_caputo(t: float, alpha: float) -> float:
    """Analytical derivative of the Caputo kernel with respect to t."""
    # d/dt [ t^{‑α} ] = -α * t^{‑α‑1}
    return -alpha * t ** (-alpha - 1) / gamma_lanczos(1 - alpha)

def fisher_information_combined(theta: float, center: float, width: float, alpha: float) -> float:
    """
    Fisher information derived from the combined weight W(θ;α).

    I_W(θ;α) = (∂_θ W)² / W .
    """
    G = gaussian_beam(theta, center, width)
    C = caputo_weights(theta, alpha)

    # ∂_θ G = G * (-(θ‑c)/w²)
    dG = G * (-(theta - center) / (width * width))

    # ∂_θ C = dC/dt where we treat θ as the time argument
    dC = derivative_caputo(theta, alpha)

    dW = dG * C + G * dC
    W = G * C
    if W == 0:
        return 0.0
    return (dW * dW) / W

# ----------------------------------------------------------------------
# Weekday sinusoidal weight vector (parent B)
# ----------------------------------------------------------------------
def weekday_weight_vector(dow: int, groups: int) -> np.ndarray:
    """
    Produce a row‑stochastic vector based on the day‑of‑week.
    `dow` follows Python's Monday=0 … Sunday=6 convention.
    """
    base_angles = np.arange(groups) * (2.0 * math.pi) / groups
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    return (raw / raw.sum()).astype(np.float64)

def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday as 0=Monday … 6=Sunday."""
    return date(year, month, day).weekday()

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def combined_candidate_scores(
    candidates: list[dict[str, str]],
    center: float,
    width: float,
    alpha: float,
) -> np.ndarray:
    """
    For each candidate dictionary containing an ISO‑8601 ``timestamp`` field,
    compute the combined weight W(timestamp;α).
    Returns a NumPy array of scores.
    """
    scores = []
    for cand in candidates:
        ts = datetime.fromisoformat(cand["timestamp"]).timestamp()
        scores.append(combined_weight(ts, center, width, alpha))
    return np.array(scores, dtype=np.float64)

def fisher_vector_from_angles(
    groups: int,
    center: float,
    width: float,
    alpha: float,
) -> np.ndarray:
    """
    Compute the Fisher information I_W for a set of discrete angles
    (0, 1, …, groups‑1) which act as surrogate “directions” for allocation.
    """
    angles = np.arange(groups, dtype=np.float64)
    vec = np.array(
        [fisher_information_combined(theta, center, width, alpha) for theta in angles],
        dtype=np.float64,
    )
    # Normalise to avoid extreme scaling before mixing with weekday weights
    if vec.sum() == 0:
        return vec
    return vec / vec.sum()

def hybrid_time_weekday_allocation(
    total_units: float,
    allocation_date: date,
    candidates: list[dict[str, str]],
    groups: int = 4,
    center: float = 0.0,
    width: float = 1.0,
    alpha: float = 0.5,
    deterministic_target_pct: float = 90.0,
) -> dict:
    """
    Core hybrid routine.

    1. Compute weekday sinusoidal weights.
    2. Compute a Fisher‑information vector from the combined Gaussian‑Caputo model.
    3. Multiply (1) and (2) to obtain a mixed importance vector.
    4. Normalise and allocate ``total_units`` proportionally.
    5. Additionally, distribute candidate scores across groups (by index modulo)
       and report the summed candidate contribution per group.

    Returns a dictionary with:
        - "allocation": per‑group unit allocation (float)
        - "candidate_scores": per‑group summed combined weights (float)
        - "mixed_importance": the normalised product of weekday & Fisher vectors
    """
    # 1. Weekday weight vector
    dow = doomsday(allocation_date.year, allocation_date.month, allocation_date.day)
    weekday_vec = weekday_weight_vector(dow, groups)

    # 2. Fisher information vector from the combined model
    fisher_vec = fisher_vector_from_angles(groups, center, width, alpha)

    # 3. Mixed importance (element‑wise product) and normalisation
    mixed_raw = weekday_vec * fisher_vec
    if mixed_raw.sum() == 0:
        mixed_vec = np.full(groups, 1.0 / groups)
    else:
        mixed_vec = mixed_raw / mixed_raw.sum()

    # 4. Allocation of total units
    allocation = total_units * mixed_vec

    # 5. Candidate scores per group
    cand_scores = combined_candidate_scores(candidates, center, width, alpha)
    group_indices = np.arange(len(cand_scores)) % groups
    candidate_sums = np.zeros(groups, dtype=np.float64)
    for idx, score in zip(group_indices, cand_scores):
        candidate_sums[idx] += score

    # Optional deterministic target (not used for the mathematics, kept for API compatibility)
    deterministic_target = (deterministic_target_pct / 100.0) * total_units

    return {
        "allocation": dict(enumerate(allocation.tolist())),
        "candidate_scores": dict(enumerate(candidate_sums.tolist())),
        "mixed_importance": dict(enumerate(mixed_vec.tolist())),
        "deterministic_target": deterministic_target,
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a handful of dummy candidates with timestamps spaced by 10 seconds
    now = datetime.utcnow()
    candidates = []
    for i in range(12):
        ts = (now.replace(microsecond=0) + i * np.timedelta64(10, "s")).isoformat()
        candidates.append({"id": str(i), "timestamp": ts})

    result = hybrid_time_weekday_allocation(
        total_units=1000.0,
        allocation_date=date.today(),
        candidates=candidates,
        groups=4,
        center=now.timestamp(),
        width=30.0,
        alpha=0.4,
    )

    print("Hybrid allocation result:")
    for key, value in result.items():
        print(f"{key}: {value}")