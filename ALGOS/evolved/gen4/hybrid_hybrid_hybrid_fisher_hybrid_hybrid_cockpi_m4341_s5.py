# DARWIN HAMMER — match 4341, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_worksh_m146_s2.py (gen3)
# parent_b: hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s0.py (gen2)
# born: 2026-05-29T23:55:04Z

"""Hybrid Algorithm: Fisher‑Info & Cockpit Trust Fusion

Parents:
- hybrid_hybrid_fisher_locali_hybrid_hybrid_worksh_m146_s2.py (Fisher information scoring,
  chronological date handling, weekday workshare vectors)
- hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s0.py (cockpit honesty/anti‑slop metrics,
  stylometry‑derived trust factor)

Mathematical Bridge:
Both parents manipulate *information density*.  In the Fisher branch the density is
`I(θ) = (∂ℓ/∂θ)² / ℓ` (the Fisher information of a Gaussian beam).  In the cockpit branch the
stylometric analysis yields a scalar *trust* `τ ∈ [0,1]`.  The hybrid treats the Fisher
information of each temporal candidate as a *density weight* `w_i` and modulates every
cockpit‑derived quantity by the same scalar `τ`.  The resulting hybrid weight matrix

    H_{i,k} = τ · w_i · W_k(dow_i)

where `W_k` is the weekday‑weight vector (k = 0…3) for the candidate’s day‑of‑week,
provides a unified representation that can drive both work‑share allocation and
velocity‑field scaling.

The module below implements this fusion, exposing three core functions:
`hybrid_weight_matrix`, `allocate_workshare`, and `trust_weighted_velocity`.
"""

import math
import random
import sys
from datetime import datetime, timezone, date
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – Fisher information and chronological utilities
# ---------------------------------------------------------------------------

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def parse_loose_datetime(raw: str) -> Optional[datetime]:
    """Best‑effort ISO‑8601 parser; returns UTC datetime or None."""
    text = raw.strip().strip("'\"`[]()")
    if not text:
        return None
    try:
        val = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if val.tzinfo is None:
            val = val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)
    except ValueError:
        return None


def weekday_weight_vector(dow: int) -> np.ndarray:
    """
    Produce a 4‑element cyclic weight vector based on day‑of‑week.
    The formulation mirrors the parent algorithm: a sinusoidal modulation
    with a small amplitude (0.2) around a unit baseline.
    """
    n = 4
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    return raw.astype(float)


# ---------------------------------------------------------------------------
# Parent B – Cockpit honesty / anti‑slop metrics and stylometry trust
# ---------------------------------------------------------------------------

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0, 1]."""
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known‑good, clamped to [0, 1]."""
    total = displayed_ok + unknown_displayed_as_ok
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, displayed_ok / total))


# Minimal stylometry dictionary (subset of the original FUNCTION_CATS)
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below beneath beside between beyond but by concerning despite down during except for from in into like near of off on onto out over past since through till to toward under up upon with within without".split()),
}


def _tokenize(text: str) -> List[str]:
    """Very simple whitespace / punctuation tokenizer."""
    return [t.strip(".,!?;:\"'()[]{}") for t in text.lower().split() if t]


def stylometry_features(text: str) -> Dict[str, float]:
    """
    Compute normalized frequencies for each FUNCTION_CATS category.
    Returns a dict mapping category -> frequency (0‑1).
    """
    tokens = _tokenize(text)
    if not tokens:
        return {cat: 0.0 for cat in FUNCTION_CATS}
    counts = {cat: 0 for cat in FUNCTION_CATS}
    for token in tokens:
        for cat, vocab in FUNCTION_CATS.items():
            if token in vocab:
                counts[cat] += 1
    total = len(tokens)
    return {cat: cnt / total for cat, cnt in counts.items()}


def trust_from_stylometry(text: str) -> float:
    """
    Derive a scalar trust value τ from stylometry.
    Heuristic: pronoun usage (self‑reference) reduces trust,
    while article usage slightly increases it.
    Result is clamped to [0,1].
    """
    feats = stylometry_features(text)
    pronoun = feats.get("pronoun", 0.0)
    article = feats.get("article", 0.0)
    # Base trust 0.7, penalize pronouns, reward articles
    raw = 0.7 - 0.4 * pronoun + 0.2 * article
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Hybrid Core – information density as common bridge
# ---------------------------------------------------------------------------

def hybrid_weight_matrix(
    datetimes: List[datetime],
    cockpit_metrics: Tuple[int, int, int, int],
    trust_text: str,
    beam_width: float = 86400.0,
) -> np.ndarray:
    """
    Build the hybrid weight matrix H_{i,k} = τ · w_i · W_k(dow_i).

    Parameters
    ----------
    datetimes : list of datetime
        Temporal candidates (UTC).
    cockpit_metrics : (displayed_ok, unknown_displayed_as_ok,
                       claims_with_evidence, total_claims_emitted)
        Raw integer counts for the cockpit honesty and anti‑slop ratios.
    trust_text : str
        Free‑form text whose stylometry supplies the base trust τ₀.
    beam_width : float, optional
        Width (in seconds) of the Gaussian beam used for Fisher scoring.
        Default corresponds to a one‑day spread.

    Returns
    -------
    np.ndarray
        Shape (N, 4) where N = len(datetimes).  Each row i contains the four
        weekday‑modulated weights for candidate i.
    """
    if not datetimes:
        return np.empty((0, 4))

    # 1. Convert datetimes to scalar θ (seconds since epoch)
    thetas = np.array([dt.timestamp() for dt in datetimes], dtype=float)

    # 2. Choose a centre (median) for the Gaussian beam
    center = float(np.median(thetas))

    # 3. Fisher information per candidate (density weight)
    fisher_weights = np.array(
        [fisher_score(theta, center, beam_width) for theta in thetas],
        dtype=float,
    )
    # Normalise to unit sum to avoid exploding scales
    if fisher_weights.sum() > 0:
        fisher_weights /= fisher_weights.sum()

    # 4. Compute scalar trust τ from cockpit metrics and stylometry
    displayed_ok, unknown_displayed_as_ok, claims_with_evidence, total_claims_emitted = cockpit_metrics
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    anti_slop = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    τ0 = trust_from_stylometry(trust_text)
    τ = honesty * anti_slop * τ0  # composite trust factor in [0,1]

    # 5. Assemble the matrix
    rows = []
    for dt, w_i in zip(datetimes, fisher_weights):
        dow = dt.weekday()  # Monday=0 … Sunday=6
        weekday_vec = weekday_weight_vector(dow)  # length‑4 vector
        rows.append(τ * w_i * weekday_vec)
    return np.vstack(rows)


def allocate_workshare(weight_matrix: np.ndarray) -> Dict[int, float]:
    """
    Convert the hybrid weight matrix into a normalized allocation per
    weekday‑slot (0‑3).  The allocation for slot k is the sum of all
    candidate contributions, renormalised to sum to 1.

    Returns
    -------
    dict {slot_index: proportion}
    """
    if weight_matrix.size == 0:
        return {k: 0.0 for k in range(4)}
    slot_sums = weight_matrix.sum(axis=0)
    total = slot_sums.sum()
    if total == 0:
        return {k: 0.0 for k in range(4)}
    return {int(k): float(slot_sums[k] / total) for k in range(4)}


def trust_weighted_velocity(
    base_velocity: np.ndarray,
    weight_matrix: np.ndarray,
) -> np.ndarray:
    """
    Produce a velocity field that is modulated by the hybrid information density.
    The base_velocity is expected to have shape (N, D) where D is the spatial
    dimension (e.g., 2 or 3).  The function returns a field of the same shape:

        V_{i} = base_velocity_i * (τ * w_i)

    where τ·w_i is the scalar factor obtained by collapsing the weekday vector
    (e.g., by its mean).  This keeps the operation matrix‑free while preserving
    the hybrid influence.

    Parameters
    ----------
    base_velocity : np.ndarray (N, D)
    weight_matrix : np.ndarray (N, 4)

    Returns
    -------
    np.ndarray (N, D)
    """
    if base_velocity.shape[0] != weight_matrix.shape[0]:
        raise ValueError("Mismatched candidate counts between velocity and weight matrix")
    # Collapse weekday vector to a single scalar per candidate (mean weighting)
    scalar_factors = weight_matrix.mean(axis=1, keepdims=True)  # shape (N,1)
    return base_velocity * scalar_factors


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Sample datetime candidates (ISO strings)
    iso_strings = [
        "2023-01-15T08:30:00Z",
        "2023-01-16T14:45:00Z",
        "2023-01-17T21:10:00Z",
        "2023-01-18T03:55:00Z",
    ]
    candidates = [parse_loose_datetime(s) for s in iso_strings]
    candidates = [dt for dt in candidates if dt is not None]

    # 2. Cockpit metric counts (arbitrary but plausible)
    cockpit_counts = (120, 30, 85, 100)  # displayed_ok, unknown_ok, claims_with_evidence, total_claims

    # 3. Text that will feed the stylometry trust estimator
    sample_text = "I think the system works well, but we need to verify the results."

    # 4. Build hybrid weight matrix
    H = hybrid_weight_matrix(candidates, cockpit_counts, sample_text)

    print("Hybrid weight matrix (rows = candidates, cols = weekday slots):")
    print(H)

    # 5. Derive a workshare allocation
    allocation = allocate_workshare(H)
    print("\nWorkshare allocation per weekday slot:")
    for slot, prop in allocation.items():
        print(f"  Slot {slot}: {prop:.3%}")

    # 6. Example base velocity field (2‑D) for each candidate
    base_vel = np.array([
        [1.0, 0.5],
        [0.8, 0.3],
        [1.2, 0.7],
        [0.9, 0.4],
    ])
    vel = trust_weighted_velocity(base_vel, H)
    print("\nTrust‑weighted velocity field:")
    print(vel)