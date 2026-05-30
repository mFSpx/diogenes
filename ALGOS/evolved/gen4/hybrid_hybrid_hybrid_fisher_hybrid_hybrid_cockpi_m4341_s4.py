# DARWIN HAMMER — match 4341, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_worksh_m146_s2.py (gen3)
# parent_b: hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s0.py (gen2)
# born: 2026-05-29T23:55:04Z

"""
Hybrid Algorithm: Fisher-Trust Workshare & Velocity Modulation

Parents:
- hybrid_hybrid_fisher_locali_hybrid_hybrid_worksh_m146_s2.py (Algorithm A)
- hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s0.py (Algorithm B)

Mathematical Bridge:
Algorithm A produces a *information density* vector for date candidates using
Fisher information scores and a weekday‑derived weight vector.  
Algorithm B produces a scalar *trust* value from cockpit honesty metrics and
stylometry evidence.  

The fusion treats the trust value as a multiplicative modulation of the
information density, yielding a *trust‑weighted density* that simultaneously
captures statistical certainty (Fisher) and evidential credibility (trust).
This density drives two downstream operations:

1. **Workshare Allocation** – the weekday weight vector is scaled by the
   trust‑weighted density, distributing workload across groups in a way that
   respects both temporal information content and source reliability.

2. **Velocity Field Modulation** – an ideal velocity field (from the rectified
   flow model) is multiplied by the same trust factor, producing a hybrid
   velocity that respects both physical flow constraints and the confidence
   derived from cockpit metrics.

The three core functions below illustrate the hybrid pipeline:
`trust_weighted_fisher`, `allocate_trust_weighted_workshare`,
and `trust_modulated_velocity`.
"""

import math
import random
import sys
from datetime import datetime, timezone, date
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Shared low‑level utilities (from Algorithm A)
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

def parse_loose_datetime(raw: str) -> datetime | None:
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

def doomsday(year: int, month: int, day: int) -> int:
    """Weekday number where Monday=0, Sunday=6."""
    return (date(year, month, day).weekday() + 1) % 7

def weekday_weight_vector(dow: int) -> np.ndarray:
    """
    Generate a 4‑component cyclic weight vector whose phase depends on the
    day‑of‑week.  The formulation mirrors Algorithm A's workshare component.
    """
    n = 4
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    # Normalize to sum‑to‑one for proper allocation
    raw_sum = raw.sum()
    return raw / raw_sum if raw_sum != 0 else raw

# ---------------------------------------------------------------------------
# Cockpit honesty / trust metrics (from Algorithm B)
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

def stylometry_features(text: str) -> np.ndarray:
    """
    Very lightweight stylometry: counts of pronouns, articles, and prepositions.
    Returns a normalized 3‑vector.
    """
    pronouns = {"i","me","my","mine","myself","you","your","yours","yourself",
                "he","him","his","she","her","hers","they","them","their","theirs",
                "we","us","our","ours","myself","yourself","himself","herself",
                "themselves","ourselves"}
    articles = {"a","an","the"}
    prepositions = {"about","above","after","against","around","as","at","before",
                    "behind","below","beneath","beside","between","beyond","but",
                    "by","concerning","despite","down","during","except","for",
                    "from","in","inside","into","like","near","of","off","on",
                    "onto","out","outside","over","past","since","through","to",
                    "toward","under","until","up","upon","with","within","without"}

    tokens = [t.strip('.,!?:;"\'').lower() for t in text.split()]
    cnt = np.array([
        sum(1 for t in tokens if t in pronouns),
        sum(1 for t in tokens if t in articles),
        sum(1 for t in tokens if t in prepositions)
    ], dtype=float)
    total = cnt.sum()
    return cnt / total if total > 0 else cnt

def compute_trust_value(displayed_ok: int, unknown_displayed_as_ok: int,
                        claims_with_evidence: int, total_claims_emitted: int,
                        text: str) -> float:
    """
    Fuse cockpit honesty, anti‑slop ratio and stylometry into a single
    trust scalar in [0, 1].  The stylometry vector is projected onto a
    learned (here hard‑coded) direction that rewards balanced language use.
    """
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    slop = anti_slop_ratio(claims_with_evidence, total_claims_emitted)

    # Simple projection: weight pronouns 0.4, articles 0.3, prepositions 0.3
    style_vec = stylometry_features(text)
    style_score = np.dot(style_vec, np.array([0.4, 0.3, 0.3]))
    # Combine multiplicatively then clip
    trust = honesty * slop * style_score
    return max(0.0, min(1.0, trust))

# ---------------------------------------------------------------------------
# Hybrid Core Functions
# ---------------------------------------------------------------------------

def trust_weighted_fisher(theta_vals: np.ndarray,
                          center: float,
                          width: float,
                          trust: float) -> np.ndarray:
    """
    Compute Fisher scores for a collection of theta values and scale them by
    the trust factor from the cockpit/telemetry side.  The result is a
    normalized density vector that sums to one.
    """
    raw = np.array([fisher_score(theta, center, width) for theta in theta_vals])
    weighted = raw * trust
    total = weighted.sum()
    return weighted / total if total != 0 else weighted

def allocate_trust_weighted_workshare(date_str: str,
                                      center: float,
                                      width: float,
                                      displayed_ok: int,
                                      unknown_displayed_as_ok: int,
                                      claims_with_evidence: int,
                                      total_claims_emitted: int,
                                      text: str) -> dict:
    """
    End‑to‑end allocation:
    1. Parse the date and obtain its weekday.
    2. Compute trust from cockpit metrics and stylometry.
    3. Generate a Fisher‑based density over four canonical theta positions.
    4. Scale the weekday weight vector by that density, yielding a final
       allocation across four abstract work groups.
    Returns a mapping group_index → allocated proportion.
    """
    dt = parse_loose_datetime(date_str)
    if dt is None:
        raise ValueError("Unable to parse date string")
    dow = dt.weekday()  # Monday=0 ... Sunday=6

    trust = compute_trust_value(displayed_ok, unknown_displayed_as_ok,
                                claims_with_evidence, total_claims_emitted,
                                text)

    # Four canonical theta positions evenly spaced on [0, 2π)
    thetas = np.linspace(0, 2 * math.pi, 4, endpoint=False)
    fisher_density = trust_weighted_fisher(thetas, center, width, trust)

    weekday_weights = weekday_weight_vector(dow)

    # Hybrid allocation = element‑wise product, then renormalize
    allocation_raw = fisher_density * weekday_weights
    allocation_sum = allocation_raw.sum()
    allocation = allocation_raw / allocation_sum if allocation_sum != 0 else allocation_raw

    return {i: float(allocation[i]) for i in range(4)}

def ideal_velocity(position: float) -> float:
    """A placeholder ideal velocity field (e.g., from rectified flow)."""
    return math.sin(position) + 0.5  # always positive for demonstration

def trust_modulated_velocity(position: float,
                             displayed_ok: int,
                             unknown_displayed_as_ok: int,
                             claims_with_evidence: int,
                             total_claims_emitted: int,
                             text: str) -> float:
    """
    Modulate the ideal velocity by the trust factor derived from cockpit
    evidence and stylometry.  The trust acts as a multiplicative gain.
    """
    trust = compute_trust_value(displayed_ok, unknown_displayed_as_ok,
                                claims_with_evidence, total_claims_emitted,
                                text)
    return ideal_velocity(position) * trust

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample inputs
    sample_date = "2023-11-15T13:45:00Z"
    center = math.pi
    width = 0.8

    displayed_ok = 42
    unknown_displayed_as_ok = 8
    claims_with_evidence = 35
    total_claims_emitted = 50
    text = ("I think that the system, although complex, works well. "
            "The data from the cockpit indicates consistency, and the "
            "operators have reported no major issues.")

    # 1. Workshare allocation
    allocation = allocate_trust_weighted_workshare(
        sample_date, center, width,
        displayed_ok, unknown_displayed_as_ok,
        claims_with_evidence, total_claims_emitted,
        text
    )
    print("Trust‑weighted workshare allocation per group:")
    for grp, prop in allocation.items():
        print(f"  Group {grp}: {prop:.4f}")

    # 2. Velocity modulation
    pos = 1.2
    vel = trust_modulated_velocity(pos,
                                   displayed_ok, unknown_displayed_as_ok,
                                   claims_with_evidence, total_claims_emitted,
                                   text)
    print(f"\nTrust‑modulated velocity at position {pos:.2f}: {vel:.4f}")

    # 3. Direct trust value sanity check
    trust_val = compute_trust_value(displayed_ok, unknown_displayed_as_ok,
                                    claims_with_evidence, total_claims_emitted,
                                    text)
    print(f"\nComputed trust scalar: {trust_val:.4f}")

    sys.exit(0)