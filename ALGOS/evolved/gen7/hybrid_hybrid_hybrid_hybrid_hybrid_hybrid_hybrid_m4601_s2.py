# DARWIN HAMMER — match 4601, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_cockpi_hybrid_percyphon_hyb_m2524_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_rbf_su_m726_s2.py (gen4)
# born: 2026-05-29T23:56:57Z

"""Hybrid Algorithm: Cockpit-Morphology Fusion

Parents:
- hybrid_hybrid_cockpit_metri_hybrid_hybrid_hybrid_m608_s2.py (cockpit metrics)
- hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_rbf_su_m726_s2.py (morphology & entropy)

Mathematical Bridge:
Both parents expose scalar quality indices that lie in the interval [0, 1].
The bridge is built by using the *sphericity* (S) and *flatness* (F) indices
from the morphology side as adaptive weights for the cockpit side
(`anti_slop_ratio` A and `cockpit_honesty` H).  The combined integrity
score I is defined as a weighted arithmetic mean:

    w_s = S / (S + F)          (sphericity weight)
    w_f = F / (S + F)          (flatness weight)
    I   = w_s * A + w_f * H

This creates a continuous mapping from shape characteristics to trust
metrics, yielding a single unified scalar that can be further processed
(e.g., by right‑handing‑time recovery priority or Euclidean distance
between feature vectors)."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter
from typing import List, Sequence, Any

import numpy as np

# ----------------------------------------------------------------------
# Morphology (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Normalized sphericity ∈ (0,1]. Higher when dimensions are similar."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness measure; larger when height is small compared to length/width."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Physical right‑ing time proxy used for recovery priority."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Scaled priority ∈ [0,1] derived from righting_time_index."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Cockpit Metrics (Parent A)
# ----------------------------------------------------------------------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Proportion of claims that are backed by evidence."""
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are truly OK."""
    total = displayed_ok + unknown_displayed_as_ok
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, displayed_ok / total))


# ----------------------------------------------------------------------
# Entropy & Distance utilities (Parent B)
# ----------------------------------------------------------------------
def shannon_entropy(observations: List[Any], is_distribution: bool = False) -> float:
    """Shannon entropy in bits."""
    xs = list(observations)
    if not xs:
        return 0.0
    if is_distribution:
        probs = [float(x) for x in xs]
        if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError("distribution must sum to 1")
    else:
        c = Counter(xs)
        total = sum(c.values())
        probs = [v / total for v in c.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must be equal length")
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def hybrid_integrity_score(m: Morphology,
                          claims_with_evidence: int,
                          total_claims_emitted: int,
                          displayed_ok: int,
                          unknown_displayed_as_ok: int) -> float:
    """
    Combine morphology (sphericity, flatness) with cockpit metrics
    (anti‑slop, honesty) into a single integrity score ∈ [0,1].

    Steps:
    1. Compute S = sphericity_index, F = flatness_index.
    2. Derive normalized weights w_s = S/(S+F), w_f = F/(S+F).
    3. Compute A = anti_slop_ratio, H = cockpit_honesty.
    4. Return I = w_s * A + w_f * H.
    """
    S = sphericity_index(m.length, m.width, m.height)
    F = flatness_index(m.length, m.width, m.height)
    if S + F == 0:
        raise ValueError("Both sphericity and flatness are zero")
    w_s = S / (S + F)
    w_f = F / (S + F)

    A = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    H = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)

    return max(0.0, min(1.0, w_s * A + w_f * H))


def hybrid_recovery_assessment(m: Morphology,
                               claims_with_evidence: int,
                               total_claims_emitted: int,
                               displayed_ok: int,
                               unknown_displayed_as_ok: int,
                               observations: List[Any]) -> dict:
    """
    Produce a richer assessment dict that includes:
    - integrity_score   (from hybrid_integrity_score)
    - recovery_priority (from morphology)
    - entropy           (Shannon entropy of observations)
    - weighted_priority (entropy‑modulated recovery priority)
    """
    integrity = hybrid_integrity_score(m,
                                       claims_with_evidence,
                                       total_claims_emitted,
                                       displayed_ok,
                                       unknown_displayed_as_ok)

    priority = recovery_priority(m)

    entropy = shannon_entropy(observations)

    # Modulate priority: higher entropy (more uncertainty) reduces priority
    weighted_priority = priority * math.exp(-entropy)

    return {
        "integrity_score": integrity,
        "recovery_priority": priority,
        "entropy_bits": entropy,
        "weighted_priority": max(0.0, min(1.0, weighted_priority)),
    }


def hybrid_distance_between_entities(m1: Morphology,
                                     m2: Morphology,
                                     metrics1: dict,
                                     metrics2: dict) -> float:
    """
    Compute a Euclidean distance between two entities using a feature vector
    that concatenates normalized morphology indices and cockpit metrics.

    Feature vector for an entity:
    [sphericity, flatness, anti_slop, honesty]

    All components are already in [0,1], so direct Euclidean distance is meaningful.
    """
    vec1 = [
        sphericity_index(m1.length, m1.width, m1.height),
        flatness_index(m1.length, m1.width, m1.height),
        metrics1.get("anti_slop", 0.0),
        metrics1.get("honesty", 0.0),
    ]
    vec2 = [
        sphericity_index(m2.length, m2.width, m2.height),
        flatness_index(m2.length, m2.width, m2.height),
        metrics2.get("anti_slop", 0.0),
        metrics2.get("honesty", 0.0),
    ]
    return euclidean(vec1, vec2)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample morphologies
    m_a = Morphology(length=2.0, width=1.8, height=1.5, mass=3.2)
    m_b = Morphology(length=1.0, width=1.0, height=2.5, mass=2.0)

    # Cockpit data for two hypothetical agents
    data_a = {
        "claims_with_evidence": 45,
        "total_claims_emitted": 60,
        "displayed_ok": 30,
        "unknown_displayed_as_ok": 5,
    }
    data_b = {
        "claims_with_evidence": 20,
        "total_claims_emitted": 40,
        "displayed_ok": 15,
        "unknown_displayed_as_ok": 10,
    }

    # Observations (e.g., categorical outcomes)
    obs_a = ["good", "good", "bad", "good", "neutral"]
    obs_b = ["bad", "bad", "bad", "neutral"]

    # Compute hybrid scores
    score_a = hybrid_integrity_score(m_a, **data_a)
    score_b = hybrid_integrity_score(m_b, **data_b)

    assessment_a = hybrid_recovery_assessment(m_a, **data_a, observations=obs_a)
    assessment_b = hybrid_recovery_assessment(m_b, **data_b, observations=obs_b)

    # Distance between entities
    metrics_a = {
        "anti_slop": anti_slop_ratio(data_a["claims_with_evidence"], data_a["total_claims_emitted"]),
        "honesty": cockpit_honesty(data_a["displayed_ok"], data_a["unknown_displayed_as_ok"]),
    }
    metrics_b = {
        "anti_slop": anti_slop_ratio(data_b["claims_with_evidence"], data_b["total_claims_emitted"]),
        "honesty": cockpit_honesty(data_b["displayed_ok"], data_b["unknown_displayed_as_ok"]),
    }
    dist = hybrid_distance_between_entities(m_a, m_b, metrics_a, metrics_b)

    # Simple sanity prints (no external dependencies)
    print(f"Integrity A: {score_a:.4f}")
    print(f"Integrity B: {score_b:.4f}")
    print("Assessment A:", assessment_a)
    print("Assessment B:", assessment_b)
    print(f"Euclidean distance between entities: {dist:.4f}")

    # Ensure the script exits cleanly
    sys.exit(0)