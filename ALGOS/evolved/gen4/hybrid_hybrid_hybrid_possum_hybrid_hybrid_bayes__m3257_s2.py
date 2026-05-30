# DARWIN HAMMER — match 3257, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_possum_filter_hybrid_caputo_fracti_m1220_s4.py (gen3)
# parent_b: hybrid_hybrid_bayes_claim_k_hybrid_hybrid_distri_m351_s1.py (gen3)
# born: 2026-05-29T23:48:54Z

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    """Simple immutable record describing an entity."""
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


# ----------------------------------------------------------------------
# Geospatial utilities
# ----------------------------------------------------------------------
def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance between two (lat, lon) points in metres."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def _entity_signature(e: Entity) -> str:
    """Canonical signature used for similarity checks."""
    return (e.address_signature or e.category).strip().lower()


# ----------------------------------------------------------------------
# Privacy‑risk modelling
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Normalised reconstruction risk in [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def spatial_aware_privacy_risk_vector(
    entities: List[Entity], delta_m: float
) -> np.ndarray:
    """
    For each entity compute a privacy risk based on the proportion of *similar*
    entities (same signature and within ``delta_m`` metres).  The entity itself
    is included in the denominator but not in the numerator, matching the
    intuition that a lone entity carries no reconstruction risk.
    """
    n = len(entities)
    risks = np.empty(n, dtype=float)

    for i, e in enumerate(entities):
        similar = [
            other
            for j, other in enumerate(entities)
            if j != i
            and _entity_signature(e) == _entity_signature(other)
            and haversine_m((e.lat, e.lon), (other.lat, other.lon)) <= delta_m
        ]
        uniq_ids = {s.id for s in similar}
        risks[i] = reconstruction_risk_score(len(uniq_ids), n)

    return risks


# ----------------------------------------------------------------------
# Fractional‑memory (Caputo) weighting
# ----------------------------------------------------------------------
def _caputo_kernel(alpha: float, t: int, tau: int) -> float:
    """Discrete Caputo kernel (t‑tau)^{-alpha} / Γ(1‑α)."""
    if t <= tau:
        return 0.0
    return (t - tau) ** (-alpha) / math.gamma(1.0 - alpha)


def fractional_memory_cost(
    base_series: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """
    Apply a left‑fractional Caputo‑type memory operator to ``base_series``.
    The result is a smoothed series that gives more weight to recent
    high‑risk values while preserving a long‑range memory effect.
    """
    n = len(base_series)
    if n == 0:
        return np.array([], dtype=float)

    cost = np.empty(n, dtype=float)
    for t in range(n):
        acc = 0.0
        for tau in range(t + 1):
            acc += base_series[tau] * _caputo_kernel(alpha, t, tau)
        cost[t] = acc
    # Normalise to keep values comparable with the original risk range [0,1]
    max_val = cost.max() if cost.max() != 0 else 1.0
    return cost / max_val


# ----------------------------------------------------------------------
# Dynamic burst‑admission model
# ----------------------------------------------------------------------
def pulse_force(peak_force: float, steps: int) -> List[float]:
    """Triangular pulse with ``steps`` discrete points."""
    if steps <= 1:
        return [peak_force]
    mid = (steps - 1) / 2.0
    denom = max(1.0, mid)
    return [
        peak_force * max(0.0, 1.0 - abs(i - mid) / denom) for i in range(steps)
    ]


def burst_admission_score(
    work_value: float,
    cost_drag: float,
    urgency_force: float,
    steps: int = 12,
) -> float:
    """
    Simulate a velocity‑limited admission process.
    The returned value is the *integrated* displacement (area under the velocity
    curve) rather than the peak velocity, providing a richer scalar that
    captures both intensity and duration.
    """
    force_series = pulse_force(urgency_force, steps)
    v = 0.0
    displacement = 0.0
    for f in force_series:
        drag = cost_drag * v * abs(v)
        acc = f - drag
        v = max(0.0, v + acc)          # velocity never goes negative
        displacement += v              # simple Euler integration
    # Scale by the supplied work value (interpreted as a budget multiplier)
    return displacement * work_value


# ----------------------------------------------------------------------
# Bayesian fusion
# ----------------------------------------------------------------------
def hybrid_bayes_update(prior: float, likelihood: float, evidence_score: float) -> float:
    """
    Proper Bayesian update with a symmetric binary hypothesis space.
    ``evidence_score`` is interpreted as P(E|H=1).  The denominator implements
    total probability.
    """
    # Clamp inputs to avoid division by zero or invalid probabilities
    prior = max(0.0, min(1.0, prior))
    likelihood = max(0.0, min(1.0, likelihood))
    evidence_score = max(0.0, min(1.0, evidence_score))

    numerator = prior * likelihood * evidence_score
    denominator = (
        numerator
        + (1.0 - prior) * (1.0 - likelihood) * (1.0 - evidence_score)
    )
    return numerator / denominator if denominator != 0 else 0.0


# ----------------------------------------------------------------------
# Evidence selection with deeper integration
# ----------------------------------------------------------------------
def hybrid_evidence_selection(
    entities: List[Entity],
    delta_m: float,
    work_value: float,
    cost_drag: float,
    urgency_force: float,
    alpha: float = 0.5,
) -> List[Entity]:
    """
    Rank entities by a composite score that fuses:
      * spatial privacy risk,
      * fractional‑memory cost (Caputo weighting),
      * burst‑admission dynamics.
    Returns the list sorted from highest to lowest composite score.
    """
    if not entities:
        return []

    # 1️⃣ Base privacy risk per entity
    base_risks = spatial_aware_privacy_risk_vector(entities, delta_m)

    # 2️⃣ Fractional‑memory augmentation (captures temporal correlation)
    mem_cost = fractional_memory_cost(base_risks, alpha=alpha)

    # 3️⃣ Global burst admission factor (same for all entities)
    burst_factor = burst_admission_score(work_value, cost_drag, urgency_force)

    # 4️⃣ Composite per‑entity score
    composite = burst_factor * base_risks * mem_cost

    # Sort while preserving entity objects
    sorted_entities = [
        ent for _, ent in sorted(
            zip(composite, entities), key=lambda pair: pair[0], reverse=True
        )
    ]
    return sorted_entities


# ----------------------------------------------------------------------
# End‑to‑end inference
# ----------------------------------------------------------------------
def hybrid_inference(
    entities: List[Entity],
    delta_m: float,
    work_value: float,
    cost_drag: float,
    urgency_force: float,
    prior: float,
    likelihood: float,
    alpha: float = 0.5,
) -> float:
    """
    Full pipeline:
      1. Rank entities using the deep fusion model.
      2. Compute an evidence score as the weighted average of the entity
         intrinsic scores (``Entity.score``) using the composite ranking as
         weights.
      3. Apply a mathematically sound Bayesian update.
    """
    selected = hybrid_evidence_selection(
        entities,
        delta_m,
        work_value,
        cost_drag,
        urgency_force,
        alpha=alpha,
    )
    if not selected:
        return prior  # nothing to update

    # Weight entity scores by their ranking position (higher rank → higher weight)
    weights = np.arange(len(selected), 0, -1, dtype=float)
    weighted_scores = np.array([e.score for e in selected]) * weights
    evidence_score = weighted_scores.sum() / weights.sum()

    return hybrid_bayes_update(prior, likelihood, evidence_score)


# ----------------------------------------------------------------------
# Demonstration (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_entities = [
        Entity("1", 37.7749, -122.4194, "category1", 0.5),
        Entity("2", 37.7859, -122.4364, "category1", 0.7),
        Entity("3", 37.7963, -122.4575, "category2", 0.3),
    ]

    result = hybrid_inference(
        entities=demo_entities,
        delta_m=1_000.0,
        work_value=10.0,
        cost_drag=0.1,
        urgency_force=5.0,
        prior=0.5,
        likelihood=0.8,
        alpha=0.6,
    )
    print(f"Posterior probability: {result:.6f}")