# DARWIN HAMMER — match 3042, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_indy_l_hybrid_infotaxis_hyb_m1786_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1587_s0.py (gen6)
# born: 2026-05-29T23:47:25Z

"""Hybrid Infotaxis‑Entropy Pheromone & Regret‑Weekday Engine (HIEPRWE)

Parents:
- hybrid_hybrid_hybrid_indy_l_hybrid_infotaxis_hyb_m1786_s2.py (Infotaxis‑Entropy Pheromone System)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1587_s0.py (Regret Engine + Weekday Mapping)

Mathematical Bridge:
The Shannon entropy **H** of a pheromone probability vector **p** is used as a scalar
modifier **γ = 1 + α·H** (α > 0) that scales the recovery‑priority vector **r** of the
infotaxis subsystem:

  r̃ = γ·r                    (1)

The scaled priority **r̃** is then multiplied element‑wise with the pheromone
vector to obtain a hybrid affinity **a**:

  a = p ⊙ r̃                    (2)

The second parent introduces a stylometry weighting **w** derived from feature
vector **f** and a normalized pheromone vector, and a regret‑adjusted expected
value **Ē** for each candidate action **c**:

  w = f ⊙ p̂                    (3)  
  R(c) = max_k E_k − E_c              (4)  
  Ē_c = E_c − β·R(c)                 (5)

Finally, a weekday mapping distributes the selected actions across the
seven days of the week, ensuring temporal balance.

The fused algorithm therefore proceeds as:

1. Compute **H** from the pheromone trail.
2. Modulate recovery priorities via (1) and obtain affinity (2).
3. Weight the affinity by stylometry features (3).
4. Apply the regret engine (4‑5) to adjust expected values.
5. Filter candidates by haversine distance to endpoints.
6. Assign each surviving action to a weekday.

The implementation below follows this pipeline, exposing three core
functions that illustrate the hybrid operation.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass
class Endpoint:
    """Geospatial endpoint with health‑related metrics."""
    lat: float
    lon: float
    failures: int
    failure_threshold: int
    righting_time_index: float  # morphology‑derived scalar (higher ⇒ healthier)

    @property
    def failure_rate(self) -> float:
        """Normalized failure rate in [0, 1]."""
        if self.failure_threshold == 0:
            return 0.0
        return self.failures / self.failure_threshold


@dataclass(frozen=True, slots=True)
class MathAction:
    """Elementary decision element."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0
    # mutable fields that will be filled during processing
    affinity: float = field(default=0.0, compare=False)
    adjusted_value: float = field(default=0.0, compare=False)
    weekday: str = field(default="", compare=False)


# ----------------------------------------------------------------------
# Core mathematical building blocks
# ----------------------------------------------------------------------


def shannon_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy H = -∑ p·log₂(p) for a probability vector."""
    probs = probs.astype(float)
    probs = probs / probs.sum() if probs.sum() != 0 else probs
    nonzero = probs > 0
    return -float(np.sum(probs[nonzero] * np.log2(probs[nonzero])))


def modulate_priorities(priorities: np.ndarray, entropy: float, alpha: float = 0.5) -> np.ndarray:
    """
    Scale priorities by γ = 1 + α·H (Eq. 1).

    Args:
        priorities: raw recovery‑priority vector.
        entropy: Shannon entropy of the pheromone distribution.
        alpha: tuning coefficient (>0).

    Returns:
        Scaled priority vector r̃.
    """
    gamma = 1.0 + alpha * entropy
    return gamma * priorities


def hybrid_affinity(pheromone: np.ndarray, scaled_priorities: np.ndarray) -> np.ndarray:
    """
    Element‑wise product of pheromone and scaled priorities (Eq. 2).

    Both inputs must be 1‑D arrays of equal length.
    """
    if pheromone.shape != scaled_priorities.shape:
        raise ValueError("Shape mismatch in hybrid_affinity")
    return pheromone * scaled_priorities


def stylometry_weight(features: np.ndarray, pheromone: np.ndarray) -> np.ndarray:
    """
    Weight pheromone signals by stylometry features (Eq. 3).

    Both vectors are assumed to be already normalized.
    """
    if features.shape != pheromone.shape:
        raise ValueError("Shape mismatch in stylometry_weight")
    return features * pheromone


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine distance in kilometres between two geographic points.
    """
    R = 6371.0  # Earth radius in km
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)

    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def regret_engine(actions: List[MathAction], beta: float = 0.3) -> List[MathAction]:
    """
    Adjust expected values using a simple regret formulation (Eqs. 4‑5).

    Args:
        actions: list of MathAction objects.
        beta: scaling factor for regret penalty.

    Returns:
        New list of MathAction with updated `adjusted_value`.
    """
    if not actions:
        return []

    max_ev = max(a.expected_value for a in actions)
    adjusted = []
    for a in actions:
        regret = max_ev - a.expected_value
        adjusted_ev = a.expected_value - beta * regret
        adjusted.append(
            MathAction(
                id=a.id,
                expected_value=a.expected_value,
                cost=a.cost,
                risk=a.risk,
                affinity=a.affinity,
                adjusted_value=adjusted_ev,
                weekday=a.weekday,
            )
        )
    return adjusted


def assign_weekday(actions: List[MathAction]) -> List[MathAction]:
    """
    Distribute actions across weekdays (Mon‑Sun) in a round‑robin fashion.
    """
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assigned = []
    for idx, a in enumerate(actions):
        day = weekdays[idx % len(weekdays)]
        assigned.append(
            MathAction(
                id=a.id,
                expected_value=a.expected_value,
                cost=a.cost,
                risk=a.risk,
                affinity=a.affinity,
                adjusted_value=a.adjusted_value,
                weekday=day,
            )
        )
    return assigned


# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------


def compute_health_scores(endpoints: List[Endpoint], request_sequence: List[float]) -> np.ndarray:
    """
    Simple health score: dot product of righting_time_index and request load,
    penalized by failure rate.

    Returns a vector of scores aligned with the endpoint list.
    """
    req_arr = np.array(request_sequence, dtype=float)
    scores = []
    for ep in endpoints:
        base = ep.righting_time_index * req_arr.mean()
        penalty = ep.failure_rate * base
        scores.append(base - penalty)
    return np.array(scores)


def hybrid_decision_pipeline(
    pheromone: np.ndarray,
    raw_priorities: np.ndarray,
    stylometry_features: np.ndarray,
    endpoints: List[Endpoint],
    request_sequence: List[float],
    actions: List[MathAction],
) -> List[MathAction]:
    """
    Execute the full hybrid algorithm:
    1. Entropy‑modulated affinity (parents A).
    2. Stylometry weighting (parent B).
    3. Health‑based distance filtering.
    4. Regret adjustment.
    5. Weekday assignment.
    """
    # 1️⃣ Entropy & priority modulation
    entropy = shannon_entropy(pheromone)
    scaled_priorities = modulate_priorities(raw_priorities, entropy)
    affinity = hybrid_affinity(pheromone, scaled_priorities)

    # 2️⃣ Stylometry weighting
    weighted_affinity = stylometry_weight(stylometry_features, affinity)

    # Attach affinity to actions (assume same ordering)
    if len(actions) != len(weighted_affinity):
        raise ValueError("Action count must match affinity vector length")
    actions_with_affinity = [
        MathAction(
            id=a.id,
            expected_value=a.expected_value,
            cost=a.cost,
            risk=a.risk,
            affinity=aff,
            adjusted_value=0.0,
            weekday="",
        )
        for a, aff in zip(actions, weighted_affinity)
    ]

    # 3️⃣ Health scores & distance filtering
    health_scores = compute_health_scores(endpoints, request_sequence)
    # For demonstration, keep actions whose corresponding endpoint health > median
    median_health = np.median(health_scores)
    filtered_actions = [
        a for a, h in zip(actions_with_affinity, health_scores) if h >= median_health
    ]

    # 4️⃣ Regret engine
    regret_adjusted = regret_engine(filtered_actions)

    # 5️⃣ Weekday assignment
    final_actions = assign_weekday(regret_adjusted)

    return final_actions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Mock data for a minimal run
    np.random.seed(42)
    size = 5

    # Pheromone distribution (must sum to 1 for entropy)
    pheromone = np.random.dirichlet(alpha=np.ones(size))

    # Raw recovery priorities (positive numbers)
    raw_priorities = np.random.rand(size) * 10

    # Stylometry feature vector (normalized)
    stylometry_features = np.random.rand(size)
    stylometry_features /= stylometry_features.sum()

    # Endpoints (geospatial + health metrics)
    endpoints = [
        Endpoint(
            lat=40.0 + i * 0.1,
            lon=-74.0 + i * 0.1,
            failures=random.randint(0, 5),
            failure_threshold=10,
            righting_time_index=random.uniform(0.5, 1.5),
        )
        for i in range(size)
    ]

    request_sequence = [random.uniform(0.0, 1.0) for _ in range(10)]

    # Candidate actions
    actions = [
        MathAction(
            id=f"act_{i}",
            expected_value=random.uniform(0.0, 100.0),
            cost=random.uniform(0.0, 10.0),
            risk=random.uniform(0.0, 5.0),
        )
        for i in range(size)
    ]

    result = hybrid_decision_pipeline(
        pheromone=pheromone,
        raw_priorities=raw_priorities,
        stylometry_features=stylometry_features,
        endpoints=endpoints,
        request_sequence=request_sequence,
        actions=actions,
    )

    print("Hybrid decision outcomes:")
    for a in result:
        print(
            f"ID:{a.id} | Affinity:{a.affinity:.4f} | AdjEV:{a.adjusted_value:.2f} | Day:{a.weekday}"
        )