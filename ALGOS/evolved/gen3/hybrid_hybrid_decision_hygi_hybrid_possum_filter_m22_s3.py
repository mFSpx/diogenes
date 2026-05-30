# DARWIN HAMMER — match 22, survivor 3
# gen: 3
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s5.py (gen1)
# parent_b: hybrid_possum_filter_hybrid_privacy_model_m53_s2.py (gen2)
# born: 2026-05-29T23:25:23Z

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, List, Tuple

import numpy as np

"""
Darwin Hammer - match 68, survivor 3
gen: 3
parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s5.py (gen2)
parent_b: hybrid_possum_filter_hybrid_privacy_model_m53_s2.py (gen2)
born: 2026-05-30T00:00:00Z

This module combines the core topologies of the decision hygiene algorithm
(hybrid_decision_hygiene_shannon_entropy_m12_s5.py) and the hybrid Possum filter
algorithm (hybrid_possum_filter_hybrid_privacy_model_m53_s2.py) into a single
unified system. The mathematical bridge is formed by merging the spatial
signature filtering concept from Possum with the resource vector formulation
from the decision hygiene algorithm.

The new system defines a 3-dimensional resource vector eᵢ = [ dᵢ, pᵢ, sᵢ ] for each
entity, where:

- dᵢ = haversine distance (in metres) from a reference location (mirroring
  the distance-threshold logic of keep_candidate in Possum),
- pᵢ = β·σᵢ, where σᵢ = 1 if the entity's signature collides with any other
  entity, otherwise 0 (treating signature duplication as a privacy-load
  analogue to the privacy-load term of the decision hygiene algorithm),
- sᵢ = score from the decision hygiene algorithm.

For each model tier, we reuse the resource vector defined in the decision
hygiene algorithm: mⱼ = [ RAMⱼ, α·τⱼ·μ ], where

- RAMⱼ is the model's RAM consumption,
- τⱼ is the tier factor (T1→1, T2→2, T3→3),
- μ is the mean(privacy_risk) over the provided records,
- α is a scaling constant.

Stacking all vectors yields a combined resource matrix A (rows = entities∪models,
columns = [spatial/RAM-load, privacy-load, score]). Selecting a subset
corresponds to a binary indicator x and must satisfy the linear constraints

Aᵀ·x ≤ [ spatial_budget, privacy_budget, decision_budget ]

where spatial_budget is the total allowed distance (or 0 for pure model
selection), privacy_budget is the privacy-budget from the decision hygiene
algorithm, and decision_budget is the maximum allowed score (or 0 for pure
spatial/mode selection). The greedy algorithm respects both topologies in a
single unified decision process.
"""

# RegExes and raw count extraction from the decision hygiene algorithm
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)
RISK_RE = re.compile(
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b",
    re.I,
)

_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

# Positive contributions (desired cues) from the decision hygiene algorithm
_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)

# Entity class combining the spatial signature filtering concept from Possum
# with the resource vector formulation from the decision hygiene algorithm
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    evidence_score: float = 0.0
    planning_score: float = 0.0
    delay_score: float = 0.0
    support_score: float = 0.0
    boundary_score: float = 0.0
    outcome_score: float = 0.0
    impulsive_score: float = 0.0
    scarcity_score: float = 0.0
    risk_score: float = 0.0
    score: float = 0.0
    address_signature: str = ""

    def calculate_resource_vector(self, beta, decision_budget):
        d = haversine_m((self.lat, self.lon), (0, 0))
        p = beta * (1 if self.address_signature else 0)
        s = (
            _POSITIVE_WEIGHTS[0] * self.evidence_score
            + _POSITIVE_WEIGHTS[1] * self.planning_score
            + _POSITIVE_WEIGHTS[2] * self.delay_score
            + _POSITIVE_WEIGHTS[3] * self.support_score
            + _POSITIVE_WEIGHTS[4] * self.boundary_score
            + _POSITIVE_WEIGHTS[5] * self.outcome_score
            + _POSITIVE_WEIGHTS[6] * self.impulsive_score
            + _POSITIVE_WEIGHTS[7] * self.scarcity_score
            + _POSITIVE_WEIGHTS[8] * self.risk_score
        )
        return np.array([d, p, s])

# Model tier resource vector from the decision hygiene algorithm
@dataclass(frozen=True)
class ModelTier:
    id: str
    ram: float
    tier_factor: float
    privacy_risk_mean: float
    alpha: float

    def calculate_resource_vector(self):
        return np.array([self.ram, self.alpha * self.tier_factor * self.privacy_risk_mean])

# Greedy algorithm respecting both topologies in a single unified decision process
def greedy_selection(entities, model_tiers, spatial_budget, privacy_budget, decision_budget, beta):
    entity_resource_vectors = [entity.calculate_resource_vector(beta, decision_budget) for entity in entities]
    model_tier_resource_vectors = [model_tier.calculate_resource_vector() for model_tier in model_tiers]
    A = np.array(entity_resource_vectors + model_tier_resource_vectors)
    x = (
        np.array([1.0] * len(entities) + [0.0] * len(model_tiers))
        if spatial_budget == 0.0
        else np.array([0.0] * len(entities) + [1.0] * len(model_tiers))
    )
    A_T_x = np.dot(A.T, x)
    constraint = np.array([spatial_budget, privacy_budget, decision_budget])
    if np.any(A_T_x > constraint):
        return None
    return A.T @ x

# Haversine distance function from the Possum filter algorithm
def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in metres between two (lat,lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000

if __name__ == "__main__":
    # Smoke test
    entities = [
        Entity("e1", 37.7749, -122.4194, "category1", evidence_score=1.0),
        Entity("e2", 37.7859, -122.4364, "category2", delay_score=2.0),
        Entity("e3", 37.7969, -122.4574, "category3", scarcity_score=3.0),
    ]
    model_tiers = [
        ModelTier("m1", 1024.0, 1.0, 0.5, 1.0),
        ModelTier("m2", 2048.0, 2.0, 0.8, 1.5),
        ModelTier("m3", 4096.0, 3.0, 0.9, 2.0),
    ]
    spatial_budget = 10000.0
    privacy_budget = 0.5
    decision_budget = 10.0
    beta = 1.0
    result = greedy_selection(entities, model_tiers, spatial_budget, privacy_budget, decision_budget, beta)
    print(result)