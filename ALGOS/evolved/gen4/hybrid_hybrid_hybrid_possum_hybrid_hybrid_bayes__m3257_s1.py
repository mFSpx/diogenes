# DARWIN HAMMER — match 3257, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_possum_filter_hybrid_caputo_fracti_m1220_s4.py (gen3)
# parent_b: hybrid_hybrid_bayes_claim_k_hybrid_hybrid_distri_m351_s1.py (gen3)
# born: 2026-05-29T23:48:54Z

import math
import numpy as np
import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

def signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def spatial_aware_privacy_risk_vector(entities: List[Entity], delta_m: float) -> np.ndarray:
    risks = np.zeros(len(entities))
    for i, entity in enumerate(entities):
        similar_entities = [e for j, e in enumerate(entities) if i != j and signature(entity) == signature(e) and haversine_m((entity.lat, entity.lon), (e.lat, e.lon)) <= delta_m]
        unique_quasi_identifiers = len(set(e.id for e in similar_entities))
        risk = reconstruction_risk_score(unique_quasi_identifiers, len(entities))
        risks[i] = risk
    return risks

def pulse_force(peak_force: float, steps: int) -> np.ndarray:
    mid = (steps - 1) / 2.0
    denom = max(1.0, mid)
    return np.array([peak_force * max(0.0, 1.0 - abs(i - mid) / denom) for i in range(steps)])

def burst_admission_score(work_value: float, cost_drag: float, urgency_force: float, steps: int = 12) -> float:
    force_series = pulse_force(urgency_force, steps)
    v = 0.0
    x = 0.0
    peak = v
    for force in force_series:
        drag = cost_drag * v * abs(v)
        acc = force - drag
        v = max(0.0, v + acc)
        x += v
        peak = max(peak, v)
    return peak

def hybrid_bayes_update(prior: float, likelihood: float, evidence_score: float) -> float:
    return prior * likelihood * evidence_score

def hybrid_evidence_selection(entities: List[Entity], delta_m: float, work_value: float, cost_drag: float, urgency_force: float) -> List[Entity]:
    risks = spatial_aware_privacy_risk_vector(entities, delta_m)
    scores = np.array([burst_admission_score(work_value, cost_drag, urgency_force) * risk for risk in risks])
    sorted_indices = np.argsort(-scores)
    return [entities[i] for i in sorted_indices]

def hybrid_inference(entities: List[Entity], delta_m: float, work_value: float, cost_drag: float, urgency_force: float, prior: float, likelihood: float) -> float:
    selected_entities = hybrid_evidence_selection(entities, delta_m, work_value, cost_drag, urgency_force)
    evidence_score = np.mean([entity.score for entity in selected_entities])
    return hybrid_bayes_update(prior, likelihood, evidence_score)

if __name__ == "__main__":
    entities = [
        Entity("1", 37.7749, -122.4194, "category1", 0.5),
        Entity("2", 37.7859, -122.4364, "category1", 0.7),
        Entity("3", 37.7963, -122.4575, "category2", 0.3)
    ]
    delta_m = 1000.0
    work_value = 10.0
    cost_drag = 0.1
    urgency_force = 5.0
    prior = 0.5
    likelihood = 0.8
    result = hybrid_inference(entities, delta_m, work_value, cost_drag, urgency_force, prior, likelihood)
    print(result)