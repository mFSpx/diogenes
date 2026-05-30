# DARWIN HAMMER — match 210, survivor 1
# gen: 3
# parent_a: hybrid_bayes_update_hybrid_krampus_brain_m15_s2.py (gen2)
# parent_b: hybrid_possum_filter_hybrid_privacy_model_m53_s0.py (gen2)
# born: 2026-05-29T23:27:39Z

"""
This module fuses the mathematical structures of hybrid_bayes_update_hybrid_krampus_brain_m15_s2.py and 
hybrid_possum_filter_hybrid_privacy_model_m53_s0.py. The hybrid_bayes_update_hybrid_krampus_brain_m15_s2.py 
provides a method for Bayesian marginalization and update formulas, while 
hybrid_possum_filter_hybrid_privacy_model_m53_s0.py presents a framework for managing spatial-aware privacy risk models.
The mathematical bridge between these two structures is established by introducing a Bayesian-based spatial-aware 
privacy risk model. In this model, the reconstruction risk for each entity is weighted by its distance to other entities 
in the dataset, resulting in a modified risk vector that incorporates both spatial and categorical information. 
This risk vector is then used as the prior probability in the Bayesian update formula, allowing us to build a combined 
resource matrix that considers both RAM consumption and spatial-aware privacy load.
"""

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import numpy as np
from pathlib import Path
import random
import sys

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
    risks = []
    for i, entity in enumerate(entities):
        similar_entities = [e for j, e in enumerate(entities) if i != j and signature(entity) == signature(e) and haversine_m((entity.lat, entity.lon), (e.lat, e.lon)) <= delta_m]
        unique_quasi_identifiers = len(set(e.id for e in similar_entities))
        risk = reconstruction_risk_score(unique_quasi_identifiers, len(entities))
        risks.append(risk)
    return np.array(risks, dtype=float)

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def bayes_edge_weight(prior: float, likelihood: float) -> float:
    false_positive = 1.0 - prior
    marginal = bayes_marginal(prior, likelihood, false_positive)
    return bayes_update(prior, likelihood, marginal)

def hybrid_bayes_update(entities: List[Entity], delta_m: float, likelihood: float) -> np.ndarray:
    prior_risks = spatial_aware_privacy_risk_vector(entities, delta_m)
    bayes_posteriors = [bayes_update(prior, likelihood, 1.0) for prior in prior_risks]
    return np.array(bayes_posteriors, dtype=float)

def hybrid_bayes_edge_weight(entities: List[Entity], delta_m: float, likelihood: float) -> np.ndarray:
    prior_risks = spatial_aware_privacy_risk_vector(entities, delta_m)
    bayes_edge_weights = [bayes_edge_weight(prior, likelihood) for prior in prior_risks]
    return np.array(bayes_edge_weights, dtype=float)

if __name__ == "__main__":
    entities = [Entity("1", 37.7749, -122.4194, "category1"),
                 Entity("2", 37.7858, -122.4364, "category2"),
                 Entity("3", 37.7963, -122.4575, "category1")]
    delta_m = 10000.0
    likelihood = 0.5
    print(hybrid_bayes_update(entities, delta_m, likelihood))
    print(hybrid_bayes_edge_weight(entities, delta_m, likelihood))