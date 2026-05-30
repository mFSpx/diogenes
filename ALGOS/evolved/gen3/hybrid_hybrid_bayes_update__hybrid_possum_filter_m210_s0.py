# DARWIN HAMMER — match 210, survivor 0
# gen: 3
# parent_a: hybrid_bayes_update_hybrid_krampus_brain_m15_s2.py (gen2)
# parent_b: hybrid_possum_filter_hybrid_privacy_model_m53_s0.py (gen2)
# born: 2026-05-29T23:27:39Z

"""
Hybrid Bayesian-Ollivier Ricci and Spatial-Aware Privacy Risk Model

Parents:
- hybrid_bayes_update_hybrid_krampus_brain_m15_s2.py (Algorithm A): Provides Bayesian marginalization and update formulas, 
  as well as feature extraction, graph construction, and Ollivier‑Ricci curvature computation.
- hybrid_possum_filter_hybrid_privacy_model_m53_s0.py (Algorithm B): Provides a method for filtering entities based on 
  their spatial distance and category similarity, and a framework for managing model resources under RAM ceiling and tier 
  exclusivity constraints.

Mathematical Bridge:
The mathematical bridge between these two structures is established by interpreting the spatial-aware privacy risk vector 
as prior probabilities on graph nodes. For each ordered pair (i, j) we compute a Bayesian posterior 
    w_{ij} = P(H_i | E_j) = prior_i * likelihood_j / marginal_{ij},
where 
    marginal_{ij} = likelihood_j * prior_i + false_positive_i * (1 - prior_i)
and 
    false_positive_i = 1 - prior_i.
These posteriors become edge weights that define the adjacency of a graph. The resulting graph is then used to compute 
the Ollivier-Ricci curvature, which can be used to analyze the geometric structure of the graph. The spatial-aware 
privacy risk vector is used to weight the reconstruction risk for each entity, allowing for a more nuanced analysis of 
privacy risks in the context of spatial relationships.

"""

from __future__ import annotations

import random
import math
import sys
import pathlib
import numpy as np
from typing import Dict, Tuple, List, Set
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Algorithm A – Bayesian primitives
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Algorithm B – Spatial-aware privacy risk model
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Hybrid Algorithm
# ----------------------------------------------------------------------
def hybrid_bayes_privacy_risk(entities: List[Entity], delta_m: float) -> Dict[Tuple[str, str], float]:
    risk_vector = spatial_aware_privacy_risk_vector(entities, delta_m)
    prior_probabilities = risk_vector / np.sum(risk_vector)
    edge_weights = {}
    for i, entity_i in enumerate(entities):
        for j, entity_j in enumerate(entities):
            if i != j:
                prior_i = prior_probabilities[i]
                likelihood_j = prior_probabilities[j]
                edge_weights[(entity_i.id, entity_j.id)] = bayes_edge_weight(prior_i, likelihood_j)
    return edge_weights

def ollivier_ricci_curvature(edge_weights: Dict[Tuple[str, str], float], entities: List[Entity]) -> float:
    # Simplified Ollivier-Ricci curvature computation for demonstration purposes
    curvature = 0.0
    for edge, weight in edge_weights.items():
        curvature += weight
    return curvature / len(edge_weights)

def analyze_privacy_risks(entities: List[Entity], delta_m: float) -> Tuple[Dict[Tuple[str, str], float], float]:
    edge_weights = hybrid_bayes_privacy_risk(entities, delta_m)
    curvature = ollivier_ricci_curvature(edge_weights, entities)
    return edge_weights, curvature

if __name__ == "__main__":
    entities = [
        Entity("1", 37.7749, -122.4194, "A"),
        Entity("2", 37.7859, -122.4364, "A"),
        Entity("3", 37.7959, -122.4574, "B"),
    ]
    delta_m = 1000.0
    edge_weights, curvature = analyze_privacy_risks(entities, delta_m)
    print("Edge Weights:")
    for edge, weight in edge_weights.items():
        print(f"{edge}: {weight}")
    print(f"Ollivier-Ricci Curvature: {curvature}")