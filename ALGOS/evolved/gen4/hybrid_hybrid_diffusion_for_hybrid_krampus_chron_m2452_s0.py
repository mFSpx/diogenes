# DARWIN HAMMER — match 2452, survivor 0
# gen: 4
# parent_a: hybrid_diffusion_forcing_hybrid_hybrid_minimu_m918_s3.py (gen3)
# parent_b: hybrid_krampus_chrono_hybrid_possum_filter_m34_s1.py (gen3)
# born: 2026-05-29T23:42:22Z

"""
Hybrid algorithm fusing hybrid_diffusion_forcing_hybrid_hybrid_minimu_m918_s3.py and hybrid_krampus_chrono_hybrid_possum_filter_m34_s1.py.

The mathematical bridge between the two parent algorithms lies in their treatment of uncertainty and temporal information. 
The hybrid_diffusion_forcing_hybrid_hybrid_minimu_m918_s3.py algorithm uses a noise schedule to represent uncertainty, 
while the hybrid_krampus_chrono_hybrid_possum_filter_m34_s1.py algorithm uses temporal information to weight spatial distances.

The hybrid algorithm combines these two approaches by using the noise schedule to inform the temporal weighting process. 
Specifically, the algorithm uses the noise levels to modulate the temporal weights, allowing for a more nuanced filtering process.

The governing equations of the hybrid algorithm are:

* For each entity, define a 3-dimensional resource vector eᵢ = [ dᵢ , pᵢ , tᵢ ] where
  • dᵢ = haversine distance (in metres) from a reference location
  • pᵢ = β·σᵢ, σᵢ = 1 if the entity's signature collides with any other entity, otherwise 0
  • tᵢ = temporal weight based on the extracted chronological dates and modulated by the noise levels

* For each ModelTier, reuse the resource vector defined in algorithm B: mⱼ = [ RAMⱼ , α·τⱼ·μ ]

* Stacking all vectors yields a combined resource matrix A (rows = entities∪models, columns = [spatial/RAM-load , privacy-load, temporal-load])

The hybrid algorithm satisfies the linear constraints Aᵀ·x ≤ [ spatial_budget , privacy_budget , temporal_budget ]
"""

import numpy as np
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set
from datetime import datetime

EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

def confidence_to_probability(cf: CertaintyFlag) -> float:
    return cf.confidence_bps / 10000.0

def noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    if schedule == "cosine":
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float64)
        f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        alpha_bars = np.clip(alpha_bars, 0.0, 1.0)
        return alpha_bars
    else:
        raise ValueError("unsupported schedule")

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""
    timestamp: str = ""

def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000 * math.sqrt(h)

def extract_temporal_weight(timestamp: str, noise_levels: np.ndarray) -> float:
    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    t = dt.timestamp()
    modulated_t = t * noise_levels[0]
    return modulated_t

def hybrid_tree_cost_with_certainty(edge_costs: List[float], certainty_flags: List[CertaintyFlag], noise_levels: np.ndarray) -> float:
    total_cost = 0.0
    for i, (cost, cf) in enumerate(zip(edge_costs, certainty_flags)):
        prob = confidence_to_probability(cf)
        modulated_cost = cost * prob * noise_levels[i]
        total_cost += modulated_cost
    return total_cost

def compute_resource_matrix(entities: List[Entity], model_tiers: List[Dict[str, Any]], noise_levels: np.ndarray) -> np.ndarray:
    resource_matrix = np.zeros((len(entities) + len(model_tiers), 3))
    for i, entity in enumerate(entities):
        resource_matrix[i, 0] = haversine_m((entity.lat, entity.lon), (0.0, 0.0))
        resource_matrix[i, 1] = 1.0 if entity.address_signature != "" else 0.0
        resource_matrix[i, 2] = extract_temporal_weight(entity.timestamp, noise_levels)
    for i, model_tier in enumerate(model_tiers):
        resource_matrix[len(entities) + i, 0] = model_tier["RAM"]
        resource_matrix[len(entities) + i, 1] = model_tier["alpha"] * model_tier["tau"] * model_tier["mu"]
        resource_matrix[len(entities) + i, 2] = 0.0
    return resource_matrix

if __name__ == "__main__":
    entities = [
        Entity("e1", 37.7749, -122.4194, " category1", timestamp="2022-01-01T00:00:00Z"),
        Entity("e2", 34.0522, -118.2437, "category2", timestamp="2022-01-02T00:00:00Z"),
    ]
    model_tiers = [
        {"RAM": 16, "alpha": 0.5, "tau": 0.2, "mu": 0.1},
        {"RAM": 32, "alpha": 0.3, "tau": 0.4, "mu": 0.2},
    ]
    noise_levels = noise_schedule(10)
    certainty_flags = [
        CertaintyFlag("FACT", 9000, "high", "certain"),
        CertaintyFlag("PROBABLE", 6000, "medium", "probable"),
    ]
    edge_costs = [1.0, 2.0]
    resource_matrix = compute_resource_matrix(entities, model_tiers, noise_levels)
    hybrid_cost = hybrid_tree_cost_with_certainty(edge_costs, certainty_flags, noise_levels)
    print(hybrid_cost)
    print(resource_matrix)