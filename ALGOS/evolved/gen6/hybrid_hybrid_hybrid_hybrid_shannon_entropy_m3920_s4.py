# DARWIN HAMMER — match 3920, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1300_s2.py (gen5)
# parent_b: shannon_entropy.py (gen0)
# born: 2026-05-29T23:52:41Z

"""
Hybrid Algorithm: Spatial-Privacy Risk × Morphological Similarity × Shannon Entropy Scheduler

This module fuses the governing equations of:
1. hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1300_s2.py (DARWIN HAMMER)
2. shannon_entropy.py

The mathematical bridge between the two parents is established by injecting the Shannon entropy 
of the morphological similarity distribution into the health calculation of each EngineEndpoint. 
The entropy term represents the uncertainty or randomness in the morphology similarity.

The health of an EngineEndpoint is calculated as:

    health_i = (1 - privacy_risk_i) * (1 - avg_ssim_i) * (1 - normalized_entropy_i) * resource_factor_i

where `avg_ssim_i` is the mean SSIM of endpoint *i* with all other endpoints, 
`normalized_entropy_i` is the Shannon entropy of the SSIM distribution of endpoint *i*, 
and `resource_factor_i` scales with the RAM/VRAM budget of the ModelTier that hosts the endpoint.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (copied/extended from the parents)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"

def compute_spatial_privacy_risk(entity: Entity, other_entities: List[Entity]) -> float:
    """
    Compute the spatial-privacy risk of an entity.

    Args:
    entity (Entity): The entity to compute the risk for.
    other_entities (List[Entity]): The list of other entities.

    Returns:
    float: The spatial-privacy risk.
    """
    total_distance = 0.0
    for other_entity in other_entities:
        distance = haversine_distance(entity.lat, entity.lon, other_entity.lat, other_entity.lon)
        total_distance += distance
    avg_distance = total_distance / len(other_entities)
    return 1 / (1 + avg_distance)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the Haversine distance between two points on a sphere.

    Args:
    lat1 (float): The latitude of the first point.
    lon1 (float): The longitude of the first point.
    lat2 (float): The latitude of the second point.
    lon2 (float): The longitude of the second point.

    Returns:
    float: The Haversine distance.
    """
    earth_radius = 6371  # kilometers
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return earth_radius * c

def compute_morphology_ssim_matrix(endpoints: List[EngineEndpoint]) -> np.ndarray:
    """
    Compute the SSIM matrix of a list of EngineEndpoints.

    Args:
    endpoints (List[EngineEndpoint]): The list of EngineEndpoints.

    Returns:
    np.ndarray: The SSIM matrix.
    """
    num_endpoints = len(endpoints)
    ssim_matrix = np.zeros((num_endpoints, num_endpoints))
    for i in range(num_endpoints):
        for j in range(i+1, num_endpoints):
            ssim = structural_similarity_index(endpoints[i].morphology, endpoints[j].morphology)
            ssim_matrix[i, j] = ssim
            ssim_matrix[j, i] = ssim
    return ssim_matrix

def structural_similarity_index(morphology1: Morphology, morphology2: Morphology) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two morphologies.

    Args:
    morphology1 (Morphology): The first morphology.
    morphology2 (Morphology): The second morphology.

    Returns:
    float: The SSIM.
    """
    # For simplicity, assume SSIM is the cosine similarity between the morphology vectors
    morphology_vector1 = np.array([morphology1.length, morphology1.width, morphology1.height, morphology1.mass])
    morphology_vector2 = np.array([morphology2.length, morphology2.width, morphology2.height, morphology2.mass])
    dot_product = np.dot(morphology_vector1, morphology_vector2)
    magnitude1 = np.linalg.norm(morphology_vector1)
    magnitude2 = np.linalg.norm(morphology_vector2)
    return dot_product / (magnitude1 * magnitude2)

def shannon_entropy(values: List[float]) -> float:
    """
    Compute the Shannon entropy of a list of values.

    Args:
    values (List[float]): The list of values.

    Returns:
    float: The Shannon entropy.
    """
    probabilities = np.array(values) / sum(values)
    return -np.sum(probabilities * np.log2(probabilities))

def compute_health(endpoint: EngineEndpoint, ssim_matrix: np.ndarray, model_tier: ModelTier, other_endpoints: List[EngineEndpoint]) -> float:
    """
    Compute the health of an EngineEndpoint.

    Args:
    endpoint (EngineEndpoint): The EngineEndpoint to compute the health for.
    ssim_matrix (np.ndarray): The SSIM matrix.
    model_tier (ModelTier): The ModelTier that hosts the endpoint.
    other_endpoints (List[EngineEndpoint]): The list of other EngineEndpoints.

    Returns:
    float: The health.
    """
    index = other_endpoints.index(endpoint)
    avg_ssim = np.mean(ssim_matrix[index])
    ssim_values = ssim_matrix[index]
    entropy = shannon_entropy(ssim_values)
    normalized_entropy = entropy / math.log2(len(ssim_values))
    resource_factor = model_tier.ram_mb / (model_tier.ram_mb + model_tier.vram_mb)
    privacy_risk = compute_spatial_privacy_risk(Entity(id="dummy", lat=0.0, lon=0.0, category="dummy"), [Entity(id="dummy", lat=0.0, lon=0.0, category="dummy")])
    return (1 - privacy_risk) * (1 - avg_ssim) * (1 - normalized_entropy) * resource_factor

def allocate_resources_based_on_health(endpoints: List[EngineEndpoint], model_tiers: List[ModelTier]) -> Dict[EngineEndpoint, ModelTier]:
    """
    Allocate resources to EngineEndpoints based on their health.

    Args:
    endpoints (List[EngineEndpoint]): The list of EngineEndpoints.
    model_tiers (List[ModelTier]): The list of ModelTiers.

    Returns:
    Dict[EngineEndpoint, ModelTier]: The allocation of resources.
    """
    ssim_matrix = compute_morphology_ssim_matrix(endpoints)
    allocation = {}
    for endpoint in endpoints:
        best_model_tier = max(model_tiers, key=lambda model_tier: compute_health(endpoint, ssim_matrix, model_tier, endpoints))
        allocation[endpoint] = best_model_tier
    return allocation

if __name__ == "__main__":
    # Smoke test
    endpoint1 = EngineEndpoint(engine_id="engine1", channel="channel1", residency="residency1", runtime="runtime1", resource_class="resource_class1", always_on=True, endpoint="endpoint1", capabilities=["capability1"], morphology=Morphology(length=1.0, width=2.0, height=3.0, mass=4.0))
    endpoint2 = EngineEndpoint(engine_id="engine2", channel="channel2", residency="residency2", runtime="runtime2", resource_class="resource_class2", always_on=False, endpoint="endpoint2", capabilities=["capability2"], morphology=Morphology(length=5.0, width=6.0, height=7.0, mass=8.0))
    model_tier1 = ModelTier(name="model_tier1", ram_mb=1024, tier="tier1", vram_mb=512)
    model_tier2 = ModelTier(name="model_tier2", ram_mb=2048, tier="tier2", vram_mb=1024)
    endpoints = [endpoint1, endpoint2]
    model_tiers = [model_tier1, model_tier2]
    allocation = allocate_resources_based_on_health(endpoints, model_tiers)
    print(allocation)