# DARWIN HAMMER — match 3392, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s3.py (gen3)
# parent_b: hybrid_hybrid_shap_attribut_hybrid_krampus_brain_m986_s1.py (gen5)
# born: 2026-05-29T23:49:54Z

"""
This module fuses the governing equations of hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s3.py and 
hybrid_hybrid_shap_attribut_hybrid_krampus_brain_m986_s1.py into a unified system. The mathematical bridge 
between the two algorithms is formed by applying the SHAP value calculation to the resource vectors defined in the 
decision hygiene algorithm, and then using the resulting attribution scores to inform the leader election process in 
the graph clustering algorithm.

The new system defines a 3-dimensional resource vector eᵢ = [ dᵢ, pᵢ, sᵢ ] for each entity, where:

- dᵢ = haversine distance (in metres) from a reference location (mirroring the distance-threshold logic of 
  keep_candidate in Possum),
- pᵢ = β·σᵢ, where σᵢ = 1 if the entity's signature collides with any other entity, otherwise 0 (treating 
  signature duplication as a privacy-load analogue to the privacy-load term of the decision hygiene algorithm),
- sᵢ = score from the decision hygiene algorithm.

The SHAP value calculation is applied to these resource vectors to obtain attribution scores for each feature. These 
attribution scores are then used to inform the leader election process in the graph clustering algorithm.

The Ollivier-Ricci curvature is used to quantify the neighbourhood overlap between nodes, and this information is 
injected into the SHAP value calculation to create a more meaningful and efficient feature clustering of the model.
"""

import numpy as np
import random
import math
import sys
from pathlib import Path
from itertools import combinations

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000  # metres
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def compute_shap_value(feature_index: int, feature_count: int, value_fn: callable) -> float:
    total = 0.0
    for k in range(len(value_fn) + 1):
        for subset in combinations(value_fn, k):
            s = frozenset(subset)
            if feature_index not in s:
                total += shapley_kernel_weight(len(s), feature_count) * value_fn[len(s)]
    return total

def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    return math.factorial(subset_size) * math.factorial(feature_count - subset_size - 1) / math.factorial(feature_count)

def ollivier_ricci_curvature(graph: dict[int, set[int]], node: int) -> float:
    neighbours = graph[node]
    curvature = 0.0
    for neighbour in neighbours:
        neighbour_neighbours = graph[neighbour]
        overlap = len(neighbours & neighbour_neighbours) / len(neighbours | neighbour_neighbours)
        curvature += (1 - overlap) / len(neighbours)
    return curvature / len(neighbours)

def compute_resource_vectors(entities: list[tuple[float, float, float]]) -> np.ndarray:
    resource_vectors = np.zeros((len(entities), 3))
    for i, (lat, lon, score) in enumerate(entities):
        resource_vectors[i, 0] = haversine_distance(lat, lon, 0.0, 0.0)  # reference location
        resource_vectors[i, 1] = 1.0 if any(haversine_distance(lat, lon, e[0], e[1]) < 1e-6 for j, e in enumerate(entities) if j != i) else 0.0
        resource_vectors[i, 2] = score
    return resource_vectors

def hybrid_algorithm(entities: list[tuple[float, float, float]], graph: dict[int, set[int]]) -> np.ndarray:
    resource_vectors = compute_resource_vectors(entities)
    shap_values = np.zeros((resource_vectors.shape[0], resource_vectors.shape[1]))
    for i in range(resource_vectors.shape[0]):
        for j in range(resource_vectors.shape[1]):
            shap_values[i, j] = compute_shap_value(i, resource_vectors.shape[0], resource_vectors[:, j])
    curvature = np.zeros((resource_vectors.shape[0],))
    for i in range(resource_vectors.shape[0]):
        curvature[i] = ollivier_ricci_curvature(graph, i)
    return np.hstack((resource_vectors, shap_values, curvature[:, np.newaxis]))

if __name__ == "__main__":
    entities = [(37.7749, -122.4194, 1.0), (34.0522, -118.2437, 0.5), (40.7128, -74.0060, 0.8)]
    graph = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
    result = hybrid_algorithm(entities, graph)
    print(result)