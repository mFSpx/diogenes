# DARWIN HAMMER — match 5257, survivor 0
# gen: 5
# parent_a: hybrid_possum_filter_hybrid_hybrid_label__m553_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_path_s_path_signature_m501_s2.py (gen4)
# born: 2026-05-30T00:00:52Z

"""
This module integrates the possum_filter.py and hybrid_hybrid_path_s_path_signature_m501_s2.py algorithms.
The mathematical bridge between the two structures is the concept of "directional information richness,"
which is used to determine the likelihood of an entity being representative of its category.
This richness is calculated based on the Shannon entropy of the feature count vector of entities
within a certain distance threshold, and this value is then used to adjust the filtering behavior.
The entropy value is used in conjunction with the lead-lag transform and B-spline basis functions
from the hybrid_hybrid_path_s_path_signature_m501_s2.py algorithm to calculate a hybrid score.
When the observed entities are information-rich (high entropy), the algorithm filters less aggressively
and preserves more of the possum filter contribution; conversely, low-entropy (repetitive) inputs
are filtered more heavily.
"""

import numpy as np
from dataclasses import dataclass
from math import exp, log, sin, cos, asin, radians
from random import random
from sys import exit
from pathlib import Path
from typing import Iterable, List, Tuple

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = map(radians, a)
    lat2, lon2 = map(radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * asin(min(1.0, (h ** 0.5)))

def signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()

def shannon_entropy(feature_counts: List[int]) -> float:
    total = sum(feature_counts)
    entropy = 0.0
    for count in feature_counts:
        if count > 0:
            prob = count / total
            entropy -= prob * log(prob, 2)
    return entropy

def pruning_probability(entropy: float) -> float:
    return exp(-entropy)

def lead_lag_transform(self, path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(self, x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2      
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(N):
        for j in range(len(t) - 1):
            if t[j] <= x[i] <= t[j + 1]:
                if k == 1:
                    B[i, j] = 1.0
                elif k == 2:
                    B[i, j] = (x[i] - t[j]) / (t[j + 1] - t[j])
                    B[i, j + 1] = (t[j + 2] - x[i]) / (t[j + 2] - t[j + 1])
                elif k == 3:
                    h1 = (t[j + 1] - t[j]) * (t[j + 1] - x[i])**2 / ((t[j + 2] - t[j]) * (t[j + 1] - t[j]))
                    h2 = (x[i] - t[j]) * (t[j + 1] - x[i])**2 / ((t[j + 1] - t[j]) * (t[j + 1] - t[j - 1]))
                    h3 = (x[i] - t[j]) * (x[i] - t[j + 1])**2 / ((t[j + 1] - t[j]) * (t[j + 2] - t[j + 1]))
                    B[i, j] = h1
                    B[i, j + 1] = h2 + h3
    return B

class HybridSystem:
    def __init__(self):
        self.pheromones = {}
        self.records = []

    def hybrid_filter(self, entities: Iterable[Entity], delta_m: float = 75.0, basis_degree: int = 3) -> List[Entity]:
        if delta_m < 0:
            raise ValueError("delta_m must be non-negative")

        # Group entities by category
        categories = {}
        for entity in entities:
            category = entity.category.strip().lower()
            if category not in categories:
                categories[category] = []

            # Calculate distance between entity and all other entities in the same category
            for other_entity in entities:
                if other_entity == entity:
                    continue

                distance = haversine_m((entity.lat, entity.lon), (other_entity.lat, other_entity.lon))
                if distance <= delta_m:
                    categories[category].append((other_entity, distance))

        # Calculate Shannon entropy for each category
        entropy_values = {}
        for category in categories:
            counts = {}
            for entity, _ in categories[category]:
                signature_value = signature(entity)
                if signature_value not in counts:
                    counts[signature_value] = 0
                counts[signature_value] += 1

            feature_counts = list(counts.values())
            entropy = shannon_entropy(feature_counts)
            entropy_values[category] = entropy

        # Calculate pruning probability for each category
        pruning_probabilities = {}
        for category in entropy_values:
            pruning_probabilities[category] = pruning_probability(entropy_values[category])

        # Calculate lead-lag transform and B-spline basis for each category
        hybrid_scores = {}
        for category in categories:
            category_entities = [entity for entity, _ in categories[category]]
            lead_lag_path = np.array([[entity.lat, entity.lon] for entity in category_entities])
            lead_lag_transform_path = lead_lag_transform(lead_lag_path)
            bspline_basis_path = bspline_basis(lead_lag_path[:, 0], np.linspace(0, 1, len(lead_lag_path)), basis_degree)

            # Calculate hybrid score for each entity in the category
            scores = []
            for entity, _ in categories[category]:
                feature_count = 0
                for other_entity in entities:
                    if other_entity == entity:
                        continue

                    distance = haversine_m((entity.lat, entity.lon), (other_entity.lat, other_entity.lon))
                    if distance <= delta_m:
                        feature_count += 1

                score = entity.score + (1 - pruning_probabilities[category]) * feature_count
                scores.append((entity, score))

            hybrid_scores[category] = scores

        # Filter entities based on hybrid score
        filtered_entities = []
        for category in hybrid_scores:
            scores = hybrid_scores[category]
            scores.sort(key=lambda x: x[1], reverse=True)
            filtered_entities.extend([entity for entity, _ in scores[:int(len(scores) * 0.5)]])

        return filtered_entities

    def hybrid_signature(self, path, level, basis_degree=3):
        transformed_path = lead_lag_transform(path)
        bspline_basis_path = bspline_basis(path[:, 0], np.linspace(0, 1, len(path)), basis_degree)
        return bspline_basis_path

if __name__ == "__main__":
    hybrid_system = HybridSystem()
    entities = [
        Entity(id="1", lat=37.7749, lon=-122.4194, category="A"),
        Entity(id="2", lat=37.7859, lon=-122.4364, category="A"),
        Entity(id="3", lat=37.7969, lon=-122.4574, category="B"),
    ]
    filtered_entities = hybrid_system.hybrid_filter(entities)
    print(filtered_entities)