# DARWIN HAMMER — match 4267, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_percep_m1880_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_possum_filter_m1001_s0.py (gen4)
# born: 2026-05-29T23:54:36Z

import math
import random
import sys
import pathlib
import numpy as np

Vector = np.ndarray


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


def compute_phash(values: list[float]) -> int:
    """Simple perceptual hash: compare each value to the mean of the first 64."""
    if not values:
        return 0
    avg = sum(values[:64]) / min(64, len(values))
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


class Sheaf:
    """Very small sheaf implementation: stores a section per node."""

    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = node_dims
        self.edges = edges
        self.sections: dict = {}

    def set_section(self, node, value: Vector) -> None:
        expected = self.node_dims.get(node)
        if expected is None:
            raise KeyError(f"Node {node} unknown")
        if value.shape[0] != expected:
            raise ValueError(f"Invalid section dimension for node {node}")
        self.sections[node] = value


class Entity:
    """Entity with spatial location and category."""

    def __init__(self, id: str, lat: float, lon: float, category: str):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.category = category


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Haversine distance between two points on a sphere."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def compute_expected_value(entity: Entity, sheaf: Sheaf) -> float:
    """Compute the expected value of an entity based on its spatial location and category."""
    phash = compute_phash(sheaf.sections.get(entity.id, np.zeros(64)).tolist())
    distance = haversine_m((entity.lat, entity.lon), (0.0, 0.0))
    return gaussian(distance, epsilon=0.1) * phash


def compute_ollivier_ricci_curvature(entities: list[Entity], sheaf: Sheaf) -> float:
    """Compute the Ollivier-Ricci curvature for a list of entities."""
    distances = [haversine_m((entity.lat, entity.lon), (0.0, 0.0)) for entity in entities]
    return np.mean(distances)


def compute_regret_weighted_strategy(entities: list[Entity], sheaf: Sheaf) -> list[float]:
    """Compute the regret-weighted strategy for a list of entities."""
    expected_values = [compute_expected_value(entity, sheaf) for entity in entities]
    ollivier_ricci_curvature = compute_ollivier_ricci_curvature(entities, sheaf)
    weights = [1.0 / (1.0 + math.exp(-x - ollivier_ricci_curvature)) for x in expected_values]
    return weights


def hybrid_operation(entities: list[Entity], sheaf: Sheaf) -> list[float]:
    """Demonstrates the hybrid operation by computing the regret-weighted strategy and 
    applying it to the entities."""
    weights = compute_regret_weighted_strategy(entities, sheaf)
    return weights


def spatial_diversity_constraint(entities: list[Entity], weights: list[float], threshold: float = 0.5) -> list[float]:
    """Apply the spatial diversity constraint to filter out entities with high similarity to already selected entities."""
    filtered_weights = []
    for i, weight in enumerate(weights):
        similarity = np.mean([haversine_m((entities[i].lat, entities[i].lon), (entities[j].lat, entities[j].lon)) for j in range(len(entities)) if i != j])
        if similarity > threshold:
            filtered_weights.append(0.0)
        else:
            filtered_weights.append(weight)
    return filtered_weights


if __name__ == "__main__":
    node_dims = {0: 64}
    edges = []
    sheaf = Sheaf(node_dims, edges)
    sheaf.set_section(0, np.random.rand(64))

    entities = [Entity(f"entity_{i}", random.uniform(-90.0, 90.0), random.uniform(-180.0, 180.0), "category") for i in range(10)]

    weights = hybrid_operation(entities, sheaf)
    filtered_weights = spatial_diversity_constraint(entities, weights)
    print(filtered_weights)