# DARWIN HAMMER — match 1133, survivor 1
# gen: 4
# parent_a: hybrid_privacy_model_pool_m7_s2.py (gen1)
# parent_b: hybrid_hybrid_semantic_neig_hybrid_hybrid_geomet_m116_s1.py (gen3)
# born: 2026-05-29T23:33:01Z

"""
Hybrid module combining the hybrid_privacy_model_pool_m7_s2.py and hybrid_hybrid_semantic_neig_hybrid_hybrid_geomet_m116_s1.py.
The mathematical bridge between the two lies in the representation of the semantic neighborhoods as multivectors, 
which allows for the use of the geometric product to compute the similarity between documents, and the use of the 
Voronoi partitioning to assign points to these neighborhoods. In this hybrid module, we integrate the privacy risk 
vector from the hybrid_privacy_model_pool_m7_s2.py with the geometric product and Voronoi partitioning from 
hybrid_hybrid_semantic_neig_hybrid_hybrid_geomet_m116_s1.py. We use the geometric product to represent the 
semantic neighborhoods as multivectors and then use the Voronoi partitioning to assign points to these neighborhoods 
based on their proximity to the seeds, while taking into account the privacy risk vector.

"""

import math
import numpy as np
import random
import sys
from pathlib import Path

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Return a normalized reconstruction risk in [0,1]."""
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def _cos(a, b):
    den = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
    return 0.0 if den == 0 else sum(x*y for x, y in zip(a, b)) / den

def pheromone_probabilities(pheromones):
    total = sum(pheromones)
    return [p / total for p in pheromones]

def entropy(probabilities, eps=1e-12):
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p / total) * math.log(max(p / total, eps)) for p in probabilities if p > 0)

def expected_entropy(p_hit, hit_state, miss_state):
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)

Point = tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Point], seeds: list[Point]) -> list[int]:
    return [nearest(point, seeds) for point in points]

def model_resource_matrix(models, ram_consumption, tier_exclusivity_penalty):
    """Return a combined resource matrix A whose rows are models and columns are [RAM, privacy-load]."""
    return np.array([[ram, tier_exclusivity_penalty] for ram, tier_exclusivity_penalty in zip(ram_consumption, tier_exclusivity_penalty)])

def select_models_hybrid(models, ram_ceiling, privacy_budget, tier_exclusivity_penalty, ram_consumption):
    """Select models based on the composite constraint L ≤ [ram_ceiling, privacy_budget]."""
    A = model_resource_matrix(models, ram_consumption, tier_exclusivity_penalty)
    # Select models that satisfy the composite constraint
    selected_models = []
    for i, model in enumerate(models):
        if A[i, 0] <= ram_ceiling and A[i, 1] <= privacy_budget:
            selected_models.append(model)
    return selected_models

def semantic_neighborhood_search(models, points, seeds):
    """Perform semantic neighborhood search using the geometric product and Voronoi partitioning."""
    # Assign points to neighborhoods based on their proximity to the seeds
    neighborhood_assignments = assign(points, seeds)
    # Compute the geometric product for each neighborhood
    geometric_products = []
    for i, point in enumerate(points):
        neighborhood = neighborhood_assignments[i]
        geometric_product = _cos(point, seeds[neighborhood])
        geometric_products.append(geometric_product)
    return geometric_products

if __name__ == "__main__":
    # Test the hybrid module
    models = ["model1", "model2", "model3"]
    ram_consumption = [10, 20, 30]
    tier_exclusivity_penalty = [1, 2, 3]
    ram_ceiling = 50
    privacy_budget = 10
    points = [(1, 2), (3, 4), (5, 6)]
    seeds = [(0, 0), (4, 4), (8, 8)]
    selected_models = select_models_hybrid(models, ram_ceiling, privacy_budget, tier_exclusivity_penalty, ram_consumption)
    geometric_products = semantic_neighborhood_search(models, points, seeds)
    print("Selected models:", selected_models)
    print("Geometric products:", geometric_products)