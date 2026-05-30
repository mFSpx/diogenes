# DARWIN HAMMER — match 3864, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_voronoi_parti_m1327_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_gini_c_m2706_s2.py (gen6)
# born: 2026-05-29T23:52:02Z

"""
This module fuses the hybrid_hybrid_hybrid_hybrid_hybrid_voronoi_parti_m1327_s0.py and 
hybrid_hybrid_hybrid_bandit_hybrid_hybrid_gini_c_m2706_s2.py algorithms by integrating 
the Shannon entropy of decision hygiene feature counts with the Gini-guided 
structural gain and temperature-aware activity.

The mathematical bridge lies in the application of the Gini-guided structural 
gain to weight the Voronoi region assignments, which are then modulated by 
the temperature-aware activity.

The hybrid algorithm combines the use of Shannon entropy to score decision 
hygiene features with the spatial partitioning provided by the Voronoi diagram, 
and the Gini-guided structural gain and temperature-aware activity to modulate 
the region assignments and generate procedural entities with unique properties.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

def shannon_entropy(counts):
    """Compute Shannon entropy from a list of counts."""
    total = sum(counts)
    entropy = 0.0
    for count in counts:
        if count > 0:
            prob = count / total
            entropy -= prob * math.log2(prob)
    return entropy

def decision_hygiene_entropy(feature_counts):
    """Compute Shannon entropy of decision hygiene feature counts."""
    return shannon_entropy(feature_counts)

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point, seeds):
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def temperature_activity(T: float, A0: float = 1.0, E: float = 0.65, T_opt: float = 25.0) -> float:
    """
    Schoolfield‐type temperature activity gate.
    Returns A(T) ∈ [0,1] where the activity peaks at T_opt.
    """
    if T <= -273.15:
        return 0.0
    kelvin = T + 273.15
    opt_kelvin = T_opt + 273.15
    activity = A0 * math.exp(-E / kelvin) * (kelvin / opt_kelvin)
    return max(0.0, min(1.0, activity))

def gini_coefficient(values: list) -> float:
    """
    Compute the Gini coefficient from a list of values.
    """
    values = sorted(values)
    n = len(values)
    index = np.arange(1, n+1)
    n_values = n * values
    return ((np.sum((2 * index - n - 1) * values)) / (n * np.sum(values)))

def structural_gain(log_probabilities: list) -> list:
    """
    Compute the structural gain using the tropical max-plus operator.
    """
    return [max(0, p) for p in log_probabilities]

def assign(points, seeds, weights, T):
    regions = {i: [] for i in range(len(seeds))}
    activity = temperature_activity(T)
    for p in points:
        region_idx = nearest(p, seeds)
        regions[region_idx].append(p)
        # weight the region assignment using decision hygiene scores and temperature activity
        region_weight = weights[region_idx] * activity
        # adjust the region assignment based on the weight
        if random.random() < region_weight:
            regions[region_idx].append(p)
    return regions

def procedural_entity_generator(points, seeds, feature_counts, T):
    weights = [0.0] * len(seeds)
    entropy = decision_hygiene_entropy(feature_counts)
    gini_values = [entropy / (1 + i) for i in range(len(seeds))]
    gini = gini_coefficient(gini_values)
    log_probabilities = [math.log(p) for p in gini_values]
    structural_gains = structural_gain(log_probabilities)
    for i in range(len(seeds)):
        weights[i] = structural_gains[i] * gini
    regions = assign(points, seeds, weights, T)
    entities = []
    for region_idx, region_points in regions.items():
        entity = {
            'region_idx': region_idx,
            'points': region_points,
            'weight': weights[region_idx]
        }
        entities.append(entity)
    return entities

if __name__ == "__main__":
    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(100)]
    seeds = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]
    feature_counts = [random.randint(1, 100) for _ in range(10)]
    T = 25.0
    entities = procedural_entity_generator(points, seeds, feature_counts, T)
    print(len(entities))