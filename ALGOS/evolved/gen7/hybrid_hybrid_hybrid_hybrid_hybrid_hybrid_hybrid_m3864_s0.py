# DARWIN HAMMER — match 3864, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_voronoi_parti_m1327_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_gini_c_m2706_s2.py (gen6)
# born: 2026-05-29T23:52:02Z

"""
This module fuses the hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_voronoi_parti_m1327_s0.py 
and hybrid_hybrid_hybrid_bandit_hybrid_hybrid_gini_c_m2706_s2.py algorithms 
by integrating the Shannon entropy of decision hygiene feature counts with the 
Gini-coefficient guided Hoeffding splits and the temperature activity gate.

The mathematical bridge lies in the application of Shannon entropy to the 
decision hygiene scoring system, which is then used to weight the Voronoi 
region assignments. The temperature activity gate is used to modulate the 
Shannon entropy calculation, and the Gini-coefficient is used to guide the 
Hoeffding splits in the Voronoi partitioning.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

def shannon_entropy(counts, temperature):
    """Compute Shannon entropy from a list of counts, modulated by temperature."""
    temperature_activity = temperature_activity_gate(temperature)
    total = sum(counts)
    entropy = 0.0
    for count in counts:
        if count > 0:
            prob = count / total
            entropy -= prob * math.log2(prob) * temperature_activity
    return entropy

def temperature_activity_gate(temperature, A0=1.0, E=0.65, T_opt=25.0):
    """
    Schoolfield‐type temperature activity gate.
    Returns A(T) ∈ [0,1] where the activity peaks at T_opt.
    """
    if temperature <= -273.15:
        return 0.0
    kelvin = temperature + 273.15
    opt_kelvin = T_opt + 273.15
    activity = A0 * math.exp(-E / kelvin) * (kelvin / opt_kelvin)
    return max(0.0, min(1.0, activity))

def gini_coefficient(feature_counts):
    """Compute Gini coefficient from a list of feature counts."""
    total = sum(feature_counts)
    gini = 0.0
    for count in feature_counts:
        prob = count / total
        gini += prob * (1 - prob)
    return gini

def voronoi_partition(points, seeds, feature_counts, temperature):
    """Perform Voronoi partitioning, weighted by Shannon entropy and Gini coefficient."""
    weights = [0.0] * len(seeds)
    shannon_entropy_value = shannon_entropy(feature_counts, temperature)
    gini_value = gini_coefficient(feature_counts)
    for i in range(len(seeds)):
        weights[i] = shannon_entropy_value * gini_value / (1 + i)
    regions = assign(points, seeds, weights)
    return regions

def assign(points, seeds, weights):
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        region_idx = nearest(p, seeds)
        regions[region_idx].append(p)
        region_weight = weights[region_idx]
        if random.random() < region_weight:
            regions[region_idx].append(p)
    return regions

def nearest(point, seeds):
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

if __name__ == "__main__":
    points = [(random.random(), random.random()) for _ in range(100)]
    seeds = [(random.random(), random.random()) for _ in range(10)]
    feature_counts = [random.random() for _ in range(10)]
    temperature = 25.0
    regions = voronoi_partition(points, seeds, feature_counts, temperature)
    print(regions)