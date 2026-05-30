# DARWIN HAMMER — match 3131, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_voronoi_parti_m53_s2.py (gen4)
# parent_b: hybrid_pheromone_hybrid_hybrid_hybrid_m1143_s2.py (gen5)
# born: 2026-05-29T23:48:07Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import numpy as np

# Shared constants
ALPHA = 0.6          # store inflow coefficient
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step for store dynamics
ETA0 = 0.01          # base learning rate for matrix updates
DELTA_MAX = 1.0      # max evasion magnitude
ALPHA_EVASION = 3.0  # decay rate for evasion schedule
HOEFFDING_DELTA = 0.

# Feature extraction and weighting
_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)
_TOTAL_ABS_WEIGHTS = np.abs(_POSITIVE_WEIGHTS) + np.abs(_NEGATIVE_WEIGHTS)

# Constants & Prototype
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)

def compute_ssim(
    x: np.ndarray,
    y: np.ndarray,
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index (SSIM) between two equal‑length vectors."""
    if len(x) != len(y):
        raise ValueError("Input vectors must have the same length")
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    k1_squared = (k1 * dynamic_range)**2
    k2_squared = (k2 * dynamic_range)**2
    numerator = (2 * mu_x * mu_y + k1_squared) * (2 * sigma_xy + k2_squared)
    denominator = (mu_x**2 + mu_y**2 + k1_squared) * (sigma_x**2 + sigma_y**2 + k2_squared)
    return numerator / denominator

def compute_voronoi_likelihood(endpoint: np.ndarray, point: np.ndarray) -> float:
    """Geometric-morphological health-distance score for assigning points to endpoints."""
    distance = np.linalg.norm(endpoint - point)
    return 1 / (1 + distance**2)

def update_surface_pheromone_signals(
    pheromone_signals: np.ndarray,
    hybrid_score: float,
    learning_rate: float = ETA0,
) -> np.ndarray:
    """Update surface pheromone signals with the hybrid score as a weighting factor."""
    return pheromone_signals * (1 + learning_rate * hybrid_score)

def update_endpoint_assignments(
    endpoints: np.ndarray,
    points: np.ndarray,
    voronoi_likelihoods: np.ndarray,
    hybrid_scores: np.ndarray,
) -> np.ndarray:
    """Update endpoint assignments based on the Voronoi-health-distance likelihood and hybrid scores."""
    assignments = np.zeros((len(points), len(endpoints)))
    for i, point in enumerate(points):
        for j, endpoint in enumerate(endpoints):
            distance = np.linalg.norm(endpoint - point)
            voronoi_likelihood = voronoi_likelihoods[j]
            hybrid_score = hybrid_scores[j]
            assignments[i, j] = voronoi_likelihood * hybrid_score
    return assignments / np.sum(assignments, axis=1, keepdims=True)

def compute_hybrid_score(
    packet: np.ndarray,
    endpoint: np.ndarray,
    point: np.ndarray,
    prototype_vector: np.ndarray,
    weight: float,
) -> float:
    """Compute the hybrid score by multiplying the SSIM-derived likelihood with the Voronoi-health-distance likelihood."""
    ssim_likelihood = compute_ssim(packet, prototype_vector)
    voronoi_likelihood = compute_voronoi_likelihood(endpoint, point)
    return ssim_likelihood**weight * voronoi_likelihood**(1 - weight)

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to have a length of 1."""
    return vector / np.linalg.norm(vector)

def compute_weighted_ssim(
    packet: np.ndarray,
    prototype_vector: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute the weighted SSIM between two vectors."""
    weighted_packet = packet * weights
    weighted_prototype = prototype_vector * weights
    return compute_ssim(weighted_packet, weighted_prototype)

if __name__ == "__main__":
    # Smoke test
    packet = np.array([0.2, 0.5, 0.3, 0.7, 0.1])
    endpoint = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    point = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    pheromone_signals = np.array([0.5, 0.5])
    hybrid_score = compute_hybrid_score(packet, endpoint, point, PROTOTYPE_VECTOR, 0.5)
    weighted_ssim = compute_weighted_ssim(packet, PROTOTYPE_VECTOR, _TOTAL_ABS_WEIGHTS)
    updated_pheromone_signals = update_surface_pheromone_signals(pheromone_signals, hybrid_score)
    print(updated_pheromone_signals)