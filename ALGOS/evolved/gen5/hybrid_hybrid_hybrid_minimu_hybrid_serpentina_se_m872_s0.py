# DARWIN HAMMER — match 872, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s3.py (gen2)
# parent_b: hybrid_serpentina_self_righ_hybrid_hybrid_fisher_m29_s1.py (gen4)
# born: 2026-05-29T23:31:31Z

"""
Hybrid Algorithm: Fusing hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s3.py 
                  and hybrid_serpentina_self_righ_hybrid_hybrid_fisher_m29_s1.py.

This hybrid algorithm combines the Bayesian and geometric utilities from 
hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s3.py with 
the morphological analysis and Fisher information from 
hybrid_serpentina_self_righ_hybrid_hybrid_fisher_m29_s1.py. 
The mathematical bridge is established by relating the 
morphological indices (sphericity and flatness) to the 
Bayesian prior probabilities.

The hybrid system uses the morphological indices to modulate 
the prior probabilities, which in turn affect the 
Bayesian updates and edge cost computations.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Set

# Basic geometric utilities ----------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

def euclidean(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# Bayesian utilities -----------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """
    Compute the marginal probability P(E) = P(E|H)P(H) + P(E|¬H)P(¬H).

    ``false_positive`` is interpreted as P(E|¬H).
    """
    if not 0.0 <= prior <= 1.0 or not 0.0 <= likelihood <= 1.0 or not 0.0 <= false_positive <= 1.0:
        raise ValueError("All probability arguments must lie in [0, 1].")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """
    Return the posterior P(H|E) = P(E|H)P(H) / P(E).
    """
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0.")
    return prior * likelihood / marginal

# Morphological utilities --------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

# Hybrid utilities -------------------------------------------------------------
def modulate_prior(morphology: Morphology) -> float:
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    flatness = flatness_index(morphology.length, morphology.width, morphology.height)
    prior = (sphericity + flatness) / 2.0
    return max(0.0, min(prior, 1.0))  # clamp prior to [0, 1]

def hybrid_edge_cost(
    a: str,
    b: str,
    nodes: Dict[str, Point],
    morphology: Morphology,
    likelihoods: Dict[Edge, float],
    false_positive: float
) -> float:
    prior = modulate_prior(morphology)
    marginal = bayes_marginal(prior, likelihoods.get((a, b), 0.0), false_positive)
    return bayes_update(prior, likelihoods.get((a, b), 0.0), marginal) * euclidean(nodes[a], nodes[b])

def hybrid_ssim(x: np.ndarray, y: np.ndarray, morphology: Morphology) -> float:
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    dynamic_range = 255.0
    k1 = 0.01
    k2 = 0.03
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    luminance = (2 * mu_x * mu_y + k1 * dynamic_range ** 2) / (mu_x ** 2 + mu_y ** 2 + k1 * dynamic_range ** 2)
    contrast = (2 * sigma_x * sigma_y + k2 * dynamic_range ** 2) / (sigma_x ** 2 + sigma_y ** 2 + k2 * dynamic_range ** 2)
    structural = (sigma_xy + k2 * dynamic_range ** 2) / (sigma_x * sigma_y + k2 * dynamic_range ** 2)
    return sphericity * luminance * contrast * structural

if __name__ == "__main__":
    morphology = Morphology(10.0, 5.0, 3.0, 1.0)
    nodes = {"A": (0.0, 0.0), "B": (3.0, 4.0)}
    likelihoods = {("A", "B"): 0.8}
    false_positive = 0.2
    print(hybrid_edge_cost("A", "B", nodes, morphology, likelihoods, false_positive))
    x = np.random.rand(10, 10)
    y = np.random.rand(10, 10)
    print(hybrid_ssim(x, y, morphology))