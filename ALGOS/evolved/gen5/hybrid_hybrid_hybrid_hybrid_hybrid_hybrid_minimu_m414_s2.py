# DARWIN HAMMER — match 414, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hdc_se_hybrid_hybrid_fracti_m265_s3.py (gen4)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s0.py (gen2)
# born: 2026-05-29T23:28:51Z

"""
Hybrid Algorithm: Fusing HMTHDC (Hybrid Morphology-Text Hyperdimensional Computing) 
with Minimum-Cost Tree Scoring and Bayesian Update

This module fuses the core of **Parent Algorithm A** (HMTHDC) with **Parent Algorithm B** 
(minimum-cost tree scoring and Bayesian update). The mathematical bridge is established 
by using the morphology-derived scalar index as the prior probability in the Bayesian 
update function, which in turn modifies the edge weights in the minimum-cost tree scoring.

The hybrid system integrates the hyperdimensional encoding of morphology and text with 
the probabilistic relevance of paths in a minimum-cost tree. This enables the joint 
consideration of physical shape information, textual evidence, and probabilistic 
relevance in a single unified representation.

The module provides three high-level hybrid operations:
* `hybrid_encode(morphology, text)` – produces the fused hypervector.
* `hybrid_tree_score(nodes, edges, labels, morphology, text)` – computes the score of 
  a minimum-cost tree with Bayesian update and label scoring.
* `hybrid_effect_estimate(morph1, text1, morph2, text2)` – similarity-based proxy for 
  a causal effect estimate between two morphology-text pairs.
"""

import numpy as np
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

Vector = List[float]
HV = np.ndarray  # bipolar hypervector (+1 / -1)

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]

def morphology_vector(m: Morphology, dim: int = 10000) -> Vector:
    return [m.length, m.width, m.height, m.mass] + [0.0] * (dim - 4)

def min_hash(text: str, dim: int = 10000) -> HV:
    hash_values = []
    for i in range(dim):
        hash_value = int(hashlib.md5(f"{text}{i}".encode()).hexdigest(), 16)
        hash_values.append(1 if hash_value % 2 == 0 else -1)
    return np.array(hash_values, dtype=np.int8)

def fractional_power_binding(morphology_vector: Vector, alpha: float) -> HV:
    hv = np.array([1 if x > 0.5 else -1 for x in morphology_vector])
    return np.power(hv, alpha)

def hybrid_encode(morphology: Morphology, text: str) -> HV:
    morphology_vec = morphology_vector(morphology)
    text_hv = min_hash(text)
    alpha = 0.5 + 0.1 * morphology.length
    weighted_hv = fractional_power_binding(morphology_vec, alpha)
    return np.multiply(weighted_hv, text_hv)

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def label_score(text: str, label: str) -> float:
    if text == label:
        return 1.0
    else:
        return 0.0

def hybrid_tree_score(nodes: List[Tuple[float, float]], edges: List[Tuple[str, str]], 
                      labels: List[str], morphology: Morphology, text: str) -> float:
    prior = 0.5 + 0.1 * morphology.length
    tree_cost = 0.0
    for edge in edges:
        node1, node2 = edge
        distance = math.hypot(nodes[int(node1)][0] - nodes[int(node2)][0], 
                              nodes[int(node1)][1] - nodes[int(node2)][1])
        likelihood = math.exp(-distance)
        marginal = bayes_marginal(prior, likelihood, 0.1)
        tree_cost += bayes_update(prior, likelihood, marginal) * label_score(text, labels[int(node1)])
    return tree_cost

def hybrid_effect_estimate(morph1: Morphology, text1: str, morph2: Morphology, text2: str) -> float:
    hv1 = hybrid_encode(morph1, text1)
    hv2 = hybrid_encode(morph2, text2)
    return np.dot(hv1, hv2) / (np.linalg.norm(hv1) * np.linalg.norm(hv2))

if __name__ == "__main__":
    morph = Morphology(10.0, 5.0, 2.0, 1.0)
    text = "example text"
    hv = hybrid_encode(morph, text)
    print(hv)

    nodes = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    edges = [("0", "1"), ("1", "2")]
    labels = ["label1", "label2", "label3"]
    tree_score = hybrid_tree_score(nodes, edges, labels, morph, text)
    print(tree_score)

    morph2 = Morphology(8.0, 4.0, 1.5, 0.8)
    text2 = "another example"
    effect_estimate = hybrid_effect_estimate(morph, text, morph2, text2)
    print(effect_estimate)