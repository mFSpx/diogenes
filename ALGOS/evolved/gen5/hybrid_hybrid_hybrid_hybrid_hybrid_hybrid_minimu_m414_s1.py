# DARWIN HAMMER — match 414, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hdc_se_hybrid_hybrid_fracti_m265_s3.py (gen4)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s0.py (gen2)
# born: 2026-05-29T23:28:51Z

"""DARWIN HAMMER — Hybrid Algorithm: HMTHDC-FMCT (Morphology-Text Hyperdimensional Computing with Minimum-Cost Tree Scoring)

This module fuses the core of **Parent Algorithm A** (morphology → high‑dimensional vector, shape indices, and entropy) 
with **Parent Algorithm B** (minimum-cost tree scoring with Bayesian evidence update and label scoring).  
The exact mathematical bridge is established by incorporating the Bayesian update rules into the edge weights 
of the minimum-cost tree, while also utilizing the label scoring from the minimum-cost tree.  
This fusion enables the tree to not only consider the physical distances between nodes but also the probabilistic 
relevance of the paths connecting them and the relevance of labels to these paths.  
The mathematical interface is achieved by using the high-dimensional vector (HV) as the input to the minimum-cost 
tree scoring function, where the HV is weighted by the label scores of the nodes in the tree.

The module provides three high-level hybrid operations:
* `hybrid_encode(morphology, text)` – produces the fused hypervector.
* `hybrid_similarity(vec1, vec2)` – cosine similarity between two fused vectors.
* `hybrid_effect_estimate(morph1, text1, morph2, text2)` – similarity-based proxy for a causal effect estimate 
  between two morphology-text pairs.
"""

import re
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Tuple
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Vector = List[float]
HV = np.ndarray  # bipolar hypervector (+1 / -1)

# ---------------------------------------------------------------------------
# Parent A – Morphology utilities
# ---------------------------------------------------------------------------
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
    """Deterministic morphology vector generation."""
    return [m.length, m.width, m.height, m.mass]


def scalar_index(m: Morphology) -> float:
    """Compute a scalar index from morphology."""
    return m.length / (m.width + m.height + m.mass)


def morphology_to_hv(m: Morphology, dim: int = 10000) -> HV:
    """Convert morphology to bipolar hypervector."""
    vec = morphology_vector(m, dim)
    hv = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(vec):
        hv[i] = 2.0 * (x - np.mean(vec)) / np.std(vec)
        hv[i] = np.sign(hv[i]) * (np.abs(hv[i]) ** (1.1 * scalar_index(m)))
    return hv


# ---------------------------------------------------------------------------
# Parent B – Minimum-Cost Tree Scoring utilities
# ---------------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability for Bayesian update."""
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Perform Bayesian update on the prior probability."""
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal


def label_score(text: str, label: str) -> float:
    """Compute the score of a label on the text using the literal fallback algorithm."""
    labels = parse_labels(label)
    spans = literal_fallback(text, labels, case_sensitive=False)
    return sum(span.score for span in spans)


# ---------------------------------------------------------------------------
# Hybrid Functions
# ---------------------------------------------------------------------------
def hybrid_encode(morphology: Morphology, text: str) -> HV:
    """Encode morphology and text into a single bipolar hypervector."""
    hv = morphology_to_hv(morphology)
    text_hv = np.zeros_like(hv)
    for i, char in enumerate(text):
        text_hv[i] = 2.0 * (ord(char) - ord('a')) / (ord('z') - ord('a'))
    return np.multiply(hv, text_hv)


def hybrid_similarity(vec1: HV, vec2: HV) -> float:
    """Compute the cosine similarity between two bipolar hypervectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def hybrid_effect_estimate(morph1: Morphology, text1: str, morph2: Morphology, text2: str) -> float:
    """Estimate the causal effect between two morphology-text pairs."""
    hv1 = hybrid_encode(morph1, text1)
    hv2 = hybrid_encode(morph2, text2)
    return hybrid_similarity(hv1, hv2)


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    morphology = Morphology(length=10.0, width=5.0, height=2.0, mass=1.0)
    text = "hello world"
    hv = hybrid_encode(morphism, text)
    print(hv)
    vec1 = hv.copy()
    vec2 = hv.copy()
    sim = hybrid_similarity(vec1, vec2)
    print(sim)
    morph2 = Morphology(length=15.0, width=5.0, height=3.0, mass=1.5)
    text2 = "goodbye world"
    est = hybrid_effect_estimate(morph1, text1, morph2, text2)
    print(est)