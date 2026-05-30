# DARWIN HAMMER — match 1463, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_krampus_brain_hybrid_indy_learning_m273_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m940_s0.py (gen5)
# born: 2026-05-29T23:36:31Z

"""
Hybrid Algorithm: Krampus-Brainmap / Indy-Learning Vector / Hybrid Bayesian-RBF-Perceptual Model

This module fuses the following parent algorithms:
- hybrid_hybrid_krampus_brain_hybrid_indy_learning_m273_s2.py: provides a high-dimensional vector representation 
  (the “brain map”) and an infotaxis decision process that uses entropy and information-gain on pheromone signals.
- hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m940_s0.py: integrates Bayesian-RBF-Perceptual Model, 
  prototype-based hybrid_score, perceptual hashing, and radial-basis-function surrogate.

Mathematical bridge:
The bridge is the shared vector space. The Indy-learning pipeline yields a term-frequency vector **v** ∈ ℝⁿ 
(n = number of ontology terms). The pheromone store holds a signal value s_i for each term i, forming a discrete 
probability distribution p_i = s_i / Σ_j s_j. Entropy H(p) = – Σ_i p_i log p_i quantifies the uncertainty of 
the pheromone field. The perceptual hash of a payload is turned into a binary feature vector. This vector is 
fed to an RBF surrogate (Gaussian kernel). The raw RBF output is modulated by the SSIM between the payload and 
a fixed prototype vector. The final fused score is the product of the SSIM, RBF output, and infotaxis update.
"""

import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Constants and utilities
# ----------------------------------------------------------------------
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)

def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index (SSIM) for 1-D signals."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)

def build_term_vector(text: str, ontology_terms: List[str]) -> np.ndarray:
    """Tokenises text and builds the ontology-based vector."""
    tokens = re.findall(r"\S+", text)
    term_freq = np.zeros(len(ontology_terms))
    for token in tokens:
        if token in ontology_terms:
            term_freq[ontology_terms.index(token)] += 1
    return term_freq

def entropy(phermone_distribution: np.ndarray) -> float:
    """Computes Shannon entropy of the current pheromone distribution."""
    phermone_distribution = phermone_distribution / phermone_distribution.sum()
    return -np.sum(phermone_distribution * np.log(phermone_distribution))

def infotaxis_update(phermone_distribution: np.ndarray, term_freq: np.ndarray, alpha: float) -> float:
    """Injects the vector into the pheromone store and returns the information gain."""
    new_phermone_distribution = phermone_distribution + alpha * term_freq
    new_entropy = entropy(new_phermone_distribution)
    return entropy(phermone_distribution) - new_entropy

def hybrid_score(packet: dict, phermone_distribution: np.ndarray, alpha: float) -> float:
    """Prototype-distance similarity used in Parent A, modulated by the SSIM and infotaxis update."""
    payload = packet.get("payload")
    if not isinstance(payload, (list, tuple)):
        return 0.0
    payload_vec = np.asarray(payload, dtype=np.float64)
    ssim = compute_ssim(payload_vec, PROTOTYPE_VECTOR)
    term_freq = build_term_vector(" ".join(str(x) for x in payload_vec), ["ENTITY", "ATTRIBUTE", "RELATIONSHIP", "ACTION", "EVENT"])
    infotaxis_update_val = infotaxis_update(phermone_distribution, term_freq, alpha)
    return ssim * np.exp(-np.linalg.norm(payload_vec - PROTOTYPE_VECTOR)) * infotaxis_update_val

if __name__ == "__main__":
    phermone_distribution = np.array([0.2, 0.5, 0.3])
    packet = {"payload": [0.1, 0.2, 0.3, 0.4, 0.5]}
    alpha = 0.1
    print(hybrid_score(packet, phermone_distribution, alpha))