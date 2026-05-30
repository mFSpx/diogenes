# DARWIN HAMMER — match 872, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s3.py (gen2)
# parent_b: hybrid_serpentina_self_righ_hybrid_hybrid_fisher_m29_s1.py (gen4)
# born: 2026-05-29T23:31:31Z

"""Hybrid algorithm combining geometric Bayesian edge costing (Parent A) with
morphology‑modulated Fisher information and SSIM similarity (Parent B).

Mathematical bridge:
- The *sphericity index* of a Morphology (Parent B) is used as a scaling factor
  for the Bayesian posterior in the edge‑cost computation of Parent A.
- The edge direction angle θ supplies a Gaussian‑beam intensity; its
  Fisher information (modulated by sphericity) is added as a multiplicative
  penalty to the base edge cost.
- Finally, a morphology‑aware SSIM between two label‑score vectors refines the
  aggregated cost, tying the two parent topologies into a single unified
  system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Set, Iterable, Callable

import numpy as np

# ----------------------------------------------------------------------
# Types and basic geometric utilities (from Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

def euclidean(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Bayesian utilities (from Parent A)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = P(E|H)P(H) + P(E|¬H)P(¬H)."""
    if not 0.0 <= prior <= 1.0 or not 0.0 <= likelihood <= 1.0 or not 0.0 <= false_positive <= 1.0:
        raise ValueError("All probability arguments must lie in [0, 1].")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P(E|H)P(H) / P(E)."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0.")
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# Simple label scoring (from Parent A)
# ----------------------------------------------------------------------
def simple_label_score(text: str, label: str) -> float:
    """Count case‑insensitive occurrences of *label* in *text*, normalised."""
    if not text:
        return 0.0
    count = text.lower().count(label.lower())
    return count / max(1, len(text.split()))

def aggregate_label_scores(text: str, label_dict: Dict[str, float]) -> float:
    """Weighted mean of simple label scores."""
    if not label_dict:
        return 0.0
    scores = [simple_label_score(text, lbl) * w for lbl, w in label_dict.items()]
    return float(np.mean(scores))

# ----------------------------------------------------------------------
# Morphology utilities (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    """Dimensionless sphericity."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    """Dimensionless flatness."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def gaussian_beam(theta: float, center: float, width: float, sphericity: float) -> float:
    """Modulated Gaussian beam intensity."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return sphericity * math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, sphericity: float,
                 eps: float = 1e-12) -> float:
    """Modulated Fisher information for the Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width, sphericity), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03,
         morphology: Morphology = None) -> float:
    """Structural Similarity Index, optionally modulated by morphology."""
    if x.shape != y.shape:
        raise ValueError("Input images must have the same dimensions")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    # Morphology can scale the constants, giving more weight to shape similarity.
    if morphology:
        scale = sphericity_index(morphology.length, morphology.width, morphology.height)
        C1 *= scale
        C2 *= scale

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def infer_morphology(nodes: Dict[str, Point],
                     masses: Dict[str, float]) -> Morphology:
    """
    Derive a Morphology instance from a set of node coordinates and per‑node masses.
    Length, width, height are taken as the extents of the point cloud.
    """
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    length = max(xs) - min(xs) if xs else 1.0
    width = max(ys) - min(ys) if ys else 1.0
    # Height is approximated from the total mass (mass ∝ volume ∝ height)
    total_mass = sum(masses.get(k, 1.0) for k in nodes)
    height = max(0.1, total_mass ** (1.0 / 3.0))  # ensure positivity
    return Morphology(length=length, width=width, height=height, mass=total_mass)


def hybrid_edge_cost(a: str,
                     b: str,
                     nodes: Dict[str, Point],
                     priors: Dict[Edge, float],
                     likelihoods: Dict[Edge, float],
                     false_positive: float,
                     label_texts: Dict[Edge, str],
                     label_weights: Dict[Edge, Dict[str, float]],
                     morphology: Morphology) -> float:
    """
    Unified edge cost:
    1. Base Euclidean distance.
    2. Bayesian posterior, scaled by the sphericity index.
    3. Fisher‑information penalty derived from the edge direction.
    4. SSIM similarity between label‑score vectors, modulated by flatness.
    """
    # ----- 1. geometric distance -------------------------------------------------
    dist = euclidean(nodes[a], nodes[b])

    # ----- 2. Bayesian posterior (Parent A) ------------------------------------
    prior = priors.get((a, b), 0.5)
    likelihood = likelihoods.get((a, b), 0.5)
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)

    # Scale posterior by sphericity (bridge)
    sph = sphericity_index(morphology.length, morphology.width, morphology.height)
    posterior_scaled = posterior * sph

    # ----- 3. Fisher information (Parent B) ------------------------------------
    # Edge direction angle θ ∈ [‑π, π]
    dx = nodes[b][0] - nodes[a][0]
    dy = nodes[b][1] - nodes[a][1]
    theta = math.atan2(dy, dx)

    # Use a fixed beam centre (0) and a width proportional to morphology size
    beam_width = max(0.1, (morphology.length + morphology.width) / 4.0)
    fisher = fisher_score(theta, center=0.0, width=beam_width, sphericity=sph)

    # ----- 4. Label‑score SSIM (Parent B) ---------------------------------------
    txt_a = label_texts.get((a, b), "")
    txt_b = label_texts.get((b, a), "")  # reverse direction may have different text
    scores_a = np.array([simple_label_score(txt_a, lbl) * w
                         for lbl, w in label_weights.get((a, b), {}).items()], dtype=float)
    scores_b = np.array([simple_label_score(txt_b, lbl) * w
                         for lbl, w in label_weights.get((b, a), {}).items()], dtype=float)

    # Pad to equal length for SSIM
    max_len = max(len(scores_a), len(scores_b))
    if max_len == 0:
        ssim_val = 1.0  # no labels -> perfect similarity
    else:
        scores_a = np.pad(scores_a, (0, max_len - len(scores_a)), constant_values=0)
        scores_b = np.pad(scores_b, (0, max_len - len(scores_b)), constant_values=0)
        ssim_val = ssim(scores_a, scores_b,
                        morphology=morphology)

    # ----- 5. Combine all components --------------------------------------------
    # Distance is the baseline; the rest are dimensionless penalties/bonuses.
    combined = dist * (1.0 + posterior_scaled + fisher) * (2.0 - ssim_val)
    return combined


def hybrid_total_cost(nodes: Dict[str, Point],
                      edges: Set[Edge],
                      priors: Dict[Edge, float],
                      likelihoods: Dict[Edge, float],
                      false_positive: float,
                      label_texts: Dict[Edge, str],
                      label_weights: Dict[Edge, Dict[str, float]],
                      masses: Dict[str, float]) -> float:
    """
    Sum of hybrid_edge_cost over all edges. Internally derives Morphology.
    """
    morphology = infer_morphology(nodes, masses)
    total = 0.0
    for a, b in edges:
        total += hybrid_edge_cost(a, b, nodes, priors, likelihoods,
                                  false_positive, label_texts, label_weights,
                                  morphology)
    return total


def random_graph(num_nodes: int = 5,
                 edge_prob: float = 0.4) -> Tuple[Dict[str, Point],
                                                Set[Edge],
                                                Dict[Edge, float],
                                                Dict[Edge, float],
                                                Dict[Edge, str],
                                                Dict[Edge, Dict[str, float]],
                                                Dict[str, float]]:
    """
    Utility to generate a small random graph with synthetic data for testing.
    Returns:
        nodes, edges, priors, likelihoods, label_texts, label_weights, masses
    """
    nodes = {f"N{i}": (random.uniform(0, 10), random.uniform(0, 10))
             for i in range(num_nodes)}
    masses = {k: random.uniform(0.5, 5.0) for k in nodes}
    edges: Set[Edge] = set()
    priors: Dict[Edge, float] = {}
    likelihoods: Dict[Edge, float] = {}
    label_texts: Dict[Edge, str] = {}
    label_weights: Dict[Edge, Dict[str, float]] = {}

    possible = [(a, b) for a in nodes for b in nodes if a != b]
    for a, b in possible:
        if random.random() < edge_prob:
            edges.add((a, b))
            priors[(a, b)] = random.random()
            likelihoods[(a, b)] = random.random()
            # synthetic text containing possible labels
            label_texts[(a, b)] = "alpha beta gamma"
            label_weights[(a, b)] = {"alpha": random.random(),
                                     "beta": random.random(),
                                     "gamma": random.random()}
    return (nodes, edges, priors, likelihoods, label_texts,
            label_weights, masses)


if __name__ == "__main__":
    # Smoke test: generate a random graph and compute the hybrid total cost.
    (nodes, edges, priors, likelihoods,
     label_texts, label_weights, masses) = random_graph(num_nodes=6, edge_prob=0.5)

    try:
        total = hybrid_total_cost(nodes, edges, priors, likelihoods,
                                  false_positive=0.1,
                                  label_texts=label_texts,
                                  label_weights=label_weights,
                                  masses=masses)
        print(f"Hybrid total cost for generated graph: {total:.4f}")
    except Exception as e:
        print(f"Error during hybrid computation: {e}", file=sys.stderr)
        sys.exit(1)