# DARWIN HAMMER — match 872, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s3.py (gen2)
# parent_b: hybrid_serpentina_self_righ_hybrid_hybrid_fisher_m29_s1.py (gen4)
# born: 2026-05-29T23:31:31Z

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Set, Iterable, Callable

Point = Tuple[float, float]
Edge = Tuple[str, str]

def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def gaussian_beam(theta: float, center: float, width: float, sphericity: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return sphericity * math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, sphericity: float,
                 eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width, sphericity), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03,
         morphology: Morphology = None) -> float:
    if x.shape != y.shape:
        raise ValueError("Input images must have the same dimensions")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

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

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not 0.0 <= prior <= 1.0 or not 0.0 <= likelihood <= 1.0 or not 0.0 <= false_positive <= 1.0:
        raise ValueError("All probability arguments must lie in [0, 1].")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0.")
    return prior * likelihood / marginal

def simple_label_score(text: str, label: str) -> float:
    if not text:
        return 0.0
    count = text.lower().count(label.lower())
    return count / max(1, len(text.split()))

def aggregate_label_scores(text: str, label_dict: Dict[str, float]) -> float:
    if not label_dict:
        return 0.0
    scores = [simple_label_score(text, lbl) * w for lbl, w in label_dict.items()]
    return float(np.mean(scores))

def infer_morphology(nodes: Dict[str, Point],
                     masses: Dict[str, float]) -> Morphology:
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    length = max(xs) - min(xs) if xs else 1.0
    width = max(ys) - min(ys) if ys else 1.0
    total_mass = sum(masses.get(k, 1.0) for k in nodes)
    height = max(0.1, total_mass ** (1.0 / 3.0))  
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
    dist = euclidean(nodes[a], nodes[b])

    prior = priors.get((a, b), 0.5)
    likelihood = likelihoods.get((a, b), 0.5)
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)

    sph = sphericity_index(morphology.length, morphology.width, morphology.height)
    posterior_scaled = posterior * sph

    dx = nodes[b][0] - nodes[a][0]
    dy = nodes[b][1] - nodes[a][1]
    theta = math.atan2(dy, dx)

    beam_width = max(0.1, (morphology.length + morphology.width) / 4.0)
    fisher = fisher_score(theta, center=0.0, width=beam_width, sphericity=sph)

    txt_a = label_texts.get((a, b), "")
    txt_b = label_texts.get((b, a), "")

    scores_a = aggregate_label_scores(txt_a, label_weights.get((a, b), {}))
    scores_b = aggregate_label_scores(txt_b, label_weights.get((b, a), {}))

    label_vector_a = np.array([scores_a])
    label_vector_b = np.array([scores_b])

    ssim_similarity = ssim(label_vector_a, label_vector_b, morphology=morphology)

    flatness = flatness_index(morphology.length, morphology.width, morphology.height)

    unified_cost = dist + posterior_scaled + fisher + (1 - ssim_similarity) * flatness

    return unified_cost