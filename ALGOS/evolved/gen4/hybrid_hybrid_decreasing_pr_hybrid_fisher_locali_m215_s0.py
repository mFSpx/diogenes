# DARWIN HAMMER — match 215, survivor 0
# gen: 4
# parent_a: hybrid_decreasing_pruning_hybrid_hybrid_minimu_m63_s1.py (gen3)
# parent_b: hybrid_fisher_localization_hybrid_ternary_route_m40_s3.py (gen2)
# born: 2026-05-29T23:27:33Z

"""
This module represents a hybrid algorithm, combining the principles of decreasing-rate pruning 
from decreasing_pruning.py and hybrid minimum-cost tree scoring with epistemic certainty computation 
from hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s1.py, along with fisher localization 
from fisher_localization.py and ternary route selection from hybrid_ternary_router_ssim_m1_s0.py.
The mathematical bridge between these two systems is established by incorporating the epistemic 
certainty flags into the edge weights of the minimum-cost tree, allowing the tree to adapt and 
re-weight its edges based on both physical distances and epistemic certainty, and then applying 
a decreasing-rate pruning schedule to the resulting tree. Additionally, we use the fisher score 
to determine the optimal angle for localization and the structural similarity index to evaluate 
the quality of the ternary route.
"""

import math
import numpy as np
import random
import sys
import pathlib
from collections.abc import Hashable

EPISTEMIC_FLAGS: tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non-negative')
    return min(1.0, lam * math.exp(-alpha * t))

def prune_edges(edges: list[Hashable], t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> list[Hashable]:
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]

def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

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

def certainty(label: str, *, confidence_bps: int, authority_class: str, rationale: str, evidence_refs: tuple[str, ...] = ()) -> dict:
    """Create an epistemic certainty flag."""
    if label not in EPISTEMIC_FLAGS:
        raise ValueError(f"Invalid label: {label}")
    return {
        "label": label,
        "confidence": confidence_bps,
        "authority_class": authority_class,
        "rationale": rationale,
        "evidence": evidence_refs
    }

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ.

    F(θ) = (∂I/∂θ)² / I  where I = Gaussian beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: Sequence[float], y: Sequence[float],
         dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index between two 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")
    n = len(x)
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))

def hybrid_metric(theta: float, center: float, width: float,
                  packet_text: str, reference_text: str) -> float:
    """Combined quality metric H = Fisher(θ) × SSIM(text, reference)."""
    f = fisher_score(theta, center, width)
    s = ssim(text_to_signal(packet_text), text_to_signal(reference_text))
    return f * s

def text_to_signal(text: str) -> np.ndarray:
    """Convert a Unicode string to a numeric signal (code‑point float array)."""
    return np.array([float(ord(ch)) for ch in text])

def best_hybrid_angle(candidates: np.ndarray, center: float, width: float,
                      packet_text: str, reference_text: str) -> float:
    """Select the angle that maximises the hybrid metric.

    Tie‑breaker: choose the angle closest to the centre when metrics are equal.
    """
    if len(candidates) == 0:
        raise ValueError("candidates must not be empty")
    return candidates[np.argmax(hybrid_metric(candidates, center, width, packet_text, reference_text))]

def hybrid_tree_pruning(edges: list[Hashable], t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> list[Hashable]:
    """Hybrid tree pruning with epistemic certainty and fisher localization."""
    certainties = [certainty(edge, seed=random.randint(0, 100)) for edge in edges]
    pruned_edges = prune_edges(edges, t, lam, alpha, seed)
    for c in certainties:
        if c["label"] == "FACT":
            pruned_edges.remove(c["edge"])
    return pruned_edges

def fisher_localized_edges(edges: list[Hashable], center: float, width: float, packet_text: str, reference_text: str) -> list[Hashable]:
    """Fisher localized edges with ternary route selection."""
    candidates = [edge for edge in edges]
    best_angles = [best_hybrid_angle(candidates, center, width, packet_text, reference_text) for edge in edges]
    best_edges = [edge for edge, angle in zip(edges, best_angles) if abs(angle - edge[1]) < 1e-6]
    return best_edges

def hybrid_edge_selection(edges: list[Hashable], t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> list[Hashable]:
    """Hybrid edge selection with epistemic certainty and fisher localization."""
    pruned_edges = hybrid_tree_pruning(edges, t, lam, alpha, seed)
    best_edges = fisher_localized_edges(pruned_edges, 0.0, 1.0, "packet", "reference")
    return best_edges

if __name__ == "__main__":
    edges = [(0, 0), (1, 1), (2, 2), (3, 3)]
    t = 10.0
    lam = 1.0
    alpha = 0.2
    seed = 42
    pruned_edges = hybrid_tree_pruning(edges, t, lam, alpha, seed)
    best_edges = fisher_localized_edges(pruned_edges, 0.0, 1.0, "packet", "reference")
    print(best_edges)