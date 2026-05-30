# DARWIN HAMMER — match 2754, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2146_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hoeffd_m881_s0.py (gen5)
# born: 2026-05-29T23:45:36Z

"""Hybrid Algorithm integrating Bayesian edge updates, Gaussian distance kernels,
Fisher information, and Hoeffding confidence bounds.

Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2146_s1.py) provides:
- Euclidean distances between points.
- Bayesian marginalisation and update for discrete probability distributions.

Parent B (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hoeffd_m881_s0.py) provides:
- Gaussian beam intensity as a similarity kernel.
- Fisher information derived from the Gaussian beam.
- Hoeffding bound to quantify confidence given a finite number of observations.

Mathematical bridge:
The Euclidean distance `d` between two nodes is turned into a Gaussian similarity
`L = exp(-0.5 * (d/σ)^2)`.  This similarity acts as the *likelihood* in a Bayesian
update of a prior edge probability `π`.  The same distance defines an angle
`θ` (the polar angle of the vector) which is fed to the Gaussian beam model to
obtain an intensity `I(θ)`.  The Fisher information `F(θ)` computed from `I(θ)`
is used as an *evidence strength* factor that scales the likelihood before the
Bayesian update.  After obtaining the posterior edge probability `p̂`, a
Hoeffding bound `ε` is evaluated on `p̂` (treated as a bounded random variable in
`[0,1]`) to produce a confidence‑adjusted score `s = p̂ - ε`.  This score can be
interpreted as a minimum‑cost edge weight that respects both probabilistic
evidence and finite‑sample uncertainty.

The module implements three core hybrid functions that embody this bridge and
exposes a small smoke‑test when run as a script.
"""

import math
import random
import sys
import pathlib
from typing import Tuple, Dict, List

import numpy as np

Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Geometry utilities (from Parent A)
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Gaussian similarity kernel (from Parent B)
# ----------------------------------------------------------------------
def gaussian_similarity(distance: float, sigma: float) -> float:
    """
    Gaussian kernel converting a distance into a similarity in (0,1].

    Parameters
    ----------
    distance: float
        Euclidean distance between two nodes.
    sigma: float
        Scale parameter of the kernel (must be > 0).

    Returns
    -------
    similarity: float
        exp(-0.5 * (distance / sigma)^2)
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    z = distance / sigma
    return math.exp(-0.5 * z * z)

# ----------------------------------------------------------------------
# Fisher information derived from a Gaussian beam (Parent B)
# ----------------------------------------------------------------------
def fisher_information(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam evaluated at angle theta.

    The beam intensity is I(θ) = exp(-0.5 * ((θ-center)/width)^2).
    Fisher information F = (∂I/∂θ)^2 / I.

    Parameters
    ----------
    theta: float
        Observation angle (radians).
    center: float
        Beam centre (radians).
    width: float
        Beam width (positive).
    eps: float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Fisher information (non‑negative).
    """
    if width <= 0.0:
        raise ValueError("width must be positive")
    intensity = max(math.exp(-0.5 * ((theta - center) / width) ** 2), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Bayesian primitives (from Parent A)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = likelihood * prior + false_positive * (1 - prior)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior = prior * likelihood / marginal."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# Hoeffding bound (from Parent B)
# ----------------------------------------------------------------------
def hoeffding_bound(range_: float, delta: float, n: int) -> float:
    """
    Hoeffding confidence radius ε = sqrt( (R² * ln(1/δ)) / (2n) ).

    Parameters
    ----------
    range_: float
        Known range R of the bounded random variable (must be > 0).
    delta: float
        Desired error probability (0 < delta < 1).
    n: int
        Number of independent observations (must be > 0).

    Returns
    -------
    float
        Confidence radius ε.
    """
    if range_ <= 0.0 or not (0.0 < delta < 1.0) or n <= 0:
        raise ValueError("Invalid arguments for Hoeffding bound")
    return math.sqrt((range_ ** 2 * math.log(1.0 / delta)) / (2.0 * n))

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def edge_likelihood(prior_prob: float,
                    distance: float,
                    sigma: float,
                    theta: float,
                    beam_center: float,
                    beam_width: float,
                    false_positive: float) -> float:
    """
    Compute the Bayesian posterior probability for an edge given geometric
    and information‑theoretic evidence.

    Steps
    -----
    1. Convert Euclidean distance to a Gaussian similarity `L`.
    2. Compute Fisher information `F` from the angular orientation `theta`.
    3. Modulate the similarity by a monotonic function of `F` (here we use
       `likelihood = L * (1 + log(1 + F))` to keep the result in (0,1]).
    4. Perform Bayesian marginalisation and update.

    Returns
    -------
    posterior: float
        Updated edge probability after incorporating both sources of evidence.
    """
    # 1. Gaussian similarity as base likelihood
    base_likelihood = gaussian_similarity(distance, sigma)

    # 2. Fisher information from the angular component
    fisher = fisher_information(theta, beam_center, beam_width)

    # 3. Combine – the log‑scale ensures the product stays bounded
    combined_likelihood = base_likelihood * (1.0 + math.log1p(fisher))

    # Clamp to [0,1] for Bayesian semantics
    combined_likelihood = max(0.0, min(1.0, combined_likelihood))

    # 4. Bayesian update
    marginal = bayes_marginal(prior_prob, combined_likelihood, false_positive)
    posterior = bayes_update(prior_prob, combined_likelihood, marginal)
    return posterior

def edge_confidence_score(prior_prob: float,
                          p1: Point,
                          p2: Point,
                          sigma: float = 1.0,
                          beam_center: float = 0.0,
                          beam_width: float = math.pi / 4,
                          false_positive: float = 0.01,
                          delta: float = 0.05,
                          n_observations: int = 30) -> float:
    """
    Produce a confidence‑adjusted edge score.

    The score is `posterior - ε` where `ε` is the Hoeffding bound evaluated on
    the posterior treated as a bounded variable in [0,1].

    Parameters
    ----------
    prior_prob: float
        Prior belief about the existence of the edge (in [0,1]).
    p1, p2: Point
        Endpoints of the edge.
    sigma, beam_center, beam_width, false_positive, delta, n_observations:
        Hyper‑parameters controlling the hybrid behaviour.

    Returns
    -------
    float
        Confidence‑adjusted score (may be negative if uncertainty dominates).
    """
    # Geometric descriptors
    dist = euclidean_distance(p1, p2)

    # Angle of the vector (used for Fisher information)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    theta = math.atan2(dy, dx)

    # Bayesian posterior using both distance and Fisher info
    posterior = edge_likelihood(prior_prob, dist, sigma,
                                theta, beam_center, beam_width,
                                false_positive)

    # Hoeffding confidence radius on the posterior (range = 1)
    epsilon = hoeffding_bound(range_=1.0, delta=delta, n=n_observations)

    return posterior - epsilon

def build_hybrid_graph(nodes: Dict[str, Point],
                       prior_edge_prob: float = 0.5,
                       sigma: float = 1.0,
                       beam_center: float = 0.0,
                       beam_width: float = math.pi / 4,
                       false_positive: float = 0.01,
                       delta: float = 0.05,
                       n_observations: int = 30) -> Dict[Edge, float]:
    """
    Construct a complete undirected graph where each edge weight is the
    confidence‑adjusted hybrid score defined by `edge_confidence_score`.

    Parameters
    ----------
    nodes: dict mapping node identifiers to 2‑D coordinates.
    All other parameters are forwarded to `edge_confidence_score`.

    Returns
    -------
    dict mapping unordered edge tuples (sorted identifiers) to scores.
    """
    scores: Dict[Edge, float] = {}
    identifiers = list(nodes.keys())
    for i in range(len(identifiers)):
        for j in range(i + 1, len(identifiers)):
            a, b = identifiers[i], identifiers[j]
            score = edge_confidence_score(
                prior_prob=prior_edge_prob,
                p1=nodes[a],
                p2=nodes[b],
                sigma=sigma,
                beam_center=beam_center,
                beam_width=beam_width,
                false_positive=false_positive,
                delta=delta,
                n_observations=n_observations,
            )
            scores[(a, b)] = score
    return scores

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph with three nodes forming a triangle
    demo_nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, math.sqrt(3) / 2),
    }

    # Build the hybrid graph
    hybrid_weights = build_hybrid_graph(
        demo_nodes,
        prior_edge_prob=0.6,
        sigma=0.8,
        beam_center=0.0,
        beam_width=math.pi / 6,
        false_positive=0.02,
        delta=0.1,
        n_observations=50,
    )

    # Print the resulting edge scores
    for edge, score in hybrid_weights.items():
        print(f"Edge {edge}: hybrid score = {score:.4f}")

    # Simple sanity check: scores should be finite numbers
    assert all(math.isfinite(v) for v in hybrid_weights.values()), "Non‑finite scores detected"
    print("Smoke test completed successfully.")