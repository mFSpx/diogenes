# DARWIN HAMMER — match 2754, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2146_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hoeffd_m881_s0.py (gen5)
# born: 2026-05-29T23:45:36Z

"""Hybrid Algorithm: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2146_hoeffding_fisher.py
Integrates Parent Algorithm A (semantic neighbor Bayesian edge weighting) and
Parent Algorithm B (Gaussian beam, Fisher information, Hoeffding bound).

Mathematical Bridge
------------------
1. Spatial distance `d` between two points is interpreted as an angular
   deviation `θ` for the Gaussian beam model of Parent B.
2. The Gaussian intensity `I(θ)` serves as the *likelihood* in the Bayesian
   update from Parent A.
3. The Bayesian posterior `P(H|E)` becomes a *confidence weight* that scales
   the effective sample size `n_eff = n * P(H|E)` used in the Hoeffding bound.
4. Fisher information `𝓘(θ)` (Parent B) is multiplied by the posterior to
   produce a *certainty* term.
5. An edge’s hybrid score combines:
   - Euclidean length (cost),
   - Bayesian‑scaled Fisher certainty,
   - Inverse Hoeffding bound (confidence radius).

The resulting scalar can be used for minimum‑cost tree construction while
preserving uncertainty quantification.

"""

import math
import random
import sys
import pathlib
import numpy as np

# Type aliases for clarity
Point = tuple[float, float]
Edge = tuple[int, int]   # indices of points in a list


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) centred at *center* with standard deviation *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angular observation."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def hoeffding_bound(range_: float, delta: float, n: int) -> float:
    """Hoeffding bound ε = sqrt( (R² * ln(1/δ)) / (2n) )."""
    if range_ <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("Invalid arguments for Hoeffding bound")
    return math.sqrt((range_ * range_ * math.log(1.0 / delta)) / (2.0 * n))


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = likelihood·prior + false_positive·(1−prior)."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("Probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = prior·likelihood / marginal."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


def hybrid_edge_weight(
    a: Point,
    b: Point,
    prior: float,
    false_positive: float,
    beam_center: float = 0.0,
    beam_width: float = 1.0,
) -> float:
    """
    Compute a hybrid weight for the edge (a,b).

    Steps:
    1. Distance → θ (treated as angular deviation).
    2. Gaussian beam gives likelihood.
    3. Bayesian posterior using prior & false positive rate.
    4. Fisher information at θ, scaled by posterior.
    5. Combine cost (distance) with certainty (scaled Fisher) into a single weight.
    """
    d = length(a, b)
    theta = d  # interpret raw distance as angular deviation

    likelihood = gaussian_beam(theta, beam_center, beam_width)
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)

    fisher = fisher_score(theta, beam_center, beam_width)

    # Higher posterior & Fisher → lower effective cost (more certain)
    certainty_factor = posterior * fisher + 1e-9  # avoid division by zero
    weight = d / certainty_factor
    return weight


def hybrid_hoeffding_confidence(
    prior: float,
    likelihood: float,
    false_positive: float,
    range_: float,
    delta: float,
    n: int,
) -> float:
    """
    Compute a confidence-adjusted Hoeffding bound.

    The Bayesian posterior rescales the effective sample size,
    yielding a tighter bound when the posterior is high.
    """
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)

    n_eff = max(1, int(n * posterior))  # at least one observation
    return hoeffding_bound(range_, delta, n_eff)


def hybrid_tree_edge_scores(
    points: list[Point],
    priors: list[float],
    false_positive: float,
    beam_center: float,
    beam_width: float,
    hoeffding_range: float,
    hoeffding_delta: float,
    n_observations: int,
) -> dict[Edge, float]:
    """
    Evaluate every possible undirected edge between points and assign a hybrid score.

    The score is lower for edges that are short, have high Bayesian‑scaled Fisher
    certainty, and enjoy a small Hoeffding confidence radius.
    """
    scores: dict[Edge, float] = {}
    m = len(points)

    for i in range(m):
        for j in range(i + 1, m):
            # Edge geometry
            d = length(points[i], points[j])
            theta = d

            # Bayesian quantities (using point‑specific prior)
            prior_i = priors[i]
            prior_j = priors[j]
            # For simplicity, average the two priors
            prior = 0.5 * (prior_i + prior_j)

            likelihood = gaussian_beam(theta, beam_center, beam_width)

            # Scaled Hoeffding bound
            ho_bound = hybrid_hoeffding_confidence(
                prior, likelihood, false_positive, hoeffding_range, hoeffding_delta, n_observations
            )

            # Scaled Fisher certainty
            fisher = fisher_score(theta, beam_center, beam_width)
            posterior = bayes_update(prior, likelihood, bayes_marginal(prior, likelihood, false_positive))
            certainty = posterior * fisher

            # Hybrid score: combine cost, certainty, and confidence radius
            # Lower score = more desirable edge
            score = d / (certainty + 1e-9) + ho_bound
            scores[(i, j)] = score

    return scores


if __name__ == "__main__":
    # Simple smoke test with a triangle of points
    pts = [(0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3) / 2)]
    priors = [0.8, 0.6, 0.9]          # example prior certainties per point
    fp = 0.05                         # false‑positive rate
    center = 0.0
    width = 1.0
    hoeff_range = 1.0
    hoeff_delta = 0.05
    n_obs = 100

    # Compute hybrid edge weights
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            w = hybrid_edge_weight(
                pts[i],
                pts[j],
                priors[i],
                fp,
                beam_center=center,
                beam_width=width,
            )
            print(f"Edge ({i},{j}) hybrid weight: {w:.6f}")

    # Compute full edge scores for tree construction
    edge_scores = hybrid_tree_edge_scores(
        points=pts,
        priors=priors,
        false_positive=fp,
        beam_center=center,
        beam_width=width,
        hoeffding_range=hoeff_range,
        hoeffding_delta=hoeff_delta,
        n_observations=n_obs,
    )
    for e, s in edge_scores.items():
        print(f"Edge {e} combined hybrid score: {s:.6f}")

    # Verify that Hoeffding bound is positive and Fisher info finite
    theta_test = 0.5
    assert gaussian_beam(theta_test, center, width) > 0
    assert fisher_score(theta_test, center, width) > 0
    assert hoeffding_bound(hoeff_range, hoeff_delta, n_obs) > 0

    print("Smoke test completed successfully.")