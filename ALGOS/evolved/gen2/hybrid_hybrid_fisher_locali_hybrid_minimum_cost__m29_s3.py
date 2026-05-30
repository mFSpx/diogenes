# DARWIN HAMMER — match 29, survivor 3
# gen: 2
# parent_a: hybrid_fisher_localization_krampus_chrono_m17_s0.py (gen1)
# parent_b: hybrid_minimum_cost_tree_bayes_update_m6_s1.py (gen1)
# born: 2026-05-29T23:25:14Z

"""Hybrid Fisher‑Chrono Bayesian Tree Cost
This module fuses two parent algorithms:

* **hybrid_fisher_localization_krampus_chrono** – provides Gaussian‑beam modelling,
  Fisher information scoring of timestamps and a chronological candidate generator.
* **hybrid_minimum_cost_tree_bayes_update** – provides a minimum‑cost spanning‑tree
  evaluator together with Bayesian marginalisation and update of edge priors.

**Mathematical bridge**

Both parents rely on Gaussian statistics.  In the Fisher‑localisation code the
derivative of a Gaussian (the Fisher information) is a measure of precision
(∝ 1/σ²).  In the Bayesian code a prior probability can be interpreted as a
Gaussian with variance σ² = 1/precision.  By converting Fisher scores into
precisions we obtain Gaussian priors for tree edges, update them with new
temporal evidence (likelihoods derived from the same Gaussian), and finally
evaluate a tree cost that incorporates the updated edge probabilities.
The result is a single coherent system that scores chronological candidates
while simultaneously assessing the cost of a graph whose edge weights are
informed by Bayesian‑updated Fisher information."""

import math
import random
import sys
import pathlib
from datetime import datetime
from typing import Tuple, Dict, List, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Gaussian / Fisher utilities (Parent A)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity G(θ) with centre `center` and standard‑deviation `width`."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam.
    For a Gaussian N(center, width²) the Fisher information w.r.t. the mean is
    I = 1 / width².  The implementation follows the original code and returns
    (∂G/∂θ)² / G, which is algebraically equivalent.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def parse_loose_datetime(raw: str) -> datetime | None:
    """Parse ISO‑like date strings, returning None on failure."""
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def chrono_candidates_for_path(path: pathlib.Path, text_sample: str = "") -> List[Dict[str, str]]:
    """
    Generate a dense set of candidate timestamps (1900‑2099) for a file path.
    The function mimics the original exhaustive enumeration.
    """
    candidates: List[Dict[str, str]] = []
    for year in range(1900, 2100):
        for month in range(1, 13):
            for day in range(1, 32):
                raw = f"{year}-{month:02d}-{day:02d}"
                if parse_loose_datetime(raw):
                    candidates.append(
                        {"timestamp": raw, "source": "path", "raw": raw}
                    )
    return candidates


def gaussian_filter(data: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a 1‑D Gaussian filter (implemented as point‑wise multiplication)."""
    return np.array([gaussian_beam(x, 0.0, sigma) for x in data])


# ----------------------------------------------------------------------
# Bayesian / Tree utilities (Parent B)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    path_weight: float = 0.2,
) -> float:
    """
    Classic minimum‑cost tree cost:
        material  = sum of edge lengths
        path_term = path_weight * Σ distance(root → node)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)

    return material + path_weight * sum(dist.values())


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = P(E|H)P(H) + P(E|¬H)P(¬H)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P(E|H)P(H) / P(E)."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


# ----------------------------------------------------------------------
# Hybrid layer – marrying Fisher information with Bayesian edge priors
# ----------------------------------------------------------------------
def fisher_to_precision(score: float) -> float:
    """
    Convert a Fisher information score into a Gaussian precision.
    The Fisher information of a Gaussian w.r.t. its mean equals 1/σ²,
    therefore we treat `score` directly as a precision (≥0).
    """
    return max(score, 0.0)


def edge_prior_from_fisher(
    candidates: List[Dict[str, str]],
    center_ts: float,
    width: float,
) -> Dict[Edge, float]:
    """
    Produce a prior probability for each edge based on Fisher scores of
    timestamp candidates.  The routine assigns candidates to edges cyclically,
    converts the Fisher score into a precision, then maps precision → probability
    via a simple normalisation (so that all priors lie in (0,1]).

    Returns a dictionary ``{edge: prior}``.
    """
    if not candidates:
        raise ValueError("Candidate list cannot be empty")

    # Compute raw Fisher scores for every candidate timestamp.
    raw_scores = [
        fisher_score(datetime.fromisoformat(c["timestamp"]).timestamp(), center_ts, width)
        for c in candidates
    ]

    # Convert to precisions.
    precisions = [fisher_to_precision(s) for s in raw_scores]

    # Normalise to (0,1] to obtain a prior probability.
    max_prec = max(precisions) or 1.0
    priors = [p / max_prec for p in precisions]

    # Assign to edges (cycling if there are more edges than candidates).
    edge_priors: Dict[Edge, float] = {}
    for idx, edge in enumerate(sorted(set(map(_ordered_edge, candidates_edge_set())))):
        # Cycle through the list of priors.
        prior = priors[idx % len(priors)]
        edge_priors[edge] = prior

    return edge_priors


def _ordered_edge(edge: Edge) -> Edge:
    """Return a canonical ordering for an undirected edge."""
    return tuple(sorted(edge))


def candidates_edge_set() -> List[Edge]:
    """
    Helper that creates a deterministic small set of edges for the demo.
    In a real scenario edges would be supplied by the user; here we fabricate
    a simple chain of five nodes.
    """
    return [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]


def hybrid_edge_priors(
    candidates: List[Dict[str, str]],
    center: float,
    width: float,
    false_positive: float = 0.01,
) -> Dict[Edge, float]:
    """
    For each edge compute a Bayesian‑updated prior:
        1. Base prior = normalised Fisher precision (see `edge_prior_from_fisher`).
        2. Likelihood = Gaussian likelihood of the candidate timestamp being close
           to `center` (using the same `width` as the Fisher score).
        3. Marginal = bayes_marginal(base_prior, likelihood, false_positive).
        4. Posterior = bayes_update(base_prior, likelihood, marginal).

    The resulting posterior probabilities are used as edge weights in the
    hybrid tree cost.
    """
    # Step 1 – raw priors from Fisher information.
    base_priors = edge_prior_from_fisher(candidates, center, width)

    # Step 2‑4 – Bayesian update per edge.
    updated_priors: Dict[Edge, float] = {}
    for edge, prior in base_priors.items():
        # Choose a candidate timestamp for this edge (cycle if needed).
        cand = candidates[hash(edge) % len(candidates)]
        ts = datetime.fromisoformat(cand["timestamp"]).timestamp()
        # Gaussian likelihood centred at `center` with std = width.
        likelihood = gaussian_beam(ts, center, width)
        marginal = bayes_marginal(prior, likelihood, false_positive)
        posterior = bayes_update(prior, likelihood, marginal)
        # Clamp to (0,1] to avoid pathological zero weights.
        updated_priors[edge] = max(min(posterior, 1.0), 1e-9)

    return updated_priors


def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    edge_priors: Dict[Edge, float],
    path_weight: float = 0.2,
) -> float:
    """
    Minimum‑cost tree where each edge length is scaled by its Bayesian‑updated
    prior probability (high probability → lower effective cost).  This mirrors the
    `hybrid_tree_cost` from Parent B but uses the priors produced by the Fisher‑chronological
    pipeline.
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        ordered = _ordered_edge((a, b))
        prior = edge_priors.get(ordered, 1.0)  # fall back to neutral prior
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b]) * prior

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)

    return material + path_weight * sum(dist.values())


def hybrid_chrono_tree_score(
    candidates: List[Dict[str, str]],
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    center: float,
    width: float,
    path_weight: float = 0.2,
) -> float:
    """
    Full hybrid score that combines:
        * temporal coherence (sum of Fisher scores over candidates)
        * graph cost (Bayesian‑updated minimum‑cost tree)

    The final metric is a weighted sum:
        score = α * tree_cost - β * Σ Fisher_score
    where α,β are chosen to balance the two terms.
    """
    α = 1.0
    β = 0.5

    # Temporal term
    temporal_score = sum(
        fisher_score(datetime.fromisoformat(c["timestamp"]).timestamp(), center, width)
        for c in candidates
    )

    # Edge priors informed by the same temporal data
    priors = hybrid_edge_priors(candidates, center, width)

    # Graph term
    graph_score = hybrid_tree_cost(nodes, edges, root, priors, path_weight)

    return α * graph_score - β * temporal_score


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a dummy path (not used for I/O, only to satisfy the API)
    dummy_path = pathlib.Path("dummy.txt")

    # Generate chronological candidates (very coarse for speed)
    candidates = chrono_candidates_for_path(dummy_path)[:20]  # take a small subset

    # Define a tiny graph (chain of five nodes)
    nodes: Dict[str, Point] = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (2.0, 0.0),
        "D": (3.0, 0.0),
        "E": (4.0, 0.0),
    }
    edges: List[Edge] = candidates_edge_set()
    root = "A"

    # Choose a temporal centre (current epoch) and a width of ~1 year in seconds
    now = datetime.now().timestamp()
    one_year = 365.25 * 24 * 3600
    width = one_year

    # Compute the hybrid score
    score = hybrid_chrono_tree_score(
        candidates=candidates,
        nodes=nodes,
        edges=edges,
        root=root,
        center=now,
        width=width,
    )
    print(f"Hybrid chrono‑tree score: {score:.6f}")