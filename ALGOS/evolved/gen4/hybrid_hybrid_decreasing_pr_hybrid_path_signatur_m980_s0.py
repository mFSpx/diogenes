# DARWIN HAMMER — match 980, survivor 0
# gen: 4
# parent_a: hybrid_decreasing_pruning_hybrid_hybrid_minimu_m63_s0.py (gen3)
# parent_b: hybrid_path_signature_kan_m30_s4.py (gen1)
# born: 2026-05-29T23:32:17Z

"""Hybrid algorithm merging graph pruning with path‑signature analysis.

Parents:
- **hybrid_decreasing_pruning_hybrid_hybrid_minimu_m63_s0.py** – provides
  Bayesian‑based pruning, edge length, and epistemic certainty handling.
- **hybrid_path_signature_kan_m30_s4.py** – supplies lead‑lag transform and
  exact level‑2 signature computation for piecewise linear paths.

Mathematical bridge:
Each graph edge defines a 2‑point geometric path.  By applying the lead‑lag
transform to this tiny path we obtain a higher‑dimensional representation
whose level‑2 signature (a matrix) captures the geometric “area” spanned by
the edge.  The Frobenius norm of that signature is interpreted as a
likelihood for a Bayesian update of the edge’s prior survival probability.
The updated posterior is then combined with Euclidean length and an
epistemic certainty flag to yield a hybrid edge score.  Finally a
time‑dependent decreasing pruning probability discards edges with low
scores, unifying both parent topologies into a single workflow.
"""

import math
import random
import sys
import pathlib
from typing import Tuple, List, Dict, Callable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Parent A – graph utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability P(E) = P(E|H)P(H) + P(E|¬H)P(¬H)."""
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P(E|H)P(H) / P(E)."""
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Tuple[str, ...] = (),
) -> Dict:
    """Create an epistemic certainty flag."""
    if label not in EPISTEMIC_FLAGS:
        raise ValueError(f"unknown epistemic flag: {label!r}")
    if not 0 <= int(confidence_bps) <= 10000:
        raise ValueError("confidence_bps must be 0..10000")
    return {
        "label": label,
        "confidence_bps": int(confidence_bps),
        "authority_class": authority_class,
        "rationale": rationale,
        "evidence_refs": evidence_refs,
    }


def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Decreasing pruning probability p(t) = min(1, λ·exp(-α·t))."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))


def prune_edges(
    edges: List[Edge],
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: int | str | None = None,
) -> List[Edge]:
    """Randomly drop edges according to the decreasing probability p(t)."""
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]


# ----------------------------------------------------------------------
# Parent B – path signature utilities
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform of a path.

    Parameters
    ----------
    path : np.ndarray of shape (T, d)

    Returns
    -------
    np.ndarray of shape (2*T-1, 2*d)
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (T, d)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)

    # even rows: (X_t, X_t)
    out[0::2, :d] = path
    out[0::2, d:] = path

    # odd rows: (X_{t+1}, X_t) – last odd row does not exist
    out[1::2, :d] = path[1:]
    out[1::2, d:] = path[:-1]

    return out


def signature_level2(path: np.ndarray) -> np.ndarray:
    """
    Exact level‑2 signature (matrix) using left‑point Riemann sums.

    Returns a (d, d) matrix for a d‑dimensional path.
    """
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running = path[:-1] - path[0]               # (T‑1, d)
    return running.T @ increments               # (d, d)


# ----------------------------------------------------------------------
# Hybrid functionality
# ----------------------------------------------------------------------
def edge_signature_matrix(edge: Edge, nodes: Dict[str, Point]) -> np.ndarray:
    """
    Compute the level‑2 signature matrix of the lead‑lag transformed
    2‑point path defined by *edge*.

    The returned matrix has shape (2*d, 2*d) where d is the spatial dimension.
    """
    p_start = np.asarray(nodes[edge[0]], dtype=float)
    p_end = np.asarray(nodes[edge[1]], dtype=float)

    # Build a (2, d) path: start → end
    raw_path = np.stack([p_start, p_end], axis=0)          # (2, d)

    # Lead‑lag transform expands to (3, 2*d)
    transformed = lead_lag_transform(raw_path)            # (3, 2*d)

    # Level‑2 signature of the transformed path
    sig = signature_level2(transformed)                   # (2*d, 2*d)
    return sig


def logistic(x: float, scale: float = 1.0) -> float:
    """Simple logistic squashing to (0,1)."""
    return 1.0 / (1.0 + math.exp(-scale * x))


def hybrid_edge_score(
    edge: Edge,
    nodes: Dict[str, Point],
    prior: float,
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    confidence_bps: int = 5000,
    authority_class: str = "auto",
    rationale: str = "hybrid scoring",
) -> float:
    """
    Produce a hybrid score for *edge* by:

    1. Computing Euclidean length.
    2. Deriving a likelihood from the Frobenius norm of the edge's
       level‑2 signature (logistic‑scaled to [0,1]).
    3. Performing a Bayesian update of the supplied *prior*.
    4. Multiplying by a normalized confidence factor.
    5. Applying the time‑dependent pruning probability (edges with high
       pruning probability receive a score of 0).

    The function returns the final scalar score (≥0).
    """
    # 1. geometric length
    len_edge = length(nodes[edge[0]], nodes[edge[1]])

    # 2. signature‑derived likelihood
    sig_mat = edge_signature_matrix(edge, nodes)
    frob_norm = np.linalg.norm(sig_mat, ord="fro")
    likelihood = logistic(frob_norm)  # in (0,1)

    # 3. Bayesian update (false positive = 1 - prior as a simple model)
    false_positive = 1.0 - prior
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)

    # 4. confidence factor (bps → [0,1])
    confidence_factor = confidence_bps / 10000.0

    # 5. pruning probability
    p_prune = prune_probability(t, lam, alpha)
    if random.random() < p_prune:
        return 0.0

    # Final hybrid score
    score = posterior * len_edge * confidence_factor
    return score


def prune_and_score_edges(
    edges: List[Edge],
    nodes: Dict[str, Point],
    prior: float,
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    confidence_bps: int = 5000,
    authority_class: str = "auto",
    rationale: str = "hybrid scoring",
    seed: int | str | None = None,
) -> List[Tuple[Edge, float]]:
    """
    Apply time‑dependent pruning and compute hybrid scores for the surviving
    edges.  Returns a list of ``(edge, score)`` tuples sorted by descending
    score.
    """
    # First apply the stochastic pruning step
    kept_edges = prune_edges(edges, t, lam, alpha, seed)

    results: List[Tuple[Edge, float]] = []
    for e in kept_edges:
        sc = hybrid_edge_score(
            e,
            nodes,
            prior,
            t,
            lam,
            alpha,
            confidence_bps,
            authority_class,
            rationale,
        )
        if sc > 0.0:
            results.append((e, sc))

    # Sort for convenience
    results.sort(key=lambda pair: pair[1], reverse=True)
    return results


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple triangle graph
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "A")]

    # Parameters
    prior = 0.6
    t = 0.0
    lam = 1.0
    alpha = 0.2
    confidence_bps = 8000
    authority_class = "demo"
    rationale = "test run"

    scored = prune_and_score_edges(
        edges,
        nodes,
        prior,
        t,
        lam,
        alpha,
        confidence_bps,
        authority_class,
        rationale,
        seed=42,
    )

    print("Hybrid edge scores (edge, score):")
    for edge, score in scored:
        print(f"{edge}: {score:.6f}")

    # Verify that the function runs without raising.
    sys.exit(0)