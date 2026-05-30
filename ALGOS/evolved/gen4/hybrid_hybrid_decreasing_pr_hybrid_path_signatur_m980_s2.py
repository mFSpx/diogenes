# DARWIN HAMMER — match 980, survivor 2
# gen: 4
# parent_a: hybrid_decreasing_pruning_hybrid_hybrid_minimu_m63_s0.py (gen3)
# parent_b: hybrid_path_signature_kan_m30_s4.py (gen1)
# born: 2026-05-29T23:32:17Z

import math
import random
from typing import Tuple, List, Dict, Callable, Optional, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Core types and constants
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

EPISTEMIC_FLAGS: Tuple[str, ...] = (
    "FACT",
    "PROBABLE",
    "POSSIBLE",
    "BULLSHIT",
    "SURE_MAYBE",
)

# Mapping from epistemic label to a multiplicative confidence weight
_EPISTEMIC_WEIGHT: Dict[str, float] = {
    "FACT": 1.0,
    "PROBABLE": 0.85,
    "POSSIBLE": 0.6,
    "SURE_MAYBE": 0.4,
    "BULLSHIT": 0.0,
}


# ----------------------------------------------------------------------
# Geometry utilities (Parent A)
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, tp: float, fp: float) -> float:
    """
    Compute the marginal probability P(E) = P(E|H)P(H) + P(E|¬H)P(¬H).

    Parameters
    ----------
    prior: float
        Prior probability P(H) ∈ [0,1].
    tp: float
        True‑positive rate P(E|H) ∈ [0,1].
    fp: float
        False‑positive rate P(E|¬H) ∈ [0,1].

    Returns
    -------
    float
        Marginal probability P(E) ∈ (0,1].
    """
    if not (0.0 <= prior <= 1.0 and 0.0 <= tp <= 1.0 and 0.0 <= fp <= 1.0):
        raise ValueError("All probabilities must be in [0,1]")
    marginal = tp * prior + fp * (1.0 - prior)
    # Guard against degenerate zero (should not happen with proper rates)
    return max(marginal, 1e-12)


def bayes_update(prior: float, tp: float, fp: float) -> float:
    """
    Posterior P(H|E) = P(E|H)P(H) / P(E).

    Returns
    -------
    float
        Updated probability ∈ [0,1].
    """
    marginal = bayes_marginal(prior, tp, fp)
    return (tp * prior) / marginal


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


# ----------------------------------------------------------------------
# Path‑signature utilities (Parent B)
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
# Hybrid utilities – improved integration
# ----------------------------------------------------------------------
def _stable_logistic(x: float, scale: float = 1.0) -> float:
    """
    Numerically stable logistic function mapping ℝ → (0,1).

    Uses tanh formulation to avoid overflow for large |x|.
    """
    return 0.5 * (1.0 + math.tanh(scale * x / 2.0))


def _sample_linear_path(
    start: np.ndarray, end: np.ndarray, n_samples: int = 5
) -> np.ndarray:
    """
    Produce ``n_samples`` linearly spaced points (including endpoints)
    between ``start`` and ``end``.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")
    return np.linspace(start, end, n_samples)


def edge_signature_matrix(
    edge: Edge,
    nodes: Dict[str, Point],
    n_samples: int = 5,
) -> np.ndarray:
    """
    Compute the level‑2 signature matrix of the lead‑lag transformed
    piecewise‑linear path representing *edge*.

    The path is first sampled linearly (default 5 points) to give the
    signature a richer geometric content than a raw 2‑point edge.
    """
    p_start = np.asarray(nodes[edge[0]], dtype=float)
    p_end = np.asarray(nodes[edge[1]], dtype=float)

    raw_path = _sample_linear_path(p_start, p_end, n_samples)  # (n_samples, d)
    transformed = lead_lag_transform(raw_path)                # (2*n_samples-1, 2*d)
    return signature_level2(transformed)                      # (2*d, 2*d)


def _signature_likelihood(
    sig_mat: np.ndarray,
    edge_len: float,
    eps: float = 1e-12,
) -> float:
    """
    Derive a likelihood from the Frobenius norm of ``sig_mat``.

    The norm is normalised by the squared edge length to obtain a scale‑free
    quantity, then squashed to (0,1) with a stable logistic.
    """
    frob = np.linalg.norm(sig_mat, ord="fro")
    # Normalise – longer edges naturally produce larger signatures
    normed = frob / (edge_len ** 2 + eps)
    return _stable_logistic(normed)


def hybrid_edge_score(
    edge: Edge,
    nodes: Dict[str, Point],
    prior: float,
    t: float,
    *,
    lam: float = 1.0,
    alpha: float = 0.2,
    confidence_bps: int = 5000,
    epistemic_label: str = "PROBABLE",
    authority_class: str = "auto",
    rationale: str = "hybrid scoring",
    signature_samples: int = 5,
    false_positive_rate: float = 0.05,
) -> float:
    """
    Compute a deep hybrid score for *edge*.

    Steps
    -----
    1. Geometric length.
    2. Level‑2 signature of a sampled lead‑lag transformed path.
    3. Likelihood from the normalised Frobenius norm (logistic‑scaled).
    4. Bayesian posterior using an explicit false‑positive rate.
    5. Epistemic weighting from ``confidence_bps`` and ``epistemic_label``.
    6. Time‑dependent attenuation: score is multiplied by
       ``1 - prune_probability(t, lam, alpha)`` instead of a hard drop.

    Returns
    -------
    float
        Final hybrid score (≥0). Zero indicates complete pruning.
    """
    # 1. geometric length
    len_edge = length(nodes[edge[0]], nodes[edge[1]])

    # 2. signature matrix (richer geometry via sampling)
    sig_mat = edge_signature_matrix(edge, nodes, n_samples=signature_samples)

    # 3. likelihood from signature
    likelihood = _signature_likelihood(sig_mat, len_edge)

    # 4. Bayesian update (true‑positive = likelihood)
    posterior = bayes_update(prior, tp=likelihood, fp=false_positive_rate)

    # 5. epistemic weighting
    if epistemic_label not in _EPISTEMIC_WEIGHT:
        raise ValueError(f"unknown epistemic label: {epistemic_label!r}")
    epistemic_weight = _EPISTEMIC_WEIGHT[epistemic_label]
    confidence_factor = (confidence_bps / 10000.0) * epistemic_weight

    # 6. time‑dependent attenuation (smooth, no stochastic drop)
    attenuation = 1.0 - prune_probability(t, lam, alpha)
    score = posterior * len_edge * confidence_factor * attenuation
    return max(score, 0.0)


def prune_and_score_edges(
    edges: List[Edge],
    nodes: Dict[str, Point],
    prior: float,
    t: float,
    *,
    lam: float = 1.0,
    alpha: float = 0.2,
    confidence_bps: int = 5000,
    epistemic_label: str = "PROBABLE",
    authority_class: str = "auto",
    rationale: str = "hybrid scoring",
    signature_samples: int = 5,
    false_positive_rate: float = 0.05,
) -> List[Tuple[Edge, float]]:
    """
    Compute hybrid scores for *edges* and return them sorted by descending
    score. Edges whose attenuated score falls below a dynamic threshold
    (proportional to the current pruning probability) are omitted.

    The threshold is defined as:

        threshold = max_score * prune_probability(t, lam, alpha)

    This yields a deterministic, time‑aware pruning that respects the
    underlying scores.
    """
    scored: List[Tuple[Edge, float]] = []
    for e in edges:
        s = hybrid_edge_score(
            e,
            nodes,
            prior,
            t,
            lam=lam,
            alpha=alpha,
            confidence_bps=confidence_bps,
            epistemic_label=epistemic_label,
            authority_class=authority_class,
            rationale=rationale,
            signature_samples=signature_samples,
            false_positive_rate=false_positive_rate,
        )
        scored.append((e, s))

    # Sort descending by score
    scored.sort(key=lambda pair: pair[1], reverse=True)

    if not scored:
        return []

    max_score = scored[0][1]
    threshold = max_score * prune_probability(t, lam, alpha)

    # Keep only edges meeting the threshold
    return [(e, s) for e, s in scored if s >= threshold]