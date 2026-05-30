# DARWIN HAMMER — match 608, survivor 2
# gen: 5
# parent_a: hybrid_cockpit_metrics_rectified_flow_m10_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m385_s0.py (gen4)
# born: 2026-05-29T23:30:04Z

import numpy as np
import math
import random
import sys
import pathlib
from typing import List, Callable, Any
from collections import Counter

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def words(text: str) -> List[str]:
    """Split a string into lowercase alphabetic words."""
    return [word for word in (text or "").lower().split() if word.isalpha()]


# ----------------------------------------------------------------------
# Metric calculations (original “cockpit_metrics”)
# ----------------------------------------------------------------------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Proportion of claims that are backed by evidence."""
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are truly OK."""
    total = displayed_ok + unknown_displayed_as_ok
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, displayed_ok / total))


# ----------------------------------------------------------------------
# Stylometry (original “hybrid_hybrid_hybrid_hard_truth_ma_kan_m27_s4”)
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": {
        "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
        "he", "him", "his", "she", "her", "hers", "they", "them", "their",
        "theirs", "we", "us", "our", "ours"
    },
    "article": {"a", "an", "the"},
    "preposition": {
        "about", "above", "after", "against", "around", "as", "at", "before",
        "behind", "below", "between", "by", "during", "for", "from", "in",
        "into", "of", "off", "on", "onto", "over", "through", "to", "under",
        "with", "without"
    },
    "auxiliary": {
        "am", "are", "be", "been", "being", "can", "could", "did", "do",
        "does", "had", "has", "have", "is", "may", "might", "must", "shall",
        "should", "was", "were", "will", "would"
    },
    "conjunction": {
        "and", "but", "or", "nor", "so", "yet", "because", "although",
        "while", "if", "when", "where", "whereas", "unless", "until"
    },
    "negation": {
        "no", "not", "never", "none", "neither", "cannot", "can't", "won't",
        "don't", "didn't", "isn't", "aren't", "wasn't", "weren't"
    },
    "quantifier": {
        "all", "any", "both", "each", "few", "many", "more", "most", "much",
        "none", "several", "some", "such"
    },
    "adverb_common": {
        "very", "really", "just", "still", "already", "also", "even", "only",
        "then", "there", "here", "now", "often", "always", "sometimes"
    },
}


def stylometry_features(text: str, dim: int) -> np.ndarray:
    """
    Compute a fixed‑length stylometry vector.

    The vector consists of normalized frequencies of the first ``dim``
    functional categories defined in ``FUNCTION_CATS``.  If ``dim`` exceeds
    the number of categories, the remaining entries are padded with zeros.
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)

    # Preserve deterministic order of categories
    cat_names = list(FUNCTION_CATS.keys())
    features = []

    for i in range(dim):
        if i < len(cat_names):
            cat = cat_names[i]
            cat_words = FUNCTION_CATS[cat]
            freq = sum(cnt[w] for w in cat_words) / total
        else:
            freq = 0.0
        features.append(freq)

    return np.array(features, dtype=np.float64)


def lsm_vector(text: str) -> np.ndarray:
    """Convenient wrapper returning a 6‑dim stylometry vector."""
    return stylometry_features(text, dim=6)


# ----------------------------------------------------------------------
# Vector‑field dynamics (shared core)
# ----------------------------------------------------------------------
def flow_target(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Desired displacement from ``x0`` to ``x1``."""
    return np.asarray(x1, dtype=np.float64) - np.asarray(x0, dtype=np.float64)


def flow_loss(v_pred: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> float:
    """Mean‑squared error between predicted and target flow."""
    diff = np.asarray(v_pred, dtype=np.float64) - flow_target(x0, x1)
    return float(np.mean(diff ** 2))


# ----------------------------------------------------------------------
# Hybrid integration – deeper mathematical coupling
# ----------------------------------------------------------------------
def weighted_vector_field(
    state: np.ndarray,
    stylometry: np.ndarray,
    t: float,
    metrics: dict[str, float],
) -> np.ndarray:
    """
    Produce a vector that blends:

    * A base dynamics term (here a simple linear drift towards the origin).
    * A stylometry‑driven perturbation.
    * Metric‑based scaling that reflects “cockpit” honesty and evidence quality.

    The weighting scheme is deliberately nonlinear to encourage richer
    interactions between the subsystems.
    """
    # Base drift: push state toward zero with strength proportional to (1‑t)
    base = -state * (1.0 - t)

    # Stylometry influence: project stylometry onto the state space
    # (repeat or truncate to match dimensionality)
    if stylometry.size != state.size:
        # Broadcast or truncate safely
        if stylometry.size < state.size:
            repeat_factor = int(np.ceil(state.size / stylometry.size))
            styl = np.tile(stylometry, repeat_factor)[: state.size]
        else:
            styl = stylometry[: state.size]
    else:
        styl = stylometry

    # Metric coupling: combine anti‑slop and honesty into a single scalar
    metric_factor = (
        0.6 * metrics.get("anti_slop", 1.0) + 0.4 * metrics.get("honesty", 1.0)
    )
    metric_factor = np.clip(metric_factor, 0.0, 1.0)

    # Non‑linear mixing
    perturb = np.tanh(styl) * metric_factor * t

    return base + perturb


def euler_solve(
    v_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    x0: np.ndarray,
    steps: int = 10,
) -> np.ndarray:
    """
    Simple explicit Euler integrator for a time‑independent vector field.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    dt = 1.0 / steps
    ts = np.linspace(0.0, 1.0 - dt, steps)
    traj = np.empty((steps + 1,) + x0.shape, dtype=np.float64)
    traj[0] = x0
    z = x0.copy()

    for k, t in enumerate(ts):
        v = v_fn(z, lsm_vector("This is a sample text"), float(t))
        z = z + dt * np.asarray(v, dtype=np.float64)
        traj[k + 1] = z

    return traj


def hybrid_solve(
    x0: np.ndarray,
    claims_with_evidence: int,
    total_claims_emitted: int,
    displayed_ok: int,
    unknown_displayed_as_ok: int,
    text: str,
    steps: int = 10,
) -> np.ndarray:
    """
    Integrated solver that fuses cockpit metrics with stylometry‑driven dynamics.

    Parameters
    ----------
    x0 : np.ndarray
        Initial state vector.
    claims_with_evidence, total_claims_emitted, displayed_ok, unknown_displayed_as_ok :
        Scalars feeding the cockpit‑metric subsystem.
    text : str
        Input text for stylometry extraction.
    steps : int, optional
        Number of Euler integration steps (default 10).

    Returns
    -------
    np.ndarray
        Trajectory of shape (steps+1, *x0.shape).
    """
    # Compute cockpit metrics once (they are static for the trajectory)
    metrics = {
        "anti_slop": anti_slop_ratio(claims_with_evidence, total_claims_emitted),
        "honesty": cockpit_honesty(displayed_ok, unknown_displayed_as_ok),
    }

    # Pre‑compute stylometry vector (static for this simple example)
    styl = lsm_vector(text)

    def combined_v(state: np.ndarray, _: np.ndarray, t: float) -> np.ndarray:
        # The second argument is ignored because we already have ``styl``.
        return weighted_vector_field(state, styl, t, metrics)

    return euler_solve(combined_v, x0, steps=steps)


# ----------------------------------------------------------------------
# Demo / sanity check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example parameters
    x0 = np.array([0.5, -0.3, 0.1])
    claims_with_evidence = 8
    total_claims_emitted = 10
    displayed_ok = 7
    unknown_displayed_as_ok = 2
    text = "The quick brown fox jumps over the lazy dog while it watches."

    traj = hybrid_solve(
        x0,
        claims_with_evidence,
        total_claims_emitted,
        displayed_ok,
        unknown_displayed_as_ok,
        text,
        steps=20,
    )
    np.set_printoptions(precision=4, suppress=True)
    print("Trajectory ({} steps):".format(traj.shape[0] - 1))
    print(traj)