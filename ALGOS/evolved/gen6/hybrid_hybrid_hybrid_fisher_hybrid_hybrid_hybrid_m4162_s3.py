# DARWIN HAMMER — match 4162, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_fisher_locali_hybrid_sketches_rlct_m33_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_indy_l_hybrid_hybrid_hybrid_m1506_s1.py (gen5)
# born: 2026-05-29T23:53:57Z

import hashlib
import math
import random
from typing import List, Tuple, Dict, Optional

import numpy as np


def gaussian_beam(theta: float, center: float, beam_width: float) -> float:
    """
    Gaussian intensity I(θ) of a beam centred at *center* with *beam_width*.

    Parameters
    ----------
    theta : float
        Angle at which the intensity is evaluated.
    center : float
        Beam centre.
    beam_width : float
        Standard deviation of the Gaussian. Must be positive.

    Returns
    -------
    float
        Intensity value I(θ).
    """
    if beam_width <= 0.0:
        raise ValueError("beam_width must be positive")
    z = (theta - center) / beam_width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, beam_width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a single angle θ under a Gaussian beam model.

    The implementation follows the definition
        F(θ) = (∂I/∂θ)² / I
    with a safeguard against division by zero.

    Parameters
    ----------
    theta : float
        Angle at which the Fisher information is evaluated.
    center : float
        Beam centre.
    beam_width : float
        Standard deviation of the Gaussian.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Fisher information at θ.
    """
    intensity = max(gaussian_beam(theta, center, beam_width), eps)
    # derivative of I w.r.t. θ
    derivative = intensity * (-(theta - center) / (beam_width * beam_width))
    return (derivative * derivative) / intensity


def _hash_item(item: str, depth: int, width: int) -> int:
    """
    Deterministic hash used by the Count‑Min sketch.

    Parameters
    ----------
    item : str
        Item to be hashed.
    depth : int
        Row index of the sketch (acts as a salt).
    width : int
        Number of columns in the sketch.

    Returns
    -------
    int
        Column index in the range [0, width).
    """
    digest = hashlib.sha256(f"{depth}:{item}".encode()).hexdigest()
    return int(digest, 16) % width


def count_min_sketch(
    items: List[str], width: int = 64, depth: int = 4
) -> np.ndarray:
    """
    Classic Count‑Min sketch implementation returning a NumPy matrix.

    Parameters
    ----------
    items : list[str]
        Iterable of hashable items.
    width : int, optional
        Number of columns (hash buckets). Must be > 0.
    depth : int, optional
        Number of hash functions / rows. Must be > 0.

    Returns
    -------
    np.ndarray
        A (depth, width) integer array containing the frequencies.
    """
    if width <= 0 or depth <= 0:
        raise ValueError("width and depth must be positive integers")
    sketch = np.zeros((depth, width), dtype=np.int64)

    for item in items:
        for d in range(depth):
            col = _hash_item(item, d, width)
            sketch[d, col] += 1
    return sketch


def aggregate_fisher_by_sketch(
    sketch: np.ndarray,
    fisher_scores: np.ndarray,
) -> float:
    """
    Produce a single scalar that quantifies the average Fisher information
    weighted by the estimated frequencies from the sketch.

    The weighting uses the median of the sketch rows for each column to obtain
    a robust frequency estimate.

    Parameters
    ----------
    sketch : np.ndarray
        Count‑Min sketch matrix of shape (depth, width).
    fisher_scores : np.ndarray
        Fisher information per item (1‑D array). Length must equal the number of
        distinct items that contributed to the sketch.

    Returns
    -------
    float
        Weighted average Fisher information.
    """
    if sketch.ndim != 2:
        raise ValueError("sketch must be a 2‑D array")
    if fisher_scores.ndim != 1:
        raise ValueError("fisher_scores must be a 1‑D array")
    # Estimate frequency per column as the median across rows (robust to hash collisions)
    freq_est = np.median(sketch, axis=0)
    # Normalise to obtain a probability distribution
    total = freq_est.sum()
    if total == 0:
        return 0.0
    probs = freq_est / total
    # Align the length of fisher_scores with the number of columns.
    # If there are more scores than columns we truncate; otherwise we pad with zeros.
    if len(fisher_scores) >= len(probs):
        fisher_vec = fisher_scores[: len(probs)]
    else:
        fisher_vec = np.concatenate(
            [fisher_scores, np.zeros(len(probs) - len(fisher_scores))]
        )
    return float(np.dot(probs, fisher_vec))


def confidence_bound_from_fisher(
    weighted_fisher: float, alpha: float = 1.0, eps: float = 1e-12
) -> float:
    """
    Translate a (weighted) Fisher information value into a confidence bound
    for a bandit algorithm. The bound shrinks like 1/√F, reflecting higher
    certainty when Fisher information is large.

    Parameters
    ----------
    weighted_fisher : float
        Weighted Fisher information (non‑negative).
    alpha : float, optional
        Scaling factor controlling exploration aggressiveness.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Confidence bound.
    """
    return alpha / math.sqrt(max(weighted_fisher, eps))


def bandit_action(
    propensity: float,
    expected_reward: float,
    confidence_bound: Optional[float] = None,
    algorithm: str = "hybrid",
) -> Dict[str, object]:
    """
    Create a bandit action dictionary. If ``confidence_bound`` is omitted,
    it defaults to a small constant (0.1) to keep the API backward compatible.

    Parameters
    ----------
    propensity : float
        Probability of selecting the action (0 ≤ propensity ≤ 1).
    expected_reward : float
        Model‑based estimate of the reward.
    confidence_bound : float | None, optional
        Exploration bonus. If ``None`` a default of 0.1 is used.
    algorithm : str, optional
        Identifier of the underlying algorithm.

    Returns
    -------
    dict
        Structured description of the chosen action.
    """
    if not (0.0 <= propensity <= 1.0):
        raise ValueError("propensity must be in [0, 1]")
    if confidence_bound is None:
        confidence_bound = 0.1
    action_id = hashlib.sha256(
        f"{propensity}:{expected_reward}:{confidence_bound}:{algorithm}".encode()
    ).hexdigest()
    return {
        "action_id": action_id,
        "propensity": propensity,
        "expected_reward": expected_reward,
        "confidence_bound": confidence_bound,
        "algorithm": algorithm,
    }


def update_bandit(
    action_id: str, reward: float, propensity: float
) -> Dict[str, object]:
    """
    Record the outcome of a bandit pull.

    Parameters
    ----------
    action_id : str
        Identifier returned by :func:`bandit_action`.
    reward : float
        Observed reward.
    propensity : float
        Propensity used when the action was taken.

    Returns
    -------
    dict
        Minimal update payload.
    """
    return {
        "context_id": action_id,
        "action_id": action_id,
        "reward": reward,
        "propensity": propensity,
    }


def hybrid_operation(
    items: List[str],
    sketch_width: int = 64,
    sketch_depth: int = 4,
    beam_center: float = 0.0,
    beam_width: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    End‑to‑end hybrid routine:

    1. Build a Count‑Min sketch of the input items.
    2. Compute a Fisher score for each *item index* interpreted as an angle.
    3. Aggregate the Fisher scores using the sketch frequencies.
    4. Derive a confidence bound from the aggregated Fisher information.

    The function returns the sketch, the raw Fisher scores and the derived
    confidence bound, enabling downstream components (e.g. a bandit) to use
    a mathematically grounded exploration term.

    Parameters
    ----------
    items : list[str]
        Collection of items to be processed.
    sketch_width : int, optional
        Number of columns in the sketch.
    sketch_depth : int, optional
        Number of rows (hash functions) in the sketch.
    beam_center : float, optional
        Centre of the Gaussian beam used for Fisher computation.
    beam_width : float, optional
        Width (standard deviation) of the Gaussian beam.

    Returns
    -------
    tuple
        ``(sketch, fisher_scores, confidence_bound)`` where
        *sketch* is a ``np.ndarray`` of shape ``(sketch_depth, sketch_width)``,
        *fisher_scores* is a 1‑D ``np.ndarray`` of length ``len(items)``,
        and *confidence_bound* is a ``float`` derived from the weighted Fisher.
    """
    sketch = count_min_sketch(items, width=sketch_width, depth=sketch_depth)

    # Treat each item index as a surrogate angle θ.
    fisher_scores = np.array(
        [
            fisher_score(theta=i, center=beam_center, beam_width=beam_width)
            for i in range(len(items))
        ],
        dtype=np.float64,
    )

    weighted_fisher = aggregate_fisher_by_sketch(sketch, fisher_scores)
    conf_bound = confidence_bound_from_fisher(weighted_fisher)

    return sketch, fisher_scores, conf_bound


if __name__ == "__main__":
    # Demo run
    demo_items = [f"item_{i}" for i in range(10)]

    sketch_mat, fisher_vec, cb = hybrid_operation(demo_items)

    print("Count‑Min Sketch (rows = hash functions):")
    for row in sketch_mat:
        print(row.tolist())

    print("\nFisher Scores (per item index):")
    print(fisher_vec.tolist())

    print(f"\nDerived confidence bound from Fisher information: {cb:.4f}")

    # Use the bound in a bandit action
    act = bandit_action(propensity=0.5, expected_reward=1.0, confidence_bound=cb)
    print("\nBandit Action:")
    print(act)

    upd = update_bandit(act["action_id"], reward=1.0, propensity=0.5)
    print("\nBandit Update:")
    print(upd)