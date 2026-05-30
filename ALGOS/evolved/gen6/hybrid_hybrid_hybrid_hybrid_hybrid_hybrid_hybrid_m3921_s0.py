# DARWIN HAMMER — match 3921, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m530_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_rectified_flo_m1340_s1.py (gen5)
# born: 2026-05-29T23:52:37Z

"""Hybrid Signature-Tree Bayesian Flow Engine
================================================

This module fuses the two parent algorithms:

* **Parent A** – ``hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m530_s3.py``  
  Provides set‑based MinHash signatures, ternary similarity, and a
  regret‑weighted decision representation.

* **Parent B** – ``hybrid_hybrid_hybrid_minimu_hybrid_rectified_flo_m1340_s1.py``  
  Supplies deterministic tree geometry utilities, a Bayesian posterior
  update, and the notion of a rectified flow (linear interpolation with
  non‑negative rectification).

**Mathematical bridge**

The bridge is built by treating each *signature* produced by Parent A as a
high‑dimensional point.  Euclidean distances between these points become the
edge lengths of a tree (Parent B).  The Bayesian posterior update is then used
to modulate those geometric edges: the prior is the MinHash similarity, the
likelihood is a sigmoid‑scaled distance, and a small false‑positive rate
(`FP`) injects regularisation.  The resulting posterior weights are finally
combined with a rectified flow interpolation, yielding a unified hybrid
score that simultaneously respects set similarity, geometric proximity, and
probabilistic belief updating.

The implementation below contains three core hybrid functions that
demonstrate this integration.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – signature utilities
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        hashlib.blake2b(data, digest_size=8).digest(), "big"
    )  # type: ignore[name-defined]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash‑style signature of length *k*."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def ternary_vector_similarity(vector_a: List[int], vector_b: List[int]) -> float:
    """Exact match similarity for ternary vectors."""
    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have equal length")
    return sum(1 for a, b in zip(vector_a, vector_b) if a == b) / len(vector_a)


# ----------------------------------------------------------------------
# Parent B – tree geometry and Bayesian update
# ----------------------------------------------------------------------


Point = Tuple[float, float]
Edge = Tuple[int, int]  # indices of nodes in the signature list


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayesian_posterior(p_prior: float, L: float, FP: float) -> float:
    """
    Bayesian posterior update.

    p_prior – prior probability (e.g. similarity)
    L       – likelihood (e.g. sigmoid of negative distance)
    FP      – false‑positive rate (regularisation term)
    """
    denominator = L * p_prior + FP * (1.0 - p_prior)
    if denominator == 0.0:
        return 0.0
    return (p_prior * L) / denominator


# ----------------------------------------------------------------------
# Hybrid layer – encoding actions as high‑dimensional points
# ----------------------------------------------------------------------


def encode_actions(actions: List[MathAction], k: int = 128) -> np.ndarray:
    """
    Encode a list of ``MathAction`` objects into a matrix of shape (N, k).

    Each row is a MinHash signature of the action identifier concatenated
    with the string representation of its expected value.
    """
    signatures = []
    for act in actions:
        tokens = [act.id, f"{act.expected_value:.6f}"]
        sig = signature(tokens, k=k)
        signatures.append(sig)
    # Convert unsigned 64‑bit ints to floats in [0,1) for geometric use
    arr = np.array(signatures, dtype=np.uint64).astype(np.float64)
    arr /= float(1 << 64)
    return arr  # shape (N, k)


# ----------------------------------------------------------------------
# Hybrid tree construction – geometry + Bayesian posterior
# ----------------------------------------------------------------------


def build_posterior_tree(
    points: np.ndarray,
    prior_fn=similarity,
    fp: float = 0.05,
    distance_scale: float = 10.0,
) -> Dict[Edge, float]:
    """
    Construct a fully connected graph where each edge weight is a Bayesian
    posterior that combines:

    * prior  – MinHash similarity of the underlying token sets,
    * likelihood – sigmoid of the negative Euclidean distance,
    * false‑positive rate – ``fp`` (small regulariser).

    The function returns a dictionary mapping ``(i, j)`` to the posterior
    weight.  Only the upper‑triangular part (i < j) is stored.
    """
    n = points.shape[0]
    tree: Dict[Edge, float] = {}
    # Pre‑compute signatures for the prior (need original integer signatures)
    # We reconstruct them from the float representation to keep the interface simple.
    # In practice one would keep the integer signatures alongside the float matrix.
    int_sigs = (points * (1 << 64)).astype(np.uint64).tolist()
    for i in range(n):
        for j in range(i + 1, n):
            # Prior similarity via MinHash signatures
            p_prior = prior_fn(int_sigs[i], int_sigs[j])

            # Euclidean distance in the continuous embedding space
            dist = np.linalg.norm(points[i] - points[j])
            # Scale distance to control sharpness of the sigmoid
            likelihood = sigmoid(np.array([-distance_scale * dist]))[0]

            post = bayesian_posterior(p_prior, likelihood, fp)
            tree[(i, j)] = post
    return tree


# ----------------------------------------------------------------------
# Hybrid rectified flow – interpolation with posterior weighting
# ----------------------------------------------------------------------


def rectified_flow_interpolate(
    src: np.ndarray,
    tgt: np.ndarray,
    steps: int = 10,
    posterior_weight: float = 1.0,
) -> List[np.ndarray]:
    """
    Generate a rectified flow from ``src`` to ``tgt`` in ``steps`` increments.

    The raw linear interpolation is first rectified (negative components set to 0)
    and then passed through a sigmoid.  Finally, the result is scaled by a
    ``posterior_weight`` that can be obtained from the tree constructed above.
    """
    if steps < 2:
        raise ValueError("steps must be >= 2")
    flow: List[np.ndarray] = []
    for t in np.linspace(0.0, 1.0, steps):
        interp = (1 - t) * src + t * tgt
        rectified = np.maximum(interp, 0.0)
        transformed = sigmoid(rectified)
        flow.append(posterior_weight * transformed)
    return flow


# ----------------------------------------------------------------------
# Hybrid scoring – combine similarity, posterior, and flow
# ----------------------------------------------------------------------


def hybrid_score(
    action_idx: int,
    flow_vec: np.ndarray,
    points: np.ndarray,
    tree: Dict[Edge, float],
) -> float:
    """
    Compute a hybrid score for a given action index.

    The score aggregates:
    * Cosine similarity between the action point and the current flow vector.
    * Posterior edge weights of all edges incident to the action node.
    The two components are multiplied to emphasise actions that are both
    geometrically aligned with the flow *and* have high Bayesian belief.
    """
    # Cosine similarity
    a = points[action_idx]
    dot = np.dot(a, flow_vec)
    norm_a = np.linalg.norm(a)
    norm_f = np.linalg.norm(flow_vec)
    if norm_a == 0.0 or norm_f == 0.0:
        cos_sim = 0.0
    else:
        cos_sim = dot / (norm_a * norm_f)

    # Sum of posterior weights of incident edges
    incident_sum = 0.0
    for (i, j), w in tree.items():
        if i == action_idx or j == action_idx:
            incident_sum += w

    return cos_sim * incident_sum


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # Create a small synthetic set of actions
    actions = [
        MathAction(id="alpha", expected_value=1.2),
        MathAction(id="beta", expected_value=2.5),
        MathAction(id="gamma", expected_value=0.7),
        MathAction(id="delta", expected_value=3.3),
    ]

    # Encode actions into high‑dimensional points
    points = encode_actions(actions, k=64)  # shape (4, 64)

    # Build the Bayesian posterior tree
    tree = build_posterior_tree(points, fp=0.07, distance_scale=15.0)

    # Pick source and target actions for a flow
    src_idx, tgt_idx = 0, 3
    src_vec = points[src_idx]
    tgt_vec = points[tgt_idx]

    # Retrieve a representative posterior weight for scaling (mean of incident edges)
    incident_weights = [
        w for (i, j), w in tree.items() if i == src_idx or j == src_idx
    ]
    scale = float(np.mean(incident_weights)) if incident_weights else 1.0

    # Generate rectified flow
    flow_steps = rectified_flow_interpolate(src_vec, tgt_vec, steps=12, posterior_weight=scale)

    # Compute hybrid scores for each intermediate flow step
    for step_idx, flow_vec in enumerate(flow_steps):
        scores = [
            hybrid_score(idx, flow_vec, points, tree) for idx in range(len(actions))
        ]
        best_action = actions[int(np.argmax(scores))]
        print(
            f"Step {step_idx:02d}: best action = {best_action.id}, "
            f"score = {max(scores):.4f}"
        )
    print("Smoke test completed successfully.")