# DARWIN HAMMER — match 3233, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_doomsd_m2255_s0.py (gen3)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s4.py (gen1)
# born: 2026-05-29T23:48:37Z

"""Hybrid NLMS–Caputo–Tree Algorithm

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – NLMS adaptive filtering with the *serpentina_morphology* metric.
* **Parent B** – Caputo fractional derivative based decay and a minimum‑cost tree scorer.

**Mathematical bridge**

The NLMS weight vector `w` is treated as a state that evolves over discrete time.
Instead of a simple instantaneous NLMS update, we weight past NLMS updates with a
Caputo‑type fractional kernel `fractional_decay(α,·)`.  This gives the new weight
vector a memory that decays algebraically, exactly as the edge‑weight decay in the
fractional tree model of Parent B.  The resulting weight vector is then used both
to (i) evaluate the serpentina morphology (recovery priority) and (ii) to score a
minimum‑cost tree whose edge weights are the fractional‑decayed contributions of
the historic NLMS updates.

Thus the hybrid system couples:
* NLMS error‑driven adaptation,
* Caputo fractional weighting of the adaptation history,
* Tree‑cost evaluation using the same fractional‑decayed weights,
* Morphology‑based priority derived from the current weights.

The three principal functions below demonstrate this integrated workflow.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Tuple

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))


def nlms_update(weights: np.ndarray, x: np.ndarray, target: float,
                mu: float = 0.5, eps: float = 1e-9) -> Tuple[np.ndarray, float]:
    """
    Standard Normalised Least‑Mean‑Squares (NLMS) weight update.

    Returns the updated weight vector and the instantaneous error.
    """
    y = predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    next_weights = weights + mu * error * x / power
    return next_weights, error


def serpentina_morphology(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Morphological metric from Parent A.
    Returns a scalar in [0,1] describing “flatness‑sphericity”.
    """
    lengths = np.abs(values)
    max_len = np.max(lengths)
    if max_len == 0.0:
        return 0.0
    flatness = np.dot(weights, lengths / max_len)
    sphericity = 1.0 - (3.0 * np.dot(weights, lengths ** 2)) / (
        4.0 * (np.dot(weights, lengths) ** 2) + 1e-12
    )
    return float((flatness + sphericity) / 2.0)


# ----------------------------------------------------------------------
# Parent‑B building blocks (fractional kernel and tree cost)
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])


def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of Γ(z) for z>0."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1.0 - z))
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2.0 * math.pi) * (t ** (z + 0.5)) * math.exp(-t) * x


def fractional_decay(alpha: float, t: int) -> float:
    """
    Caputo‑type decay kernel:  t^{α‑1} / Γ(α)
    """
    if t <= 0:
        return 0.0
    return (t ** (alpha - 1.0)) / gamma_lanczos(alpha)


def fractional_weighted_history(history: List[np.ndarray], alpha: float) -> np.ndarray:
    """
    Compute a weighted sum of past weight vectors using the fractional decay kernel.
    `history[0]` is the most recent weight vector, `history[-1]` the oldest.
    """
    if not history:
        raise ValueError("History must contain at least one weight vector.")
    L = len(history)
    kernels = np.array([fractional_decay(alpha, k + 1) for k in range(L)])
    kernels /= kernels.sum()  # normalise
    weighted = sum(kernels[i] * history[i] for i in range(L))
    return weighted


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fractional_tree_cost(nodes: List[int],
                         edges: List[Tuple[int, int, np.ndarray]],
                         root: int,
                         weight_history: List[np.ndarray],
                         alpha: float) -> float:
    """
    Minimum‑cost tree scoring where each edge weight is the fractional‑decayed
    projection of the historic NLMS weight vectors onto the edge feature vector.

    Parameters
    ----------
    nodes : list of node identifiers
    edges : list of (parent, child, feature_vector) tuples
    root  : identifier of the root node (must be in `nodes`)
    weight_history : list of past NLMS weight vectors (most recent first)
    alpha : fractional order (0<α≤1)

    Returns
    -------
    total cost (float)
    """
    if root not in nodes:
        raise ValueError("Root must be one of the nodes.")
    # Build adjacency for a directed tree
    adj = {n: [] for n in nodes}
    for p, c, feat in edges:
        adj[p].append((c, feat))

    # Pre‑compute the fractional‑weighted weight vector
    w_frac = fractional_weighted_history(weight_history, alpha)

    # Depth‑first traversal accumulating edge costs
    total = 0.0
    stack = [(root, 0.0)]  # (node, accumulated_cost)
    while stack:
        node, acc = stack.pop()
        total = max(total, acc)  # we keep the longest root‑to‑leaf cost
        for child, feat in adj.get(node, []):
            # Edge cost = |w_frac·feat|  (absolute projection)
            edge_cost = abs(float(np.dot(w_frac, feat)))
            stack.append((child, acc + edge_cost))
    return total


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_nlms_fractional_step(weights: np.ndarray,
                                x: np.ndarray,
                                target: float,
                                history: List[np.ndarray],
                                mu: float = 0.5,
                                eps: float = 1e-9,
                                alpha: float = 0.8) -> Tuple[np.ndarray, List[np.ndarray], float]:
    """
    Perform one NLMS adaptation step and immediately blend the new weight vector
    into the historic list using a Caputo fractional kernel.

    Returns
    -------
    new_weights : updated weight vector after NLMS step
    new_history : updated list of historic weight vectors (most recent first)
    error       : instantaneous NLMS error
    """
    # 1️⃣ NLMS instantaneous update
    new_weights, error = nlms_update(weights, x, target, mu, eps)

    # 2️⃣ Insert at front of history (most recent)
    new_history = [new_weights] + history

    # 3️⃣ Apply fractional weighting to keep a bounded history length
    max_len = 10  # hyper‑parameter: how many past steps we keep
    if len(new_history) > max_len:
        new_history = new_history[:max_len]

    # 4️⃣ Produce a fractional‑blended weight vector for downstream use
    blended = fractional_weighted_history(new_history, alpha)

    return blended, new_history, error


def hybrid_recovery_priority(date: Tuple[int, int, int],
                             values: np.ndarray,
                             weights: np.ndarray,
                             alpha: float = 0.7) -> float:
    """
    Compute a recovery priority that mixes the serpentina morphology (Parent A)
    with a fractional‑decayed tree cost (Parent B).

    The priority is a convex combination:
        priority = λ * morphology + (1‑λ) * (1 / (1 + tree_cost))

    where λ = α (fractional order) to keep a single tunable parameter.
    """
    # Morphology component
    morph = serpentina_morphology(values, weights)

    # Dummy tree for illustration – a small binary tree whose edge features are
    # derived from the weight vector itself.
    nodes = [0, 1, 2]
    # Feature vectors are simple slices of the weight vector (padded if needed)
    dim = len(weights)
    def slice_feat(i):
        start = (i * dim) // 3
        end = ((i + 1) * dim) // 3
        feat = np.zeros(dim)
        feat[start:end] = weights[start:end]
        return feat
    edges = [
        (0, 1, slice_feat(0)),
        (0, 2, slice_feat(1)),
    ]
    # Use a trivial history consisting of the current weights repeated
    history = [weights] * 3
    tree_c = fractional_tree_cost(nodes, edges, root=0, weight_history=history, alpha=alpha)

    # Convert cost to a bounded score (larger cost → smaller contribution)
    cost_score = 1.0 / (1.0 + tree_c)

    # Convex blend
    lam = max(0.0, min(1.0, alpha))
    priority = lam * morph + (1.0 - lam) * cost_score
    return float(priority)


def hybrid_process_sequence(initial_weights: np.ndarray,
                            inputs: List[np.ndarray],
                            targets: List[float],
                            alpha: float = 0.8,
                            mu: float = 0.5) -> Tuple[np.ndarray, List[float]]:
    """
    Run a full sequence of NLMS‑fractional updates, returning the final blended
    weight vector and a list of instantaneous errors.

    This demonstrates the end‑to‑end hybrid pipeline.
    """
    if len(inputs) != len(targets):
        raise ValueError("inputs and targets must have the same length.")
    history: List[np.ndarray] = [initial_weights]
    errors: List[float] = []
    w = initial_weights.copy()
    for x, t in zip(inputs, targets):
        w, history, err = hybrid_nlms_fractional_step(
            w, x, t, history, mu=mu, eps=1e-9, alpha=alpha
        )
        errors.append(err)
    return w, errors


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Initial NLMS weight vector (dimension 6)
    w0 = np.random.randn(6)

    # Synthetic input‑target sequence
    xs = [np.random.randn(6) for _ in range(5)]
    ts = [float(np.dot(w0, x) + 0.1 * random.random()) for x in xs]  # noisy targets

    # Run the hybrid sequence
    final_w, err_list = hybrid_process_sequence(w0, xs, ts, alpha=0.75, mu=0.6)

    # Compute a priority for a fictitious date and some value vector
    date = (2026, 5, 29)
    values = np.abs(np.random.randn(6))
    priority = hybrid_recovery_priority(date, values, final_w, alpha=0.75)

    # Print results (just to verify that no exception occurs)
    print("Final blended weights:", final_w)
    print("NLMS errors per step :", err_list)
    print("Recovery priority   :", priority)