# DARWIN HAMMER — match 24, survivor 4
# gen: 2
# parent_a: hybrid_distributed_leader_e_thanatosis_m65_s1.py (gen1)
# parent_b: hybrid_hoeffding_tree_tropical_maxplus_m18_s0.py (gen1)
# born: 2026-05-29T23:25:24Z

"""Hybrid Leader‑Election / Hoeffding‑Tree with Tropical Max‑Plus.

Parents:
- hybrid_distributed_leader_e_thanatosis_m65_s1 (probabilistic broadcast &
  simulated‑annealing acceptance)
- hybrid_hoeffding_tree_tropical_maxplus_m18_s0 (Hoeffding bound for
  splitting & tropical max‑plus algebra for split evaluation)

Mathematical bridge:
Both parents decide *whether* a structural change is kept.  The leader‑election
algorithm uses an acceptance probability  p = exp(−ΔE/T)  (simulated
annealing).  The Hoeffding‑tree algorithm decides to split a node when the
observed gain gap exceeds a Hoeffding bound ε.  We treat the Hoeffding bound
as a *temperature‑like* quantity: a larger ε makes acceptance harder, analogous
to a low temperature.  Conversely, the tropical max‑plus evaluation supplies a
scalar “energy” E for each candidate split (higher tropical gain = lower
energy).  The hybrid therefore:

1. Computes a Hoeffding bound ε for each candidate split.
2. Evaluates the split’s tropical gain G (max‑plus polynomial).
3. Defines ΔE = ε – G (the smaller the bound and the larger the gain, the more
   favorable the split).
4. Uses a cooling schedule T_k and the simulated‑annealing acceptance
   probability to decide whether to keep the split and promote the split’s root
   to a leader.

The result is a decision‑tree‑like structure where each accepted split also
acts as a distributed leader in the underlying graph."""

from __future__ import annotations
import random
import math
import sys
import pathlib
from collections.abc import Mapping, Hashable
import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]

# ----------------------------------------------------------------------
# Parent A – probabilistic primitives
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    """Probability that a node broadcasts in the current phase."""
    if total_phases < 1 or current_phase < 1:
        raise ValueError('phases and phase must be positive')
    return min(1.0, 1.0 / (2 ** max(0, total_phases - current_phase)))


def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Simulated‑annealing acceptance probability."""
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Geometric cooling schedule."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)

# ----------------------------------------------------------------------
# Parent B – Hoeffding bound and tropical algebra
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound ε = sqrt( (r^2 * ln(1/δ)) / (2n) )."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def t_add(x, y):
    """Tropical addition (max)."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication (ordinary addition)."""
    return np.add(x, y)


def t_polyval(coeffs, x):
    """
    Evaluate a tropical polynomial:
        P(x) = max_i ( coeff_i + i * x )
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(len(coeffs), dtype=float).reshape((-1,) + (1,) * x.ndim)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents * x
    return np.max(terms, axis=0)

# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def hybrid_split_score(best_gain: float, second_best_gain: float,
                       r: float, delta: float, n: int,
                       tropical_coeffs: np.ndarray) -> tuple[float, float]:
    """
    Compute the hybrid “energy” for a candidate split.

    Returns
    -------
    epsilon : Hoeffding bound ε
    delta_e : ΔE = ε - G, where G is the tropical gain (higher gain → lower energy)
    """
    epsilon = hoeffding_bound(r, delta, n)
    # Tropical gain G is evaluated at the current best_gain.
    # The polynomial encodes how gain translates to a tropical “energy”.
    G = float(t_polyval(tropical_coeffs, best_gain))
    delta_e = epsilon - G
    return epsilon, delta_e


def hybrid_leader_election_with_tree(
    graph: Graph,
    phases: int = 6,
    t0: float = 1.0,
    alpha: float = 0.95,
    seed: int | str | None = None,
    delta: float = 0.05,
    tropical_coeffs: np.ndarray | None = None,
) -> set[Node]:
    """
    Perform a distributed leader election where each elected leader is the
    root of a Hoeffding‑tree split that has passed a tropical‑gain‑augmented
    acceptance test.

    Parameters
    ----------
    graph : adjacency mapping of the network.
    phases : number of broadcast phases.
    t0, alpha : cooling schedule parameters.
    seed : random seed.
    delta : confidence parameter for Hoeffding bound.
    tropical_coeffs : coefficients of the tropical polynomial used to map
                      split gain to tropical “energy”.  If None, a default
                      linear polynomial is used.

    Returns
    -------
    set of nodes that became leaders.
    """
    rng = random.Random(seed)
    if tropical_coeffs is None:
        tropical_coeffs = np.array([0.0, 1.0])  # P(x) = max(0, x)

    undecided = set(graph)
    leaders: set[Node] = set()
    # Statistics placeholder: for each node we store a dummy tuple
    # (best_gain, second_best_gain, r, n).  In a real system these would be
    # collected from streaming data.
    stats: dict[Node, tuple[float, float, float, int]] = {
        n: (rng.random(), rng.random() * 0.5, 1.0, 10) for n in graph
    }

    for phase in range(1, phases + 1):
        if not undecided:
            break

        # ---- Broadcast step (Parent A) ----
        p = broadcast_probability(phases, phase)
        broadcasts = {n for n in undecided if rng.random() < p}
        # A node becomes a *candidate* leader if none of its neighbours also broadcast.
        candidates = {n for n in broadcasts if not (graph.get(n, set()) & broadcasts)}

        # ---- Hybrid acceptance (Bridge) ----
        temperature = cooling_temperature(phase, t0, alpha)
        new_leaders = set()

        for node in candidates:
            best_gain, second_gain, r, n_obs = stats[node]
            epsilon, delta_e = hybrid_split_score(
                best_gain, second_gain, r, delta, n_obs, tropical_coeffs
            )
            accept_prob = acceptance_probability(delta_e, temperature)
            if rng.random() < accept_prob:
                new_leaders.add(node)

        # Update sets
        leaders.update(new_leaders)
        undecided.difference_update(new_leaders)

    return leaders


def evaluate_hybrid_tropical_network(
    x: np.ndarray,
    layers: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    Evaluate a feed‑forward network where each layer uses tropical
    addition (max) and tropical multiplication (addition).  This function
    demonstrates the tropical algebra side of the hybrid.

    Parameters
    ----------
    x : input vector.
    layers : list of (W, b) where W is weight matrix and b is bias vector.

    Returns
    -------
    Output vector after tropical propagation.
    """
    h = np.asarray(x, dtype=float).ravel()
    for W, b in layers:
        W = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float)
        # Tropical affine transformation: y_i = max_j (W_ij + h_j) + b_i
        y = t_add(np.max(t_mul(W, h[:, None]), axis=0), b)
        h = y
    return h


def hybrid_tree_grow(
    data: np.ndarray,
    labels: np.ndarray,
    max_depth: int = 3,
    delta: float = 0.05,
    tropical_coeffs: np.ndarray | None = None,
    seed: int | str | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Grow a shallow Hoeffding‑tree where each split decision is filtered
    through the simulated‑annealing acceptance mechanism.  The returned
    structure is a list of tropical layers (W, b) that can be evaluated with
    ``evaluate_hybrid_tropical_network``.

    This function showcases the integration of:
    * Hoeffding bound (Parent B)
    * Tropical max‑plus evaluation (Parent B)
    * Acceptance probability & cooling (Parent A)
    """
    rng = random.Random(seed)
    if tropical_coeffs is None:
        tropical_coeffs = np.array([0.0, 1.0])

    n_features = data.shape[1]
    layers: list[tuple[np.ndarray, np.ndarray]] = []

    # Simple recursive split (not a full streaming implementation)
    def recursive_split(X, y, depth):
        if depth >= max_depth or len(np.unique(y)) == 1:
            # Leaf: produce a constant tropical layer
            leaf_val = np.mean(y)
            W = np.zeros((1, n_features))
            b = np.full((1,), leaf_val)
            layers.append((W, b))
            return

        # Compute a naive gain for each feature (variance reduction)
        gains = []
        thresholds = []
        for f in range(n_features):
            thresh = np.median(X[:, f])
            left = y[X[:, f] <= thresh]
            right = y[X[:, f] > thresh]
            if len(left) == 0 or len(right) == 0:
                gains.append(-np.inf)
                thresholds.append(thresh)
                continue
            var_before = np.var(y)
            var_after = (len(left) * np.var(left) + len(right) * np.var(right)) / len(y)
            gain = var_before - var_after
            gains.append(gain)
            thresholds.append(thresh)

        best_idx = int(np.argmax(gains))
        best_gain = gains[best_idx]
        second_gain = float(np.partition(gains, -2)[-2]) if len(gains) > 1 else 0.0

        # Hoeffding parameters (using range r = max_gain - min_gain)
        r = max(gains) - min(gains) if gains else 1.0
        n_obs = len(y)

        epsilon, delta_e = hybrid_split_score(
            best_gain, second_gain, r, delta, n_obs, tropical_coeffs
        )
        temperature = cooling_temperature(depth + 1)
        accept = rng.random() < acceptance_probability(delta_e, temperature)

        if accept:
            # Build a tropical linear layer that approximates the split:
            # output = max( w_f * x_f + b_left , w_f * x_f + b_right )
            f = best_idx
            thresh = thresholds[best_idx]
            w = np.zeros((2, n_features))
            w[0, f] = 0.0          # left branch weight
            w[1, f] = 0.0          # right branch weight
            b = np.array([ -thresh, thresh])  # tropical bias encodes the threshold
            layers.append((w, b))

            left_mask = X[:, f] <= thresh
            recursive_split(X[left_mask], y[left_mask], depth + 1)
            recursive_split(X[~left_mask], y[~left_mask], depth + 1)
        else:
            # Reject split → leaf
            leaf_val = np.mean(y)
            W = np.zeros((1, n_features))
            b = np.full((1,), leaf_val)
            layers.append((W, b))

    recursive_split(data, labels, 0)
    return layers

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple graph for leader election
    sample_graph: dict[int, set[int]] = {
        0: {1, 2},
        1: {0, 3},
        2: {0},
        3: {1},
        4: set(),
    }

    leaders = hybrid_leader_election_with_tree(
        sample_graph,
        phases=5,
        t0=2.0,
        alpha=0.9,
        seed=42,
        delta=0.05,
        tropical_coeffs=np.array([0.0, 0.5, 1.0]),
    )
    print("Leaders selected:", leaders)

    # Tiny dataset for hybrid tree
    X = np.array([[0.1], [0.4], [0.6], [0.9]])
    y = np.array([0, 0, 1, 1])

    layers = hybrid_tree_grow(
        X,
        y,
        max_depth=2,
        delta=0.05,
        tropical_coeffs=np.array([0.0, 1.0]),
        seed=123,
    )
    print("\nTropical layers (W, b):")
    for i, (W, b) in enumerate(layers):
        print(f"Layer {i}: W=\n{W}\nb={b}")

    # Evaluate the constructed tropical network on a new point
    test_point = np.array([0.55])
    out = evaluate_hybrid_tropical_network(test_point, layers)
    print("\nNetwork output for", test_point, ":", out)