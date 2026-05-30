# DARWIN HAMMER — match 24, survivor 5
# gen: 2
# parent_a: hybrid_distributed_leader_e_thanatosis_m65_s1.py (gen1)
# parent_b: hybrid_hoeffding_tree_tropical_maxplus_m18_s0.py (gen1)
# born: 2026-05-29T23:25:24Z

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
        h = t_add(t_mul(W, h), b)
    return h


def improved_hybrid_leader_election_with_tree(
    graph: Graph,
    phases: int = 6,
    t0: float = 1.0,
    alpha: float = 0.95,
    seed: int | str | None = None,
    delta: float = 0.05,
    tropical_coeffs: np.ndarray | None = None,
) -> set[Node]:
    """
    Improved version of hybrid_leader_election_with_tree.

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
    stats: dict[Node, tuple[float, float, float, int]] = {
        n: (rng.random(), rng.random() * 0.5, 1.0, 10) for n in graph
    }

    for phase in range(1, phases + 1):
        if not undecided:
            break

        p = broadcast_probability(phases, phase)
        broadcasts = {n for n in undecided if rng.random() < p}
        candidates = {n for n in broadcasts if not (graph.get(n, set()) & broadcasts)}

        temperature = cooling_temperature(phase, t0, alpha)
        new_leaders = set()

        for node in candidates:
            best_gain, second_gain, r, n_obs = stats[node]
            epsilon, delta_e = hybrid_split_score(
                best_gain, second_gain, r, delta, n_obs, tropical_coeffs
            )
            # Improved acceptance probability
            accept_prob = 1.0 / (1.0 + math.exp(-delta_e / temperature))
            if rng.random() < accept_prob:
                new_leaders.add(node)

        leaders.update(new_leaders)
        undecided.difference_update(new_leaders)

        # Update node stats
        for node in undecided:
            best_gain, second_gain, r, n_obs = stats[node]
            if rng.random() < 0.1:  # 10% chance of updating node stats
                stats[node] = (rng.random(), rng.random() * 0.5, 1.0, 10)

    return leaders