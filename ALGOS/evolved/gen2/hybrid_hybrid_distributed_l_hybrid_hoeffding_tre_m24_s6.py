# DARWIN HAMMER — match 24, survivor 6
# gen: 2
# parent_a: hybrid_distributed_leader_e_thanatosis_m65_s1.py (gen1)
# parent_b: hybrid_hoeffding_tree_tropical_maxplus_m18_s0.py (gen1)
# born: 2026-05-29T23:25:24Z

import numpy as np
import random
from collections.abc import Mapping, Hashable
import math

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]

# ----------------------------------------------------------------------
# Parent A – probabilistic primitives
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    if total_phases < 1 or current_phase < 1:
        raise ValueError('phases and phase must be positive')
    return min(1.0, 1.0 / (2 ** max(0, total_phases - current_phase)))

def acceptance_probability(delta_e: float, temperature: float) -> float:
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)

def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)

# ----------------------------------------------------------------------
# Parent B – Hoeffding bound and tropical algebra
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def t_add(x, y):
    return np.maximum(x, y)

def t_mul(x, y):
    return np.add(x, y)

def t_polyval(coeffs, x):
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
    epsilon = hoeffding_bound(r, delta, n)
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
    rng = random.Random(seed)
    if tropical_coeffs is None:
        tropical_coeffs = np.array([0.0, 1.0])  

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
            accept_prob = acceptance_probability(delta_e, temperature)
            if rng.random() < accept_prob:
                new_leaders.add(node)

        leaders.update(new_leaders)
        undecided.difference_update(new_leaders)

    return leaders

def evaluate_hybrid_tropical_network(
    x: np.ndarray,
    layers: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    h = np.asarray(x, dtype=float).ravel()
    for W, b in layers:
        W = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float)
        y = np.maximum(np.add(np.dot(W, h), b), 0)
        h = y
    return h

def improved_hybrid_leader_election_with_tree(
    graph: Graph,
    phases: int = 6,
    t0: float = 1.0,
    alpha: float = 0.95,
    seed: int | str | None = None,
    delta: float = 0.05,
    tropical_coeffs: np.ndarray | None = None,
    learning_rate: float = 0.1,
) -> set[Node]:
    rng = random.Random(seed)
    if tropical_coeffs is None:
        tropical_coeffs = np.array([0.0, 1.0])  

    undecided = set(graph)
    leaders: set[Node] = set()
    stats: dict[Node, tuple[float, float, float, int]] = {
        n: (rng.random(), rng.random() * 0.5, 1.0, 10) for n in graph
    }
    tropical_coefficients = np.copy(tropical_coeffs)
    losses = []

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
                best_gain, second_gain, r, delta, n_obs, tropical_coefficients
            )
            accept_prob = acceptance_probability(delta_e, temperature)
            if rng.random() < accept_prob:
                new_leaders.add(node)

            # Update tropical coefficients using gradient descent
            loss = np.abs(delta_e)
            losses.append(loss)
            gradient = np.array([loss * (-1) ** i for i in range(len(tropical_coefficients))])
            tropical_coefficients -= learning_rate * gradient

        leaders.update(new_leaders)
        undecided.difference_update(new_leaders)

    return leaders

def main():
    graph = {
        'A': {'B', 'C'},
        'B': {'A', 'D'},
        'C': {'A', 'D'},
        'D': {'B', 'C'}
    }
    leaders = improved_hybrid_leader_election_with_tree(graph)
    print(leaders)

if __name__ == "__main__":
    main()