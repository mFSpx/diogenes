# DARWIN HAMMER — match 1131, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_liquid_time_c_hybrid_hybrid_hybrid_m259_s2.py (gen4)
# parent_b: hybrid_pheromone_hybrid_distributed_l_m41_s1.py (gen2)
# born: 2026-05-29T23:33:04Z

import numpy as np
import hashlib
import random
from collections.abc import Mapping, Hashable
from typing import List, Dict, Set, Tuple, Any

# -------------------- MinHash utilities --------------------

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: List[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length k for the given token list."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity based on MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# -------------------- Graph utilities --------------------

def broadcast_probability(phase: int, step: int) -> float:
    """
    Return p = 1 / 2**(phase‑step) clamped to [0, 1].
    When phase ≤ step the probability is 1 (full broadcast).
    """
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive")
    exponent = max(0, phase - step)
    return min(1.0, 1.0 / (2**exponent))


def maximal_independent_set(
    graph: Mapping[Hashable, Set[Hashable]],
    phases: int = 8,
    seed: Any = None,
) -> Set[Hashable]:
    """
    Approximate a maximal independent set using probabilistic local broadcasts.
    The probability schedule follows broadcast_probability(phase, phases).
    """
    rng = random.Random(seed)
    undecided: Set[Hashable] = set(graph)
    leaders: Set[Hashable] = set()
    blocked: Set[Hashable] = set()

    for phase in range(1, phases + 1):
        if not undecided:
            break
        p = broadcast_probability(phase, phases)
        broadcasts = {n for n in undecided if rng.random() < p}
        new_leaders = {
            n
            for n in broadcasts
            if not (graph.get(n, set()) & broadcasts)
        }
        leaders.update(new_leaders)
        blocked.update({nbr for n in new_leaders for nbr in graph.get(n, set())})
        undecided.difference_update(blocked)

    return leaders


def update_pheromones(
    leaders: Set[Hashable],
    pheromones: Dict[Hashable, float],
    decay: float = 0.9,
    boost: float = 0.3,
) -> Dict[Hashable, float]:
    """
    Decay all pheromone values and boost those belonging to current leaders.
    """
    new_pheromones: Dict[Hashable, float] = {}
    for node, val in pheromones.items():
        new_val = val * decay
        if node in leaders:
            new_val += boost
        new_pheromones[node] = new_val
    # Ensure nodes that become leaders for the first time receive a boost
    for node in leaders:
        if node not in new_pheromones:
            new_pheromones[node] = boost
    return new_pheromones


# -------------------- Liquid Time Constant (LTC) --------------------

def ltc_f(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    sim_factor: float,
) -> np.ndarray:
    """
    Core LTC dynamics.
    The similarity factor scales the external input I, allowing the
    temporal response to be modulated by how similar a node is to its neighbours.
    """
    return np.dot(x, W) + I * sim_factor + b


# -------------------- Hybrid algorithm --------------------

def hybrid_ltc_pheromone(
    node_id: Hashable,
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    node_tokens: List[str],
    graph: Mapping[Hashable, Set[Hashable]],
    token_map: Mapping[Hashable, List[str]],
    pheromones: Dict[Hashable, float],
    k: int = 128,
) -> Tuple[np.ndarray, Dict[Hashable, float]]:
    """
    Compute the LTC output for a single node while:
      * weighting the input by the average MinHash similarity to neighbours,
      * scaling the result by a pheromone level derived from a leader‑election process.
    Returns the updated output vector and the refreshed pheromone dictionary.
    """
    # 1️⃣ MinHash signatures
    sig_self = signature(node_tokens, k=k)
    neighbour_ids = graph.get(node_id, set())
    if neighbour_ids:
        neighbour_sigs = [signature(token_map[n], k=k) for n in neighbour_ids]
        sim_vals = [similarity(sig_self, s) for s in neighbour_sigs]
        sim_factor = sum(sim_vals) / len(sim_vals)  # average similarity ∈ [0,1]
    else:
        sim_factor = 0.0

    # 2️⃣ LTC dynamics with similarity‑aware input
    base = ltc_f(x, I, W, b, sim_factor)

    # 3️⃣ Leader election & pheromone update
    leaders = maximal_independent_set(graph, phases=8, seed=42)
    new_pheromones = update_pheromones(leaders, pheromones)

    pheromone_level = new_pheromones.get(node_id, 0.0)

    # 4️⃣ Fuse pheromone influence (multiplicative scaling)
    output = base * (1.0 + pheromone_level)

    return output, new_pheromones


# -------------------- Example usage --------------------

def _example():
    # Node‑wise data
    node_features = {
        1: np.array([1.0, 2.0, 3.0]),
        2: np.array([0.5, 1.5, 2.5]),
        3: np.array([2.0, 1.0, 0.0]),
    }
    I = np.array([4.0, 5.0, 6.0])
    W = np.array(
        [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ]
    )
    b = np.array([16.0, 17.0, 18.0])

    # Tokens per node
    token_map = {
        1: ["hello", "world"],
        2: ["foo", "bar", "baz"],
        3: ["lorem", "ipsum", "dolor"],
    }

    # Simple undirected graph
    graph = {
        1: {2, 3},
        2: {1, 3},
        3: {1, 2},
    }

    # Initialise pheromones to zero
    pheromones: Dict[Hashable, float] = {n: 0.0 for n in graph}

    # Run the hybrid update for each node
    for node_id, x in node_features.items():
        out, pheromones = hybrid_ltc_pheromone(
            node_id=node_id,
            x=x,
            I=I,
            W=W,
            b=b,
            node_tokens=token_map[node_id],
            graph=graph,
            token_map=token_map,
            pheromones=pheromones,
        )
        print(f"Node {node_id} output: {out}")

if __name__ == "__main__":
    _example()