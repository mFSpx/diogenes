# DARWIN HAMMER — match 4759, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2151_s3.py (gen6)
# parent_b: hybrid_state_space_duality_hybrid_hybrid_liquid_m1992_s0.py (gen3)
# born: 2026-05-29T23:57:57Z

"""Hybrid Temperature-Dependent State Space with Tree‑Guided Diffusion (HTDS‑TGD)

Parents
-------
* **Algorithm A** – Implements a Schoolfield temperature‐dependent developmental rate
  and a tree‑metric utility. The core operator is `temperature_dependent_state_transition`,
  scaling a state matrix by a biologically‑motivated rate.

* **Algorithm B** – Provides a hybrid state‑space duality (SSD) combined with
  Liquid‑Time‑Constant Diffusion Forcing (LTC‑DF). The core operators are
  `signature` (MinHash), a similarity‑driven diffusion timestep `t_i`,
  and a noisy input construction `diffuse_input`.

Mathematical Bridge
-------------------
For each node of a phylogenetic tree we treat the hidden state of a
state‑space model as a temperature‑scaled system:


A_t   = ρ(T_node) · A_0                     # temperature‑dependent transition
h_{t} = A_t @ h_{t-1} + B @ x_noisy
y_t   = C @ h_t


The diffusion timestep `t_i` is derived from the MinHash similarity `s`
between the node’s current token signature and the accumulated signature
so far:


t_i = round((1 - s) * T_max)
x_noisy = √ᾱ[t_i]·x + √(1-ᾱ[t_i])·ε


Tree distances modulate the similarity weighting, allowing spatially
structured diffusion. This module fuses the temperature scaling from
Algorithm A with the SSD + LTC‑DF loop of Algorithm B, yielding a single
unified hybrid system.

"""

import math
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Algorithm A – Temperature & Tree utilities
# ----------------------------------------------------------------------


class SchoolfieldParams:
    """Parameter container for the Schoolfield developmental rate."""
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol⁻¹ K⁻¹


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield temperature‑dependent developmental rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)


def temperature_dependent_state_transition(A: np.ndarray, temp_k: float,
                                            params: SchoolfieldParams = SchoolfieldParams()) -> np.ndarray:
    """Scale a state‑transition matrix by the developmental rate."""
    rate = developmental_rate(temp_k, params)
    return rate * A


def tree_metrics(nodes: Dict[str, Tuple[float, float]],
                 edges: List[Tuple[str, str]],
                 root: str) -> Tuple[Dict[str, List[str]],
                                    Dict[Tuple[str, str], float],
                                    Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
        edge_len[(b, a)] = edge_len[(a, b)]

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                stack.append(nxt)
    return adj, edge_len, dist


# ----------------------------------------------------------------------
# Algorithm B – MinHash & Diffusion utilities
# ----------------------------------------------------------------------


MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: List[str], k: int = 128) -> List[int]:
    """MinHash signature of a token list."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(seed, t) for t in toks) for seed in range(k)]


def minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Fraction of equal components between two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must be of equal length")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def _alpha_bar_schedule(T: int) -> List[float]:
    """
    Simple cosine‑based cumulative product schedule for diffusion.
    ᾱ[t] ∈ (0,1], decreasing with t.
    """
    return [math.cos((t / T) * math.pi / 2) ** 2 for t in range(T + 1)]


def diffuse_input(x: np.ndarray, t_i: int, alpha_bar: List[float]) -> np.ndarray:
    """
    Apply LTC‑DF diffusion to input vector `x` at timestep `t_i`.

    Parameters
    ----------
    x : np.ndarray
        Original input vector.
    t_i : int
        Diffusion timestep (0 ≤ t_i ≤ T).
    alpha_bar : list[float]
        Pre‑computed schedule ᾱ[t].

    Returns
    -------
    np.ndarray
        Noisy version of `x`.
    """
    if t_i < 0 or t_i >= len(alpha_bar):
        raise IndexError("t_i out of schedule bounds")
    alpha = alpha_bar[t_i]
    eps = np.random.normal(size=x.shape)
    return math.sqrt(alpha) * x + math.sqrt(1.0 - alpha) * eps


# ----------------------------------------------------------------------
# Hybrid Core – Temperature‑scaled State Space + Tree‑Guided Diffusion
# ----------------------------------------------------------------------


def hybrid_state_update(node_id: str,
                        h_prev: np.ndarray,
                        x: np.ndarray,
                        A0: np.ndarray,
                        B: np.ndarray,
                        C: np.ndarray,
                        temp_k: float,
                        token_pool: List[str],
                        prev_sig: List[int],
                        T_max: int,
                        alpha_bar: List[float]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Perform one hybrid update for a given tree node.

    Steps
    -----
    1. Scale the base transition matrix `A0` by the temperature‑dependent rate.
    2. Compute a MinHash signature of the node's token pool.
    3. Derive similarity `s` to the previous accumulated signature.
    4. Map `s` → diffusion timestep `t_i`.
    5. Diffuse the raw input `x` → `x_noisy`.
    6. State‑space update: h_new = A_t @ h_prev + B @ x_noisy
    7. Output projection: y = C @ h_new
    8. Return new hidden state, output, and current signature (to be accumulated).

    Returns
    -------
    h_new : np.ndarray
        Updated hidden state.
    y : np.ndarray
        Output vector.
    cur_sig : list[int]
        Current MinHash signature for accumulation.
    """
    # 1. Temperature scaling
    A_t = temperature_dependent_state_transition(A0, temp_k)

    # 2. Current signature
    cur_sig = signature(token_pool)

    # 3. Similarity to previous signature (if none, treat as 0 similarity)
    s = minhash_similarity(prev_sig, cur_sig) if prev_sig else 0.0

    # 4. Diffusion timestep
    t_i = int(round((1.0 - s) * T_max))

    # 5. Diffuse input
    x_noisy = diffuse_input(x, t_i, alpha_bar)

    # 6. State update
    h_new = A_t @ h_prev + B @ x_noisy

    # 7. Output projection
    y = C @ h_new

    return h_new, y, cur_sig


def run_hybrid_simulation(nodes: Dict[str, Tuple[float, float]],
                          edges: List[Tuple[str, str]],
                          root: str,
                          temperatures: Dict[str, float],
                          token_map: Dict[str, List[str]],
                          dim: int = 8,
                          steps_per_node: int = 3) -> Dict[str, np.ndarray]:
    """
    Simulate the hybrid system on a tree.

    Parameters
    ----------
    nodes, edges, root : tree definition.
    temperatures : mapping node → temperature in Kelvin.
    token_map : mapping node → list of tokens (strings) used for MinHash.
    dim : dimensionality of hidden state / input vectors.
    steps_per_node : how many internal hybrid steps to run at each node.

    Returns
    -------
    outputs : dict node → final output vector y after its local steps.
    """
    # Build adjacency (unused directly but kept for possible extensions)
    adj, _, _ = tree_metrics(nodes, edges, root)

    # Initialise linear operators (simple identity / scaled random matrices)
    rng = np.random.default_rng(42)
    A0 = rng.normal(size=(dim, dim))
    B = rng.normal(size=(dim, dim))
    C = rng.normal(size=(dim, dim))

    # Diffusion schedule
    T_max = 10
    alpha_bar = _alpha_bar_schedule(T_max)

    # Containers
    hidden_states: Dict[str, np.ndarray] = {}
    outputs: Dict[str, np.ndarray] = {}
    accumulated_sig: List[int] = []  # starts empty

    # Depth‑first traversal from root
    stack = [(root, None)]  # (node, parent)
    while stack:
        node, parent = stack.pop()
        # Initialise hidden state (zero vector) if not yet set
        h = hidden_states.get(node, np.zeros(dim))

        # Run a few hybrid steps using node‑specific temperature and tokens
        for _ in range(steps_per_node):
            x_raw = rng.normal(size=dim)  # mock external input
            temp_k = temperatures.get(node, 298.15)
            tokens = token_map.get(node, [])
            h, y, cur_sig = hybrid_state_update(
                node_id=node,
                h_prev=h,
                x=x_raw,
                A0=A0,
                B=B,
                C=C,
                temp_k=temp_k,
                token_pool=tokens,
                prev_sig=accumulated_sig,
                T_max=T_max,
                alpha_bar=alpha_bar,
            )
            # Accumulate signature for the next step (simple OR‑like accumulation)
            accumulated_sig = [min(a, b) for a, b in zip(accumulated_sig or cur_sig, cur_sig)]

        hidden_states[node] = h
        outputs[node] = y

        # Push children (avoid revisiting parent)
        for child in adj[node]:
            if child != parent:
                stack.append((child, node))

    return outputs


def compute_tree_distance_matrix(nodes: Dict[str, Tuple[float, float]],
                                 edges: List[Tuple[str, str]],
                                 root: str) -> np.ndarray:
    """
    Produce a dense matrix `D` where D[i, j] is the root‑to‑node distance
    between nodes i and j (order follows `list(nodes.keys())`).

    This function demonstrates the reuse of `tree_metrics` inside the hybrid
    context.
    """
    _, _, dist = tree_metrics(nodes, edges, root)
    order = list(nodes.keys())
    n = len(order)
    D = np.zeros((n, n))
    for i, ni in enumerate(order):
        for j, nj in enumerate(order):
            D[i, j] = abs(dist[ni] - dist[nj])
    return D


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny tree
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
    }
    edges = [("A", "B"), ("A", "C")]
    root = "A"

    # Temperatures (Kelvin)
    temps = {
        "A": 295.0,
        "B": 300.0,
        "C": 285.0,
    }

    # Tokens per node (simple words)
    token_map = {
        "A": ["alpha", "beta", "gamma"],
        "B": ["beta", "delta"],
        "C": ["epsilon", "zeta", "alpha"],
    }

    # Run the hybrid simulation
    out = run_hybrid_simulation(nodes, edges, root, temps, token_map, dim=6, steps_per_node=2)

    # Print results
    for node, vec in out.items():
        print(f"Node {node} final output (first 3 entries): {vec[:3]}")

    # Demonstrate distance matrix utility
    D = compute_tree_distance_matrix(nodes, edges, root)
    print("Root‑to‑node distance matrix:\n", D)