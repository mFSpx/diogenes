# DARWIN HAMMER — match 3015, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_physarum_netw_hybrid_infotaxis_min_m967_s0.py (gen5)
# parent_b: hybrid_hoeffding_tree_gini_coefficient_m13_s0.py (gen1)
# born: 2026-05-29T23:47:14Z

"""Hybrid Physarum‑Infotaxis – Hoeffding‑Gini Fusion

Parents:
- hybrid_hybrid_physarum_netw_hybrid_infotaxis_min_m967_s0.py
- hybrid_hoeffding_tree_gini_coefficient_m13_s0.py

Mathematical bridge:
The Physarum‑based conductance update  
    Δg_uv = (Φ_uv – Δ_uv) · w_uv
is enriched with two statistical scalars taken from the Hoeffding‑Gini side:

* **Gini coefficient (G)** of the whole conductance vector measures inequality of resource
  allocation across the network.  A high G indicates a few dominant edges.
* **Hoeffding bound (ε)** supplies a confidence interval for the observed flux change,
  using the range *r* of flux values, confidence *δ* and number of observations *n*.

The edge‑wise weight *w_uv* is defined as  

    w_uv = (1 + H_sig) / (1 + G + ε)

where *H_sig* is the Shannon entropy of the MinHash signature attached to the edge.
Thus the update simultaneously respects local information content (entropy), global
resource inequality (Gini) and statistical reliability (Hoeffding).  The same
scalar combination is reused to decide whether a split (topological refinement) is
warranted via a hybrid bound `G + ε`.

The module implements:
1. MinHash signature generation and its entropy.
2. Gini coefficient of conductances.
3. Hoeffding bound.
4. Hybrid conductance update that fuses all three ingredients.
5. Split decision based on the hybrid bound.

All components are pure NumPy / std‑lib and runnable as a self‑contained script.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple, Set

import numpy as np

# ----------------------------------------------------------------------
# MinHash utilities (from Parent A)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash for a (seed, token) pair."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """
    Compute a MinHash signature of length *k* for the given token collection.
    Empty token sets yield a signature of maximal hash values.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    token_set: Set[str] = {t for t in tokens if t}
    if not token_set:
        return [(1 << 64) - 1] * k

    signature: List[int] = [(1 << 64) - 1] * k
    for i in range(k):
        min_val = (1 << 64) - 1
        for token in token_set:
            h = _hash(i, token)
            if h < min_val:
                min_val = h
        signature[i] = min_val
    return signature


def shannon_entropy(signature: List[int]) -> float:
    """Shannon entropy (base‑2) of a MinHash signature interpreted as a discrete distribution."""
    if not signature:
        return 0.0
    # Convert hash values to frequencies by counting occurrences
    counts: Dict[int, int] = {}
    for v in signature:
        counts[v] = counts.get(v, 0) + 1
    total = len(signature)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / total
        entropy -= p * math.log2(p)
    return entropy


# ----------------------------------------------------------------------
# Gini coefficient (from Parent B)
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    """Gini inequality coefficient."""
    xs = sorted((float(x) for x in values))
    if not xs or sum(xs) == 0.0:
        return 0.0
    # Remove leading negatives as per original implementation
    if xs[0] < 0.0:
        for i, x in enumerate(xs):
            if x >= 0.0:
                xs = xs[i:]
                break
        if not xs or sum(xs) == 0.0:
            return 0.0
    n = len(xs)
    numerator = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return numerator / (n * sum(xs))


# ----------------------------------------------------------------------
# Hoeffding bound (from Parent B)
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a bounded random variable."""
    if r <= 0.0 or not (0.0 < delta < 1.0) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def hybrid_bound(values: Iterable[float], r: float, delta: float, n: int) -> float:
    """Hybrid bound = Gini + Hoeffding."""
    return gini_coefficient(values) + hoeffding_bound(r, delta, n)


# ----------------------------------------------------------------------
# Core data structures (Physarum side)
# ----------------------------------------------------------------------
@dataclass
class Edge:
    """Represents a directed edge (u → v) in the Physarum network."""
    u: int
    v: int
    conductance: float
    restriction: np.ndarray  # shape (d, d) – arbitrary symmetric matrix
    signature: List[int]     # MinHash signature
    flux: float = 0.0        # observed flux Φ_uv
    discrepancy: float = 0.0  # Δ_uv term from physarum dynamics

    def entropy(self) -> float:
        return shannon_entropy(self.signature)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_global_gini(edges: List[Edge]) -> float:
    """Gini coefficient of the conductance distribution across all edges."""
    return gini_coefficient([e.conductance for e in edges])


def compute_hoeffding_eps(fluxes: List[float], delta: float = 0.05) -> float:
    """
    Compute a Hoeffding bound ε using the observed flux range.
    r = max(flux) - min(flux)
    n = number of observations (len(fluxes))
    """
    if not fluxes:
        return 0.0
    r = max(fluxes) - min(fluxes)
    n = len(fluxes)
    return hoeffding_bound(r, delta, n)


def hybrid_conductance_update(edge: Edge, global_gini: float, eps: float) -> None:
    """
    Perform a single hybrid update of the edge's conductance.

    Δg = (Φ - Δ) * w
    w = (1 + H_sig) / (1 + G + ε)

    The function mutates ``edge.conductance`` in place.
    """
    H = edge.entropy()
    weight = (1.0 + H) / (1.0 + global_gini + eps)
    delta_g = (edge.flux - edge.discrepancy) * weight
    edge.conductance = max(edge.conductance + delta_g, 0.0)  # conductance stays non‑negative


def should_split_edge(edge: Edge, all_edges: List[Edge],
                     r: float, delta: float, n: int,
                     best_gain: float, second_best_gain: float) -> bool:
    """
    Hybrid split decision.
    - Compute hybrid bound B = G + ε.
    - Split if best_gain - second_best_gain > B.
    """
    values = [e.conductance for e in all_edges]
    B = hybrid_bound(values, r, delta, n)
    return (best_gain - second_best_gain) > B


# ----------------------------------------------------------------------
# Demonstration utilities
# ----------------------------------------------------------------------
def random_restriction_matrix(dim: int = 3) -> np.ndarray:
    """Generate a random symmetric positive‑definite matrix for the restriction tensor."""
    A = np.random.randn(dim, dim)
    return np.dot(A, A.T) + np.eye(dim) * 0.1  # ensure PD


def simulate_network(num_edges: int = 5) -> List[Edge]:
    """Create a toy Physarum network with random initialisation."""
    edges: List[Edge] = []
    for i in range(num_edges):
        u, v = i, (i + 1) % num_edges
        conductance = random.uniform(0.1, 1.0)
        restriction = random_restriction_matrix()
        # Tokens are just dummy words; in real use they would be sensory cues.
        tokens = [f"token{j}" for j in random.sample(range(20), k=5)]
        signature = minhash_signature(tokens, k=64)
        flux = random.uniform(0.0, 2.0)
        discrepancy = random.uniform(0.0, 0.5)
        edges.append(Edge(u, v, conductance, restriction, signature, flux, discrepancy))
    return edges


def run_hybrid_iteration(edges: List[Edge]) -> None:
    """One iteration of the hybrid algorithm over the whole edge set."""
    # Global statistics
    G = compute_global_gini(edges)
    eps = compute_hoeffding_eps([e.flux for e in edges])

    # Update each edge
    for e in edges:
        hybrid_conductance_update(e, G, eps)


if __name__ == "__main__":
    # Smoke test: initialise a small network, run a few hybrid iterations,
    # and print final conductances.
    random.seed(42)
    np.random.seed(42)

    net = simulate_network(num_edges=7)

    print("Initial conductances:")
    print([round(e.conductance, 4) for e in net])

    for it in range(3):
        run_hybrid_iteration(net)

    print("\nConductances after 3 hybrid iterations:")
    print([round(e.conductance, 4) for e in net])

    # Demonstrate split decision on the first edge
    edge0 = net[0]
    best_gain = random.uniform(0.2, 1.0)
    second_best_gain = random.uniform(0.0, best_gain * 0.8)
    split = should_split_edge(
        edge0,
        net,
        r=1.5,               # example flux range
        delta=0.05,
        n=10,
        best_gain=best_gain,
        second_best_gain=second_best_gain,
    )
    print(f"\nShould split edge {edge0.u}->{edge0.v}? {'Yes' if split else 'No'}")
    print(f"Best gain={best_gain:.3f}, second best={second_best_gain:.3f}")