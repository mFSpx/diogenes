# DARWIN HAMMER — match 2678, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s2.py (gen4)
# parent_b: hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py (gen1)
# born: 2026-05-29T23:43:35Z

"""Hybrid Tree‑Bayes‑Evasion Algorithm.

Parents
-------
- **hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s2.py**  
  Provides tree metric construction, deterministic edge costs and a
  path‑signature computation based on iterated integrals (level‑1 and
  level‑2).

- **hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py**  
  Supplies a continuous‑optimisation schedule (`evasion_delta`),
  statistical gating via a Hoeffding bound and a signal‑to‑noise
  confidence scalar.

Mathematical Bridge
-------------------
The bridge is the *probabilistic weight* assigned to each edge of the
tree. In the original minimum‑cost tree Bayes update these weights are
updated with a deterministic cost function. Here we reinterpret the
weight update as a Bayesian posterior where the *likelihood* is a
function of the **signal‑to‑noise gap** `Δ = signal - noise`.  The gap is
scaled by the **evasion schedule** `δ(t)` from the optimisation parent
and by a **Hoeffding epsilon** `ε` that quantifies statistical confidence
in the observed `signal/noise` counts.  The resulting posterior
probabilities are then used as multiplicative factors in the
iterated‑integral (path‑signature) computation, thereby coupling the
tree‑metric structure with the adaptive optimisation dynamics.

The hybrid algorithm therefore consists of three tightly coupled
operations:

1. `tree_metrics` – builds adjacency and Euclidean edge lengths.
2. `hybrid_bayes_update` – Bayesian edge‑weight update using
   `Δ`, `δ(t)` and `ε`.
3. `hybrid_path_signature` – computes level‑1 and level‑2 signatures
   weighted by the posterior edge probabilities.

These three functions together form a unified system that can be used
for adaptive routing, probabilistic graphical modelling, or any task
requiring a dynamic interplay between discrete tree structure and
continuous optimisation feedback.
"""

import math
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Sequence, Tuple, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Utilities from Parent A (tree handling)
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping ordered edge (a, b) → Euclidean length
    root_dist : dict mapping node → distance from root (sum of edge lengths)
    """
    adj: Dict[str, List[str]] = defaultdict(list)
    edge_len: Dict[Edge, float] = {}
    root_dist: Dict[str, float] = {root: 0.0}

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        d = length(nodes[a], nodes[b])
        edge_len[(a, b)] = d
        edge_len[(b, a)] = d

    # BFS to compute root distances
    visited = {root}
    frontier = [root]
    while frontier:
        cur = frontier.pop(0)
        for nb in adj[cur]:
            if nb not in visited:
                root_dist[nb] = root_dist[cur] + edge_len[(cur, nb)]
                visited.add(nb)
                frontier.append(nb)

    return adj, edge_len, root_dist


# ----------------------------------------------------------------------
# Utilities from Parent B (optimisation & statistics)
# ----------------------------------------------------------------------
def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    """Exponential decay schedule for evasion magnitude."""
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)


def clamp(x: Vector, lower: float, upper: float) -> List[float]:
    """Clamp each component of a vector to [lower, upper]."""
    return [min(upper, max(lower, xi)) for xi in x]


def shannon_entropy(data: Sequence[int]) -> float:
    """Return Shannon entropy in bits for a sequence of byte values."""
    if not data:
        return 0.0
    counts = np.bincount(np.array(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(data)
    return -np.sum(probs * np.log2(probs))


def hoeffding_epsilon(n: int, R: float = 1.0, delta: float = 0.05) -> float:
    """
    Hoeffding bound epsilon for a bounded random variable in [0, R].

    Parameters
    ----------
    n : int
        Number of independent observations.
    R : float, default 1.0
        Range of the random variable.
    delta : float, default 0.05
        Desired failure probability.

    Returns
    -------
    epsilon : float
        Upper bound on the deviation of the empirical mean from the true mean.
    """
    if n <= 0:
        return float('inf')
    return math.sqrt((R ** 2 * math.log(2 / delta)) / (2 * n))


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def hybrid_bayes_update(
    edges: List[Edge],
    prior: Dict[Edge, float],
    signal: Dict[Edge, float],
    noise: Dict[Edge, float],
    t: int,
    t_max: int,
    obs_count: int,
) -> Dict[Edge, float]:
    """
    Bayesian posterior update for edge weights.

    The likelihood for an edge is modelled as
        L(e) = exp( Δ(e) * δ(t) * (1 + ε) )
    where Δ(e) = signal[e] - noise[e] (confidence gap),
          δ(t)   = evasion schedule,
          ε      = Hoeffding epsilon based on the number of observations.

    The posterior is proportional to prior * L(e) and is normalised to
    sum to 1 over all edges.

    Parameters
    ----------
    edges : list of edges.
    prior : dict edge → prior probability (must sum to 1).
    signal, noise : dict edge → observed scalar values.
    t, t_max : current and maximal time step for the evasion schedule.
    obs_count : number of observations used to compute ε.

    Returns
    -------
    posterior : dict edge → updated probability.
    """
    delta_t = evasion_delta(t, t_max)
    epsilon = hoeffding_epsilon(obs_count)

    unnorm = {}
    for e in edges:
        Δ = signal.get(e, 0.0) - noise.get(e, 0.0)
        likelihood = math.exp(Δ * delta_t * (1.0 + epsilon))
        unnorm[e] = prior.get(e, 0.0) * likelihood

    total = sum(unnorm.values())
    if total == 0.0:
        # fallback to uniform distribution
        uniform = 1.0 / len(edges)
        return {e: uniform for e in edges}
    posterior = {e: v / total for e, v in unnorm.items()}
    return posterior


def hybrid_path_signature(
    nodes: Dict[str, Point],
    edges: List[Edge],
    posterior: Dict[Edge, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute weighted level‑1 and level‑2 path signatures for the tree.

    Level‑1 (first order) signature is the weighted sum of edge vectors.
    Level‑2 (second order) signature is the weighted sum of outer products
    of edge vectors, i.e. Σ w_e * (v_e ⊗ v_e).

    The edge vector v_e is defined as (x_b - x_a, y_b - y_a) for edge (a, b).

    Parameters
    ----------
    nodes : dict node → (x, y) coordinate.
    edges : list of directed edges (a, b).  The direction is arbitrary but
            must be consistent for the signature.
    posterior : dict edge → weight (probability) from `hybrid_bayes_update`.

    Returns
    -------
    sig1 : np.ndarray of shape (2,) – weighted first‑order signature.
    sig2 : np.ndarray of shape (4,) – flattened weighted second‑order signature.
    """
    sig1 = np.zeros(2)
    sig2 = np.zeros((2, 2))

    for a, b in edges:
        w = posterior.get((a, b), posterior.get((b, a), 0.0))
        if w == 0.0:
            continue
        vec = np.array([nodes[b][0] - nodes[a][0],
                        nodes[b][1] - nodes[a][1]], dtype=float)
        sig1 += w * vec
        sig2 += w * np.outer(vec, vec)

    return sig1, sig2.ravel()


def hybrid_step(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    prior: Dict[Edge, float],
    signal_obs: Dict[Edge, float],
    noise_obs: Dict[Edge, float],
    t: int,
    t_max: int,
    obs_count: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Perform a full hybrid iteration:
        1. Build tree metrics.
        2. Update edge posteriors with Bayesian/optimisation bridge.
        3. Compute weighted path signatures.

    Returns
    -------
    root_dist : dict node → distance from root (unchanged by optimisation).
    sig1, sig2 : weighted signatures from `hybrid_path_signature`.
    """
    # 1. Tree metrics (adjacency & Euclidean lengths)
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)

    # 2. Bayesian posterior update using signal/noise and evasion schedule
    posterior = hybrid_bayes_update(
        edges, prior, signal_obs, noise_obs, t, t_max, obs_count
    )

    # 3. Weighted path signature
    sig1, sig2 = hybrid_path_signature(nodes, edges, posterior)

    return root_dist, sig1, sig2


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny tree
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]

    # Uniform prior over directed edges (both orientations allowed)
    prior = {e: 1.0 / len(edges) for e in edges}

    # Synthetic observations
    random.seed(42)
    signal_obs = {e: random.uniform(0.5, 1.0) for e in edges}
    noise_obs = {e: random.uniform(0.0, 0.5) for e in edges}

    # Parameters for the hybrid schedule
    t = 3
    t_max = 10
    obs_count = 50

    # Run one hybrid iteration
    root_dist, sig1, sig2 = hybrid_step(
        nodes,
        edges,
        root="A",
        prior=prior,
        signal_obs=signal_obs,
        noise_obs=noise_obs,
        t=t,
        t_max=t_max,
        obs_count=obs_count,
    )

    # Simple sanity prints (no assertions, just ensure no crash)
    print("Root distances:", root_dist)
    print("Weighted level‑1 signature:", sig1)
    print("Weighted level‑2 signature (flattened):", sig2)