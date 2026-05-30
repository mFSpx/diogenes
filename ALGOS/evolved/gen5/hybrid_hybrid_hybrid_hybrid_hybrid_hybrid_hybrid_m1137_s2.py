# DARWIN HAMMER — match 1137, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m296_s0.py (gen4)
# born: 2026-05-29T23:33:00Z

"""Hybrid Minimum‑Cost Tree with Multivector‑Weighted Hoeffding Decision

Parents
-------
* **Parent A** – ``hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s1.py``  
  Provides geometric tree utilities (node coordinates, Euclidean edge lengths).

* **Parent B** – ``hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m296_s0.py``  
  Introduces a Clifford‑algebra ``Multivector`` representation and the
  Hoeffding bound used in incremental decision‑tree learning.

Mathematical Bridge
-------------------
The bridge consists of two steps:

1. **Multivector encoding of LSM vectors** – each node carries a lexical‑semantic‑metric
   (LSM) vector *ℓ*.  The vector is promoted to a grade‑1 multivector
   ``M(ℓ)`` whose basis blades are singletons ``{i}``.  The scalar part of the
   geometric product ``M(ℓ_u) * M(ℓ_v)`` is exactly the Euclidean dot‑product
   ``ℓ_u·ℓ_v``.

2. **Hoeffding‑scaled hybrid edge cost** – for an edge *(u,v)* with geometric
   length ``d_uv`` we compute

       w_uv = ½·(ℓ_u·ℓ_v)                # same as Parent A’s LSM weight
       ε    = HoeffdingBound(n, R, δ)    # statistical confidence from Parent B
       c_uv = (w_uv·d_uv)·(1+ε)           # hybrid cost inflated by Hoeffding ε

   The total tree cost ``C`` is the sum of ``c_uv`` over all edges.  A
   Gaussian Bayesian update (Parent A) yields the posterior estimate
   ``Ċ``.

The module below implements this fused pipeline with three public functions
``tree_metrics``, ``hybrid_tree_cost`` and ``bayesian_update`` plus a small
smoke test."""

import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – geometric tree utilities (trimmed to essentials)
# ----------------------------------------------------------------------
def euclidean_length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float]]:
    """
    Build adjacency list and edge‑length map for a rooted tree.

    Returns
    -------
    adjacency : dict mapping node → list of children (directed away from root)
    edge_len  : dict mapping (parent, child) → Euclidean length
    """
    adjacency: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}

    # orient edges away from the root using BFS
    visited = {root}
    frontier = [root]
    while frontier:
        cur = frontier.pop(0)
        for u, v in edges:
            if u == cur and v not in visited:
                child = v
            elif v == cur and u not in visited:
                child = u
            else:
                continue
            adjacency[cur].append(child)
            visited.add(child)
            frontier.append(child)
            edge_len[(cur, child)] = euclidean_length(nodes[cur], nodes[child])

    return adjacency, edge_len


# ----------------------------------------------------------------------
# Parent B – minimal Multivector class (grade‑1 only) and Hoeffding bound
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return sorted blade and sign after bubble‑sorting (handles duplicates)."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel duplicate basis vectors
                lst.pop(j)
                lst.pop(j)  # element that moved to position j
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    """Geometric product of two basis blades (grade‑1 only)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """
    Grade‑1 multivector (vector) in a Clifford algebra Cl(n,0).

    Internally stored as ``components`` mapping a frozenset of basis indices
    (always of size 1) to a float coefficient.
    """

    def __init__(self, components: Dict[int, float], n: int):
        # keep only non‑zero components and wrap index in frozenset
        self.components: Dict[frozenset, float] = {
            frozenset({i}): float(v) for i, v in components.items() if v != 0.0
        }
        self.n = int(n)

    def dot(self, other: "Multivector") -> float:
        """Scalar part of the geometric product = Euclidean dot product."""
        s = 0.0
        for blade, val in self.components.items():
            if blade in other.components:
                s += val * other.components[blade]
        return s

    def __mul__(self, other: "Multivector") -> Tuple[float, Any]:
        """
        Return (scalar_part, bivector_part) of the geometric product.
        For grade‑1 vectors the bivector part is ignored by the caller.
        """
        scalar = self.dot(other)
        # Bivector part (wedge) is not needed for the hybrid cost;
        # we return it as a placeholder ``None`` to keep the signature simple.
        return scalar, None

    @staticmethod
    def from_numpy(vec: np.ndarray) -> "Multivector":
        """Convenient factory from a 1‑D NumPy array."""
        comps = {int(i): float(v) for i, v in enumerate(vec)}
        return Multivector(comps, n=vec.size)


def hoeffding_bound(n: int, R: float = 1.0, delta: float = 0.05) -> float:
    """
    Hoeffding bound ε = sqrt( (R^2 * ln(1/δ)) / (2n) ).

    Parameters
    ----------
    n : int
        Number of observed samples.
    R : float, optional
        Range of the random variable (default 1.0).
    delta : float, optional
        Desired confidence (default 0.05 → 95% confidence).

    Returns
    -------
    epsilon : float
    """
    if n <= 0:
        return float('inf')
    return math.sqrt((R * R * math.log(1.0 / delta)) / (2.0 * n))


# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_edge_cost(
    length: float,
    mv_u: Multivector,
    mv_v: Multivector,
    n_samples: int,
    R: float = 1.0,
    delta: float = 0.05,
) -> float:
    """
    Compute the hybrid cost for a single edge.

    c_uv = (½·(ℓ_u·ℓ_v)·d_uv)·(1 + ε)

    where ε is the Hoeffding bound based on ``n_samples``.
    """
    w = 0.5 * mv_u.dot(mv_v)                # LSM weight (Parent A)
    eps = hoeffding_bound(n_samples, R, delta)
    return w * length * (1.0 + eps)


def hybrid_tree_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    lsm_vectors: Dict[str, np.ndarray],
    n_samples: int,
) -> float:
    """
    Compute the total hybrid cost of a rooted tree.

    Steps
    -----
    1. Build adjacency and Euclidean edge lengths (Parent A).
    2. Encode each node's LSM vector as a grade‑1 Multivector (Parent B).
    3. For every directed edge (parent → child) evaluate ``hybrid_edge_cost``.
    4. Sum all contributions.
    """
    adjacency, edge_len = tree_metrics(nodes, edges, root)

    # Encode LSM vectors as multivectors
    mv: Dict[str, Multivector] = {
        nid: Multivector.from_numpy(vec) for nid, vec in lsm_vectors.items()
    }

    total = 0.0
    for parent, children in adjacency.items():
        for child in children:
            length = edge_len[(parent, child)]
            total += hybrid_edge_cost(length, mv[parent], mv[child], n_samples)

    return total


def bayesian_update(
    prior_cost: float,
    observed: float,
    sigma_prior: float,
    sigma_obs: float,
) -> float:
    """
    Gaussian Bayesian update of the tree cost.

    Ċ = (σ_y²·C + σ_C²·y) / (σ_y² + σ_C²)
    """
    num = (sigma_obs ** 2) * prior_cost + (sigma_prior ** 2) * observed
    den = (sigma_obs ** 2) + (sigma_prior ** 2)
    return num / den if den != 0 else float('nan')


# ----------------------------------------------------------------------
# Public API (three demonstrative functions)
# ----------------------------------------------------------------------
def generate_random_lsm_vectors(
    node_ids: List[str],
    dim: int = 8,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Create reproducible random LSM vectors for each node."""
    rng = np.random.default_rng(seed)
    return {nid: rng.normal(loc=0.0, scale=1.0, size=dim) for nid in node_ids}


def compute_hybrid_cost_with_posterior(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    n_samples: int,
    observed_cost: float,
    sigma_prior: float = 1.0,
    sigma_obs: float = 0.5,
) -> Tuple[float, float]:
    """
    Convenience wrapper that:
    * generates LSM vectors,
    * evaluates the hybrid tree cost,
    * returns the Bayesian‑updated posterior cost.
    """
    lsm = generate_random_lsm_vectors(list(nodes.keys()))
    prior = hybrid_tree_cost(nodes, edges, root, lsm, n_samples)
    posterior = bayesian_update(prior, observed_cost, sigma_prior, sigma_obs)
    return prior, posterior


def gini_impurity(labels: List[int]) -> float:
    """
    Classic Gini impurity used in Hoeffding trees (Parent B).

    G = 1 - Σ (p_i)²
    """
    if not labels:
        return 0.0
    total = len(labels)
    counts = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    impurity = 1.0 - sum((c / total) ** 2 for c in counts.values())
    return impurity


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny tree
    node_coords = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
    }
    edge_list = [("A", "B"), ("A", "C"), ("B", "D")]
    root_node = "A"

    # Parameters for the hybrid evaluation
    sample_count = 250          # number of observations for Hoeffding bound
    observed_resource = 12.3    # simulated measured VRAM usage

    # Run the full pipeline
    prior_cost, posterior_cost = compute_hybrid_cost_with_posterior(
        nodes=node_coords,
        edges=edge_list,
        root=root_node,
        n_samples=sample_count,
        observed_cost=observed_resource,
    )

    print(f"Hybrid prior cost      : {prior_cost:.6f}")
    print(f"Observed resource usage: {observed_resource:.6f}")
    print(f"Posterior (Bayesian)   : {posterior_cost:.6f}")

    # Demonstrate Gini impurity on a dummy label set
    dummy_labels = [random.choice([0, 1]) for _ in range(20)]
    print(f"Gini impurity of dummy labels: {gini_impurity(dummy_labels):.6f}")