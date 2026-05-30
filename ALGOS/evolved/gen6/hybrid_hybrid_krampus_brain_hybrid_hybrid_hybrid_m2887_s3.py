# DARWIN HAMMER — match 2887, survivor 3
# gen: 6
# parent_a: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s5.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1137_s2.py (gen5)
# born: 2026-05-29T23:46:28Z

"""Hybrid Feature‑Tree Fusion
==========================

This module fuses the two parent algorithms:

* **Parent A** – ``krampus_brainmap.py`` – provides deterministic
  feature extraction from arbitrary text, yielding a 24‑dimensional
  numeric vector per document.

* **Parent B** – ``hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m296_s0.py`` –
  supplies geometric tree utilities, a grade‑1 *Multivector* encoding of
  vectors, a Hoeffding‑bound confidence term and a hybrid edge‑cost
  formula.

**Mathematical bridge**

1. Each node of a tree is a document. Its feature dictionary from
   Parent A is converted to a grade‑1 multivector **M(ℓ)** (grade‑1
   blades correspond to vector components). The geometric product of two
   such multivectors reduces to the Euclidean dot‑product:

   ``scalar(M(ℓ_u) * M(ℓ_v)) = ℓ_u · ℓ_v``

2. For an edge *(u,v)* with Euclidean length ``d_uv`` (Parent B) we define

   ``w_uv = ½·(ℓ_u·ℓ_v)``

   ``ε    = HoeffdingBound(n, R, δ)``  (statistical confidence)

   ``c_uv = (w_uv·d_uv)·(1+ε)``

   The total tree cost is the sum of ``c_uv`` over all edges.  A simple
   Gaussian Bayesian update yields a posterior estimate of the cost.

The three public functions below demonstrate the hybrid pipeline:
``build_feature_tree``, ``hybrid_tree_cost`` and ``bayesian_update``.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

# ---------------------------------------------------------------------------
# Deterministic pseudo‑random generator based on text content (Parent A)
# ---------------------------------------------------------------------------


def _rng_from_text(text: str) -> random.Random:
    """Create a deterministic RNG seeded from a stable hash of *text*."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Parent A – Feature extraction (deterministic stub)
# ---------------------------------------------------------------------------


def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑features for demonstration."""
    rnd = _rng_from_text(text)
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "operator_ledger_density",
        "operator_recursion_score",
        "operator_directive_ratio",
        "operator_target_density",
        "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy",
        "psyche_dissociative_index",
        "psyche_wrath_velocity",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
        "resilience_logic_crucifixion_index",
        "resilience_conspiracy_grounding_ratio",
        "resilience_chaotic_good_tax",
        "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density",
        "rainmaker_asset_structuring_weight",
        "rainmaker_pitch_formatting_ratio",
        "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline",
        "telemetry_manic_velocity",
    ]
    return {k: rnd.random() * 10.0 for k in keys}


def extract_master_vector(text: str) -> np.ndarray:
    """
    Return a 24‑dimensional numeric vector (numpy array) derived from the
    full feature dictionary.  The order follows the ``keys`` list in
    ``extract_full_features``.
    """
    if not text.strip():
        return np.zeros(24)
    full = extract_full_features(text)
    # Preserve the same ordering as in ``extract_full_features``.
    ordered_keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "operator_ledger_density",
        "operator_recursion_score",
        "operator_directive_ratio",
        "operator_target_density",
        "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy",
        "psyche_dissociative_index",
        "psyche_wrath_velocity",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
        "resilience_logic_crucifixion_index",
        "resilience_conspiracy_grounding_ratio",
        "resilience_chaotic_good_tax",
        "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density",
        "rainmaker_asset_structuring_weight",
        "rainmaker_pitch_formatting_ratio",
        "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline",
        "telemetry_manic_velocity",
    ]
    return np.array([full[k] for k in ordered_keys], dtype=float)


# ---------------------------------------------------------------------------
# Parent B – Geometric tree utilities (trimmed)
# ---------------------------------------------------------------------------


def euclidean_length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float]]:
    """
    Build an adjacency list (directed away from *root*) and a mapping
    ``edge → length`` using Euclidean geometry.
    """
    # Build undirected adjacency first.
    undirected: Dict[str, List[str]] = {k: [] for k in nodes}
    for u, v in edges:
        undirected[u].append(v)
        undirected[v].append(u)

    # BFS from root to orient edges.
    adjacency: Dict[str, List[str]] = {k: [] for k in nodes}
    visited = {root}
    queue = [root]

    while queue:
        cur = queue.pop(0)
        for nb in undirected[cur]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
                adjacency[cur].append(nb)

    # Edge lengths.
    edge_len: Dict[Tuple[str, str], float] = {}
    for parent, children in adjacency.items():
        for child in children:
            edge_len[(parent, child)] = euclidean_length(nodes[parent], nodes[child])

    return adjacency, edge_len


# ---------------------------------------------------------------------------
# Multivector grade‑1 encoding (Parent B)
# ---------------------------------------------------------------------------


class Multivector:
    """
    Grade‑1 multivector (vector) over ℝⁿ.  Internally stores a 1‑D numpy
    array ``components``.  The geometric product of two grade‑1
    multivectors reduces to the scalar dot‑product plus a bivector term.
    For the hybrid cost we only need the scalar part, which equals the
    Euclidean dot product.
    """

    def __init__(self, components: np.ndarray):
        if components.ndim != 1:
            raise ValueError("Multivector must be a 1‑D array")
        self.components = components.astype(float)

    def __mul__(self, other: "Multivector") -> "MultivectorProduct":
        if not isinstance(other, Multivector):
            raise TypeError("Geometric product defined only between Multivectors")
        # Scalar (dot) part:
        scalar = float(np.dot(self.components, other.components))
        # Bivector part is ignored for this hybrid algorithm.
        return MultivectorProduct(scalar)

    def __repr__(self) -> str:
        return f"Multivector({self.components})"


class MultivectorProduct:
    """Container for the scalar part of a geometric product."""

    def __init__(self, scalar: float):
        self.scalar = scalar

    def __float__(self) -> float:
        return self.scalar

    def __repr__(self) -> str:
        return f"MultivectorProduct(scalar={self.scalar})"


# ---------------------------------------------------------------------------
# Hoeffding bound (Parent B)
# ---------------------------------------------------------------------------


def hoeffding_bound(n: int, R: float, delta: float) -> float:
    """
    Hoeffding bound ε such that with probability 1‑δ the true mean lies
    within ε of the empirical mean for n i.i.d. samples bounded in
    ``[a, b]`` where ``R = b‑a``.
    """
    if n <= 0:
        raise ValueError("Sample size n must be positive")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt((R ** 2 * math.log(2.0 / delta)) / (2.0 * n))


# ---------------------------------------------------------------------------
# Hybrid pipeline
# ---------------------------------------------------------------------------


def build_feature_tree(
    texts: Dict[str, str],
    positions: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, Multivector]]:
    """
    1. Extract a 24‑dimensional feature vector for each node using
       ``extract_master_vector``.
    2. Encode each vector as a grade‑1 ``Multivector``.
    3. Build adjacency and edge‑length structures via ``tree_metrics``.

    Returns ``(adjacency, edge_lengths, node_mv)``.
    """
    # 1‑2. Feature → Multivector
    node_mv: Dict[str, Multivector] = {}
    for nid, txt in texts.items():
        vec = extract_master_vector(txt)
        node_mv[nid] = Multivector(vec)

    # 3. Geometry
    adjacency, edge_lengths = tree_metrics(positions, edges, root)
    return adjacency, edge_lengths, node_mv


def hybrid_tree_cost(
    adjacency: Dict[str, List[str]],
    edge_lengths: Dict[Tuple[str, str], float],
    node_mv: Dict[str, Multivector],
    *,
    hoeffding_n: int = 30,
    hoeffding_R: float = 10.0,
    hoeffding_delta: float = 0.05,
) -> float:
    """
    Compute the hybrid cost ``C = Σ c_uv`` over all directed edges
    (parent → child) using the formula:

        w_uv = ½·(ℓ_u·ℓ_v)          # scalar part of geometric product
        ε    = HoeffdingBound(...)
        c_uv = (w_uv·d_uv)·(1+ε)

    ``node_mv`` supplies the multivector representation of each node.
    """
    epsilon = hoeffding_bound(hoeffding_n, hoeffding_R, hoeffding_delta)
    total = 0.0
    for parent, children in adjacency.items():
        for child in children:
            # scalar part of geometric product
            prod = node_mv[parent] * node_mv[child]  # MultivectorProduct
            w_uv = 0.5 * prod.scalar
            d_uv = edge_lengths[(parent, child)]
            c_uv = (w_uv * d_uv) * (1.0 + epsilon)
            total += c_uv
    return total


def bayesian_update(
    prior_mean: float,
    prior_variance: float,
    observation: float,
    observation_variance: float,
) -> Tuple[float, float]:
    """
    Perform a conjugate Gaussian update:

        posterior_variance = 1 / (1/prior_variance + 1/obs_variance)
        posterior_mean     = posterior_variance *
                             (prior_mean/prior_variance + observation/obs_variance)

    Returns ``(posterior_mean, posterior_variance)``.
    """
    if prior_variance <= 0 or observation_variance <= 0:
        raise ValueError("Variances must be positive")
    precision_post = 1.0 / prior_variance + 1.0 / observation_variance
    posterior_variance = 1.0 / precision_post
    posterior_mean = posterior_variance * (
        prior_mean / prior_variance + observation / observation_variance
    )
    return posterior_mean, posterior_variance


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Sample documents (node identifiers → text)
    docs = {
        "A": "The quick brown fox jumps over the lazy dog.",
        "B": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "C": "Quantum entanglement defies classical intuition.",
        "D": "Neural networks approximate arbitrary functions.",
    }

    # 2‑D positions for each node (arbitrary layout)
    pos = {
        "A": (0.0, 0.0),
        "B": (2.0, 1.0),
        "C": (4.0, 0.0),
        "D": (6.0, 1.5),
    }

    # Undirected edges; the root will be 'A'
    edge_list = [("A", "B"), ("B", "C"), ("C", "D")]

    # Build the hybrid structure
    adj, edge_len, node_mv = build_feature_tree(docs, pos, edge_list, root="A")

    # Compute hybrid cost
    cost = hybrid_tree_cost(adj, edge_len, node_mv)

    # Bayesian update with a vague prior
    prior_mu, prior_var = 0.0, 1e6
    obs_var = 1.0  # assumed observation noise
    post_mu, post_var = bayesian_update(prior_mu, prior_var, cost, obs_var)

    print("Adjacency:", adj)
    print("Edge lengths:", {k: round(v, 3) for k, v in edge_len.items()})
    print("Hybrid tree cost:", round(cost, 4))
    print("Posterior mean after Bayesian update:", round(post_mu, 4))
    print("Posterior variance:", round(post_var, 4))

    # Simple sanity check – cost should be a finite positive number
    assert math.isfinite(cost) and cost > 0, "Cost computation failed"
    sys.exit(0)