# DARWIN HAMMER — match 98, survivor 0
# gen: 3
# parent_a: hybrid_ternary_router_hybrid_minimum_cost__m36_s4.py (gen2)
# parent_b: hybrid_ternary_router_hybrid_minimum_cost__m36_s5.py (gen2)
# born: 2026-05-29T23:25:47Z

"""Hybrid algorithm merging the Bayesian edge‑prior update from
`hybrid_minimum_cost_tree_bayes_update` (Parent A) with the
ternary‑router style graph handling of `ternary_router` (Parent B).

Mathematical bridge
-------------------
Both parents treat a graph **G = (V, E)** with Euclidean node coordinates.
Parent A updates each edge prior *pₑ* using Bayes’ rule with a scalar
likelihood *ℓ* and false‑positive rate *α*:

    pₑ' = pₑ·ℓ·eₑ / (pₑ·ℓ·eₑ + (1−pₑ)·α)

where *eₑ* is an evidence factor (default 1).  
Parent B builds a tree cost from the sum of edge lengths weighted by the
posterior *pₑ'* plus a path‑weight term proportional to the distance from
the root to every node.

The hybrid replaces the explicit Python loops with **matrix‑vector**
operations using NumPy, allowing simultaneous Bayesian updates of all
edges and a compact expression for the total cost:

* Let **L** be the *|V|×|V|* matrix of Euclidean lengths (0 on the diagonal,
  symmetric, non‑zero only where an edge exists).
* Let **P** be the vector of edge priors ordered identically to the
  non‑zero entries of **L**.
* Let **E** be the evidence vector (same ordering).

The posterior vector **P'** is obtained element‑wise:

    P' = (P * ℓ * E) / (P * ℓ * E + (1−P) * α)

The updated material cost is the dot product **Lₑ · P'**, i.e. the sum of
each edge length times its posterior probability.  
The root‑to‑node distances are computed once by a depth‑first traversal;
they are then summed and multiplied by *γ* (the `path_weight`).

All three core functions below are pure NumPy / standard‑library code,
demonstrating the hybrid operation without external dependencies."""


import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Basic I/O helpers (shared with parents)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNTIME_DIR = ROOT / "04_RUNTIME" / "fairyfuse"
DEFAULT_HEARTBEAT = RUNTIME_DIR / "ternary_router_heartbeat.jsonl"


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def emit_json(obj: Any) -> None:
    """Print a JSON object with deterministic key order."""
    print(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


# ----------------------------------------------------------------------
# Geometry utilities
# ----------------------------------------------------------------------
def euclidean_length(a: Point, b: Point) -> float:
    """Straight‑line distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_length_matrix(
    nodes: Dict[str, Point], edges: List[Edge]
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Return a symmetric matrix L where L[i, j] = Euclidean length if (i, j) is an edge,
    otherwise 0. Also return the ordered list of edge index pairs matching the
    non‑zero entries of L (used for vectorised prior updates).
    """
    idx_map = {name: i for i, name in enumerate(sorted(nodes))}
    n = len(nodes)
    L = np.zeros((n, n), dtype=float)
    edge_idx: List[Tuple[int, int]] = []

    for a, b in edges:
        i, j = idx_map[a], idx_map[b]
        length = euclidean_length(nodes[a], nodes[b])
        L[i, j] = L[j, i] = length
        edge_idx.append((i, j))

    return L, edge_idx


# ----------------------------------------------------------------------
# Vectorised Bayesian update (mathematical core from Parent A)
# ----------------------------------------------------------------------
def bayes_update_priors_vector(
    priors: np.ndarray,
    evidence: np.ndarray,
    likelihood: float = 0.8,
    false_positive: float = 0.1,
) -> np.ndarray:
    """
    Perform element‑wise Bayesian update of edge priors.

    Parameters
    ----------
    priors : np.ndarray
        Prior probabilities for each edge (shape: (m,)).
    evidence : np.ndarray
        Evidence scaling factor per edge (shape: (m,)).
    likelihood : float
        True‑positive likelihood ℓ.
    false_positive : float
        False‑positive rate α.

    Returns
    -------
    np.ndarray
        Posterior probabilities for each edge.
    """
    eff_like = likelihood * evidence
    numerator = priors * eff_like
    denominator = numerator + (1.0 - priors) * false_positive
    # Guard against division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        posterior = np.where(denominator > 0, numerator / denominator, 0.0)
    return posterior


# ----------------------------------------------------------------------
# Depth‑first distance accumulation from a root (Parent B style)
# ----------------------------------------------------------------------
def compute_root_distances(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Dict[str, float]:
    """
    Return a mapping node → cumulative Euclidean distance from `root`
    along the spanning tree defined by `edges`. Assumes the edge set forms a
    connected undirected graph.
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + euclidean_length(nodes[cur], nodes[nxt])
                stack.append(nxt)
    return dist


# ----------------------------------------------------------------------
# Hybrid cost function (core fusion)
# ----------------------------------------------------------------------
def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    edge_priors: Dict[Edge, float],
    evidence: Dict[Edge, float] | None = None,
    path_weight: float = 0.2,
    likelihood: float = 0.8,
    false_positive: float = 0.1,
) -> float:
    """
    Compute the total cost of a tree by:
      1. Vectorised Bayesian update of edge priors using supplied evidence.
      2. Summing updated edge lengths weighted by posterior probabilities.
      3. Adding a path‑weight term proportional to the sum of root‑to‑node distances.

    This merges the Bayesian update of Parent A with the tree‑cost logic of
    Parent B, but operates on NumPy matrices for efficiency.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Build length matrix and ordered edge list
    # ------------------------------------------------------------------
    L, edge_idx = build_length_matrix(nodes, edges)

    # ------------------------------------------------------------------
    # 2️⃣ Assemble prior and evidence vectors in the same order as edge_idx
    # ------------------------------------------------------------------
    priors = np.array([edge_priors[e] for e in edges], dtype=float)
    if evidence is None:
        evidence_vec = np.ones_like(priors)
    else:
        evidence_vec = np.array([evidence.get(e, 1.0) for e in edges], dtype=float)

    # ------------------------------------------------------------------
    # 3️⃣ Vectorised Bayesian update
    # ------------------------------------------------------------------
    posteriors = bayes_update_priors_vector(
        priors, evidence_vec, likelihood=likelihood, false_positive=false_positive
    )

    # ------------------------------------------------------------------
    # 4️⃣ Updated material cost = Σ lengthₑ * posteriorₑ
    # ------------------------------------------------------------------
    lengths = np.array([L[i, j] for i, j in edge_idx], dtype=float)
    material_cost = float(np.dot(lengths, posteriors))

    # ------------------------------------------------------------------
    # 5️⃣ Path‑weight contribution from root distances
    # ------------------------------------------------------------------
    root_distances = compute_root_distances(nodes, edges, root)
    path_cost = path_weight * sum(root_distances.values())

    return material_cost + path_cost


# ----------------------------------------------------------------------
# Ternary router helper – selects up to three best children per node
# ----------------------------------------------------------------------
def ternary_children_selection(
    nodes: Dict[str, Point],
    edges: List[Edge],
    edge_posteriors: np.ndarray,
    max_children: int = 3,
) -> Dict[str, List[str]]:
    """
    For each node, choose up to `max_children` neighboring nodes with the
    highest posterior probabilities. Returns a dict mapping node → list of
    selected child identifiers.
    """
    # Build adjacency list with associated posterior indices
    adj: Dict[str, List[Tuple[str, int]]] = {n: [] for n in nodes}
    for idx, (a, b) in enumerate(edges):
        adj[a].append((b, idx))
        adj[b].append((a, idx))

    selection: Dict[str, List[str]] = {}
    for node, nbrs in adj.items():
        # Sort neighbors by descending posterior probability
        sorted_nbrs = sorted(nbrs, key=lambda nb: edge_posteriors[nb[1]], reverse=True)
        selection[node] = [nbr for nbr, _ in sorted_nbrs[:max_children]]
    return selection


# ----------------------------------------------------------------------
# End‑to‑end demonstration combining all pieces
# ----------------------------------------------------------------------
def demo_hybrid():
    """Run a tiny example to verify that the hybrid functions cooperate."""
    # Simple square graph
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")]
    root = "A"

    # Uniform priors, with a slight bias on the diagonal edge
    edge_priors = {e: 0.5 for e in edges}
    edge_priors[("A", "C")] = 0.2

    # Simulated evidence (strengthening the diagonal)
    evidence = {e: 1.5 if e == ("A", "C") else 1.0 for e in edges}

    cost = hybrid_tree_cost(
        nodes,
        edges,
        root,
        edge_priors,
        evidence=evidence,
        path_weight=0.2,
        likelihood=0.8,
        false_positive=0.1,
    )
    emit_json({"hybrid_cost": cost})

    # Show ternary selection based on the posterior vector
    priors_vec = np.array([edge_priors[e] for e in edges], dtype=float)
    evidence_vec = np.array([evidence.get(e, 1.0) for e in edges], dtype=float)
    post_vec = bayes_update_priors_vector(priors_vec, evidence_vec)

    selection = ternary_children_selection(nodes, edges, post_vec, max_children=3)
    emit_json({"ternary_selection": selection})


if __name__ == "__main__":
    # Smoke test – should run without raising any exception.
    demo_hybrid()