# DARWIN HAMMER — match 98, survivor 1
# gen: 3
# parent_a: hybrid_ternary_router_hybrid_minimum_cost__m36_s4.py (gen2)
# parent_b: hybrid_ternary_router_hybrid_minimum_cost__m36_s5.py (gen2)
# born: 2026-05-29T23:25:47Z

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
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Edge]]:
    """
    Return a symmetric matrix L where L[i, j] = Euclidean length if (i, j) is an edge,
    otherwise 0. Also return the ordered list of edge index pairs matching the
    non‑zero entries of L (used for vectorised prior updates) and edge list.
    """
    idx_map = {name: i for i, name in enumerate(sorted(nodes))}
    n = len(nodes)
    L = np.zeros((n, n), dtype=float)
    edge_idx: List[Tuple[int, int]] = []
    edge_list: List[Edge] = []

    for a, b in edges:
        i, j = idx_map[a], idx_map[b]
        length = euclidean_length(nodes[a], nodes[b])
        L[i, j] = L[j, i] = length
        edge_idx.append((i, j))
        edge_list.append((a, b))

    return L, edge_idx, edge_list


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
    L, edge_idx, edge_list = build_length_matrix(nodes, edges)

    # ------------------------------------------------------------------
    # 2️⃣ Assemble prior and evidence vectors in the same order as edge_idx
    # ------------------------------------------------------------------
    priors = np.array([edge_priors[e] for e in edge_list], dtype=float)
    if evidence is None:
        evidence_vec = np.ones_like(priors)
    else:
        evidence_vec = np.array([evidence.get(e, 1.0) for e in edge_list], dtype=float)

    # ------------------------------------------------------------------
    # 3️⃣ Vectorised Bayesian update
    # ------------------------------------------------------------------
    posteriors = bayes_update_priors_vector(
        priors, evidence_vec, likelihood=likelihood, false_positive=false_positive
    )

    # ------------------------------------------------------------------
    # 4️⃣ Updated material cost = Σ lengthₑ * posteriorₑ
    # ------------------------------------------------------------------
    material_cost = np.sum(L.flat[edge_idx] * posteriors)

    # ------------------------------------------------------------------
    # 5️⃣ Path‑weight term = γ * Σ root‑to‑node distances
    # ------------------------------------------------------------------
    root_distances = compute_root_distances(nodes, edges, root)
    path_cost = path_weight * sum(root_distances.values())

    return material_cost + path_cost