# DARWIN HAMMER — match 13, survivor 2
# gen: 1
# parent_a: krampus_brainmap.py (gen0)
# parent_b: ollivier_ricci_curvature.py (gen0)
# born: 2026-05-29T23:19:18Z

"""Hybrid Krampus-Ollivier module.

This module fuses two distinct mathematical pipelines:

* **Krampus brain‑map (Parent A)** – extracts a high‑dimensional
  feature vector from free‑form text and projects it deterministically
  onto a 3‑axis space using a weighted linear combination
  (`brain_xyz`).

* **Ollivier‑Ricci curvature (Parent B)** – builds a weighted graph,
  computes lazy random‑walk distributions on each node, evaluates the
  Earth‑Mover (Wasserstein‑1) distance between neighbouring nodes and
  derives a curvature value `kappa(x, y) = 1 - W₁(mₓ,m_y)/d(x,y)`.

**Mathematical bridge**

Each text is represented by a *master vector* **v** ∈ ℝⁿ (n≈20).  
Treat every master vector as a node in a graph **G**.  Edge weights are
the Euclidean distance `d(i,j)=‖v_i‑v_j‖₂`.  By thresholding these
distances we obtain an un‑weighted adjacency list suitable for the
Ollivier‑Ricci routine.  The resulting curvature `κ(i,j)` quantifies
how tightly the neighbourhoods of two texts overlap.  We inject the
*average incident curvature* of a node as an additional scalar feature
`curvature_score` into the original Krampus projection, thus creating a
single unified mapping from raw text to a 3‑D point that respects both
semantic feature composition and graph‑geometric connectivity.

The three core hybrid functions are:

* `hybrid_build_adj` – builds the adjacency structure from a list of
  master vectors.
* `hybrid_node_curvature` – runs Ollivier‑Ricci on that graph and
  returns per‑node average curvature.
* `hybrid_brain_xyz` – augments the original `brain_xyz` with the
  curvature score to produce the final 3‑D coordinates.

All code is self‑contained and relies only on the standard library and
NumPy.
"""

from __future__ import annotations

import math
import random
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – Krampus feature extraction (trimmed for brevity)
# ---------------------------------------------------------------------------

def extract_full_features(text: str) -> Dict[str, float]:
    """Placeholder stub – in a real system this would call the
    specialised Krampus sticker extractors.  Here we fabricate deterministic
    pseudo‑features from the input string for demonstration purposes.
    """
    # deterministic pseudo‑random based on text hash
    rnd = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "operator_ledger_density",
        "operator_recursion_score", "operator_directive_ratio",
        "operator_target_density", "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy", "psyche_dissociative_index",
        "psyche_wrath_velocity", "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric", "resilience_swarm_orchestration_density",
        "resilience_logic_crucifixion_index", "resilience_conspiracy_grounding_ratio",
        "resilience_chaotic_good_tax", "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density", "rainmaker_asset_structuring_weight",
        "rainmaker_pitch_formatting_ratio", "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline", "telemetry_manic_velocity",
    ]
    return {k: rnd.random() * 10 for k in keys}


def extract_master_vector(text: str) -> Dict[str, float]:
    """Human‑readable 20+ dimensional vector for exports/maps."""
    if not text.strip():
        return {}
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "ledger_density": f.get("operator_ledger_density", 0.0),
        "recursion_score": f.get("operator_recursion_score", 0.0),
        "directive_ratio": f.get("operator_directive_ratio", 0.0),
        "target_density": f.get("operator_target_density", 0.0),
        "forensic_shield_ratio": f.get("psyche_forensic_shield_ratio", 0.0),
        "poetic_entropy": f.get("psyche_poetic_entropy", 0.0),
        "dissociative_index": f.get("psyche_dissociative_index", 0.0),
        "wrath_velocity": f.get("psyche_wrath_velocity", 0.0),
        "bureaucratic_weaponization_index": f.get(
            "resilience_bureaucratic_weaponization_index", 0.0
        ),
        "resource_exhaustion_metric": f.get(
            "resilience_resource_exhaustion_metric", 0.0
        ),
        "swarm_orchestration_density": f.get(
            "resilience_swarm_orchestration_density", 0.0
        ),
        "logic_crucifixion_index": f.get(
            "resilience_logic_crucifixion_index", 0.0
        ),
        "conspiracy_grounding_ratio": f.get(
            "resilience_conspiracy_grounding_ratio", 0.0
        ),
        "chaotic_good_tax": f.get("resilience_chaotic_good_tax", 0.0),
        "corporate_grit_tension": f.get("rainmaker_corporate_grit_tension", 0.0),
        "countdown_density": f.get("rainmaker_countdown_density", 0.0),
        "asset_structuring_weight": f.get(
            "rainmaker_asset_structuring_weight", 0.0
        ),
        "pitch_formatting_ratio": f.get("rainmaker_pitch_formatting_ratio", 0.0),
        "agent_symmetry_ratio": f.get("telemetry_agent_symmetry_ratio", 0.0),
        "protocol_discipline": f.get("telemetry_protocol_discipline", 0.0),
        "manic_velocity": f.get("telemetry_manic_velocity", 0.0),
    }


def brain_xyz(master: Dict[str, float]) -> Dict[str, float]:
    """Original deterministic 3‑axis projection."""
    x_architect_operator = (
        master.get("visceral_ratio", 0.0) * 8
        + master.get("ledger_density", 0.0) * 6
        + min(master.get("directive_ratio", 0.0), 8.0) / 8
        + master.get("recursion_score", 0.0) * 4
    )
    y_psyche_resilience = (
        master.get("forensic_shield_ratio", 0.0) * 6
        + master.get("poetic_entropy", 0.0) * 4
        + min(master.get("dissociative_index", 0.0), 8.0) / 8
        + master.get("resource_exhaustion_metric", 0.0) * 6
        + master.get("bureaucratic_weaponization_index", 0.0) * 4
    )
    z_rainmaker_sprint = (
        master.get("corporate_grit_tension", 0.0) * 6
        + master.get("countdown_density", 0.0) * 6
        + master.get("asset_structuring_weight", 0.0) * 4
        + master.get("swarm_orchestration_density", 0.0) * 4
        + master.get("chaotic_good_tax", 0.0) * 4
        + master.get("agent_symmetry_ratio", 0.0) * 0.5
        + master.get("protocol_discipline", 0.0) * 0.2
        + master.get("manic_velocity", 0.0) * 0.4
    )
    return {"x": x_architect_operator, "y": y_psyche_resilience, "z": z_rainmaker_sprint}


# ---------------------------------------------------------------------------
# Parent B – Ollivier‑Ricci curvature primitives
# ---------------------------------------------------------------------------

def lazy_rw_distribution(adj: Dict[int, List[int]], node: int, alpha: float = 0.5) -> Dict[int, float]:
    """Lazy random walk distribution centred at *node*."""
    neighbours = adj.get(node, [])
    deg = len(neighbours)
    dist = {node: alpha}
    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            dist[nb] = dist.get(nb, 0.0) + spread
    return dist


def bfs_distances(adj: Dict[int, List[int]]) -> Tuple[np.ndarray, List[int]]:
    """All‑pairs shortest‑path distances via BFS."""
    node_ids = sorted(adj.keys())
    n = len(node_ids)
    idx = {v: i for i, v in enumerate(node_ids)}
    D = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    for src in node_ids:
        si = idx[src]
        visited = {src}
        q = deque([(src, 0)])
        while q:
            node, dist = q.popleft()
            D[si, idx[node]] = dist
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, dist + 1))
    return D, node_ids


def wasserstein1_graph(
    mu: Dict[int, float],
    nu: Dict[int, float],
    D: np.ndarray,
    node_ids: List[int],
) -> float:
    """Greedy approximation of the Wasserstein‑1 distance."""
    n = len(node_ids)
    idx = {v: i for i, v in enumerate(node_ids)}

    supply = np.array([mu.get(v, 0.0) for v in node_ids], dtype=np.float64)
    demand = np.array([nu.get(v, 0.0) for v in node_ids], dtype=np.float64)

    # Normalise
    s_sum = supply.sum()
    d_sum = demand.sum()
    if s_sum > 0:
        supply /= s_sum
    if d_sum > 0:
        demand /= d_sum

    # Sorted cost entries
    costs = [(D[i, j], i, j) for i in range(n) for j in range(n)]
    costs.sort(key=lambda t: t[0])

    total_cost = 0.0
    eps = 1e-12
    supply_rem = supply.copy()
    demand_rem = demand.copy()

    for cost, i, j in costs:
        if supply_rem[i] < eps or demand_rem[j] < eps:
            continue
        flow = min(supply_rem[i], demand_rem[j])
        total_cost += cost * flow
        supply_rem[i] -= flow
        demand_rem[j] -= flow

    return float(total_cost)


def ollivier_ricci(adj: Dict[int, List[int]], alpha: float = 0.5) -> Dict[Tuple[int, int], float]:
    """Compute Ollivier‑Ricci curvature for each edge of *adj*."""
    D, node_ids = bfs_distances(adj)
    curvatures: Dict[Tuple[int, int], float] = {}
    seen: set[Tuple[int, int]] = set()

    for x in adj:
        for y in adj[x]:
            edge = (min(x, y), max(x, y))
            if edge in seen:
                continue
            seen.add(edge)

            xi = node_ids.index(x)
            yi = node_ids.index(y)
            d_xy = D[xi, yi]
            if d_xy < 1e-12:
                curvatures[(x, y)] = 1.0
                continue

            mu = lazy_rw_distribution(adj, x, alpha)
            nu = lazy_rw_distribution(adj, y, alpha)
            w1 = wasserstein1_graph(mu, nu, D, node_ids)
            curvatures[(x, y)] = 1.0 - w1 / d_xy

    return curvatures


# ---------------------------------------------------------------------------
# Hybrid layer – linking the two worlds
# ---------------------------------------------------------------------------

# Fixed ordering of master‑vector keys to guarantee consistent Euclidean embedding
_FEATURE_ORDER = [
    "visceral_ratio", "tech_ratio", "legal_osint_ratio", "ledger_density",
    "recursion_score", "directive_ratio", "target_density",
    "forensic_shield_ratio", "poetic_entropy", "dissociative_index",
    "wrath_velocity", "bureaucratic_weaponization_index",
    "resource_exhaustion_metric", "swarm_orchestration_density",
    "logic_crucifixion_index", "conspiracy_grounding_ratio",
    "chaotic_good_tax", "corporate_grit_tension", "countdown_density",
    "asset_structuring_weight", "pitch_formatting_ratio",
    "agent_symmetry_ratio", "protocol_discipline", "manic_velocity",
]


def vector_from_master(master: Dict[str, float]) -> np.ndarray:
    """Convert a master dict to a deterministic NumPy vector using _FEATURE_ORDER."""
    return np.array([master.get(k, 0.0) for k in _FEATURE_ORDER], dtype=np.float64)


def hybrid_build_adj(masters: List[Dict[str, float]], distance_thresh: float = 5.0) -> Dict[int, List[int]]:
    """
    Build an undirected adjacency list where each node corresponds to a master vector.
    An edge (i, j) exists iff the Euclidean distance between vectors i and j
    is ≤ *distance_thresh*.
    """
    vectors = [vector_from_master(m) for m in masters]
    adj: Dict[int, List[int]] = {i: [] for i in range(len(vectors))}

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dist = np.linalg.norm(vectors[i] - vectors[j])
            if dist <= distance_thresh:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def hybrid_node_curvature(masters: List[Dict[str, float]], distance_thresh: float = 5.0, alpha: float = 0.5) -> List[float]:
    """
    Compute the average Ollivier‑Ricci curvature incident to each node.
    Returns a list aligned with *masters* where each entry is the mean curvature
    of edges touching that node (0.0 if the node is isolated).
    """
    adj = hybrid_build_adj(masters, distance_thresh)
    curv_dict = ollivier_ricci(adj, alpha)

    # Aggregate per‑node
    node_sums: Dict[int, float] = {i: 0.0 for i in range(len(masters))}
    node_counts: Dict[int, int] = {i: 0 for i in range(len(masters))}

    for (x, y), kappa in curv_dict.items():
        node_sums[x] += kappa
        node_s