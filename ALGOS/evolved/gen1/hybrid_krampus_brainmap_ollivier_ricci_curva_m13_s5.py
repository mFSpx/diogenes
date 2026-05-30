# DARWIN HAMMER — match 13, survivor 5
# gen: 1
# parent_a: krampus_brainmap.py (gen0)
# parent_b: ollivier_ricci_curvature.py (gen0)
# born: 2026-05-29T23:19:18Z

from __future__ import annotations

import hashlib
import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Deterministic pseudo‑random generator based on text content
# ---------------------------------------------------------------------------


def _rng_from_text(text: str) -> random.Random:
    """Create a deterministic RNG seeded from a stable hash of *text*."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Parent A – Krampus feature extraction (deterministic stub)
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


def extract_master_vector(text: str) -> Dict[str, float]:
    """Human‑readable 24‑dimensional vector."""
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


def brain_xyz(master: Dict[str, float], curvature_score: float) -> Dict[str, float]:
    """
    Deeper integration of curvature:
    - each axis is first computed as in the original formulation;
    - the curvature score modulates the axes multiplicatively;
    - an additional curvature‑derived offset is added to each axis
      to guarantee that graph geometry influences the final position.
    """
    # base axes (identical to original)
    x_base = (
        master.get("visceral_ratio", 0.0) * 8
        + master.get("ledger_density", 0.0) * 6
        + min(master.get("directive_ratio", 0.0), 8.0) / 8
        + master.get("recursion_score", 0.0) * 4
    )
    y_base = (
        master.get("forensic_shield_ratio", 0.0) * 6
        + master.get("poetic_entropy", 0.0) * 4
        + min(master.get("dissociative_index", 0.0), 8.0) / 8
        + master.get("resource_exhaustion_metric", 0.0) * 6
        + master.get("bureaucratic_weaponization_index", 0.0) * 4
    )
    z_base = (
        master.get("corporate_grit_tension", 0.0) * 6
        + master.get("countdown_density", 0.0) * 6
        + master.get("asset_structuring_weight", 0.0) * 4
        + master.get("swarm_orchestration_density", 0.0) * 4
        + master.get("chaotic_good_tax", 0.0) * 4
        + master.get("agent_symmetry_ratio", 0.0) * 0.5
        + master.get("protocol_discipline", 0.0) * 0.2
        + master.get("manic_velocity", 0.0) * 0.4
    )

    # curvature modulation: map curvature ∈ (‑∞,1] to a factor ≈[0.5,1.5]
    factor = 1.0 + 0.5 * curvature_score  # curvature_score is already averaged per node

    # small offset proportional to curvature to break possible degeneracies
    offset = curvature_score * 2.0

    return {
        "x": x_base * factor + offset,
        "y": y_base * factor + offset,
        "z": z_base * factor + offset,
    }


# ---------------------------------------------------------------------------
# Parent B – Ollivier‑Ricci curvature primitives (enhanced)
# ---------------------------------------------------------------------------


def lazy_rw_distribution(
    adj: Dict[int, List[int]], node: int, alpha: float = 0.5
) -> Dict[int, float]:
    """Lazy random‑walk distribution centred at *node*."""
    neighbours = adj.get(node, [])
    deg = len(neighbours)
    dist: Dict[int, float] = {node: alpha}
    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            dist[nb] = dist.get(nb, 0.0) + spread
    return dist


def _all_pairs_shortest_paths(adj: Dict[int, List[int]]) -> Tuple[np.ndarray, List[int]]:
    """
    Compute exact all‑pairs shortest‑path distances using BFS from each node.
    Returns distance matrix D (float, ∞ for unreachable) and ordered node list.
    """
    nodes = sorted(adj.keys())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    D = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    for src in nodes:
        si = idx[src]
        visited = {src}
        q = deque([(src, 0)])
        while q:
            cur, d = q.popleft()
            ci = idx[cur]
            D[si, ci] = d
            for nb in adj.get(cur, []):
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))
    return D, nodes


def _expected_transport_cost(
    dist_matrix: np.ndarray,
    p: Dict[int, float],
    q: Dict[int, float],
    node_index: Dict[int, int],
) -> float:
    """
    Upper‑bound on the 1‑Wasserstein distance using the product coupling:
        W₁(p,q) ≤ Σ_{i,j} d(i,j) p_i q_j
    """
    cost = 0.0
    for i, pi in p.items():
        ii = node_index[i]
        for j, qj in q.items():
            jj = node_index[j]
            d = dist_matrix[ii, jj]
            if np.isfinite(d):
                cost += d * pi * qj
    return cost


def compute_edge_curvature(
    adj: Dict[int, List[int]],
    dist_matrix: np.ndarray,
    node_index: Dict[int, int],
    alpha: float = 0.5,
) -> Dict[Tuple[int, int], float]:
    """
    Compute Ollivier‑Ricci curvature for every undirected edge (i,j).
    Uses the product‑coupling upper bound for W₁.
    """
    curv: Dict[Tuple[int, int], float] = {}
    for i, neighbours in adj.items():
        pi = lazy_rw_distribution(adj, i, alpha)
        for j in neighbours:
            if i < j:  # ensure each edge once
                pj = lazy_rw_distribution(adj, j, alpha)
                w1 = _expected_transport_cost(dist_matrix, pi, pj, node_index)
                d_ij = dist_matrix[node_index[i], node_index[j]]
                if d_ij == 0 or not np.isfinite(d_ij):
                    kappa = 0.0
                else:
                    kappa = 1.0 - w1 / d_ij
                curv[(i, j)] = kappa
    return curv


def average_node_curvature(
    edge_curv: Dict[Tuple[int, int], float], nodes: List[int]
) -> Dict[int, float]:
    """Mean curvature over incident edges for each node."""
    agg: Dict[int, List[float]] = {n: [] for n in nodes}
    for (i, j), k in edge_curv.items():
        agg[i].append(k)
        agg[j].append(k)
    return {n: (sum(vals) / len(vals) if vals else 0.0) for n, vals in agg.items()}


# ---------------------------------------------------------------------------
# Hybrid utilities
# ---------------------------------------------------------------------------


def _vector_from_master(master: Dict[str, float]) -> np.ndarray:
    """Consistent ordering of master‑vector components for distance computation."""
    keys = sorted(master.keys())
    return np.array([master[k] for k in keys], dtype=np.float64)


def hybrid_build_adj(
    masters: List[Dict[str, float]], k: int = 5
) -> Tuple[Dict[int, List[int]], np.ndarray, List[int]]:
    """
    Build a symmetric k‑nearest‑neighbour adjacency list.
    Returns (adjacency, distance_matrix, node_order).
    """
    if not masters:
        return {}, np.array([[]]), []

    vectors = np.stack([_vector_from_master(m) for m in masters])
    # pairwise Euclidean distances
    diff = vectors[:, None, :] - vectors[None, :, :]
    dists = np.linalg.norm(diff, axis=2)

    n = len(masters)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        # indices of k nearest neighbours (excluding self)
        neigh_idx = np.argpartition(dists[i], k + 1)[: k + 1]
        neigh_idx = neigh_idx[neigh_idx != i]  # drop self if present
        for j in neigh_idx:
            adj[i].append(j)
            adj[j].append(i)  # enforce symmetry

    # remove possible duplicate entries
    for i in adj:
        adj[i] = sorted(set(adj[i]))

    # compute exact shortest‑path distances for curvature
    dist_matrix, node_order = _all_pairs_shortest_paths(adj)
    return adj, dist_matrix, node_order


def hybrid_node_curvature(
    masters: List[Dict[str, float]],
    adj: Dict[int, List[int]],
    dist_matrix: np.ndarray,
    node_order: List[int],
    alpha: float = 0.5,
) -> Dict[int, float]:
    """
    Run the Ollivier‑Ricci pipeline and return per‑node average curvature.
    """
    node_index = {node: i for i, node in enumerate(node_order)}
    edge_curv = compute_edge_curvature(adj, dist_matrix, node_index, alpha)
    return average_node_curvature(edge_curv, node_order)


def hybrid_brain_xyz(
    master: Dict[str, float], curvature_score: float
) -> Dict[str, float]:
    """Convenient wrapper that mirrors the original API."""
    return brain_xyz(master, curvature_score)


# ---------------------------------------------------------------------------
# Example orchestration (can be removed / adapted by downstream code)
# ---------------------------------------------------------------------------


def hybrid_process_texts(texts: List[str]) -> List[Dict[str, float]]:
    """
    Full pipeline:
    1. Extract master vectors.
    2. Build graph.
    3. Compute curvature scores.
    4. Produce final 3‑D coordinates.
    Returns a list of dictionaries with keys 'x','y','z' (and optionally
    intermediate data for debugging).
    """
    masters = [extract_master_vector(t) for t in texts]
    adj, dist_mat, order = hybrid_build_adj(masters, k=5)
    curv_per_node = hybrid_node_curvature(masters, adj, dist_mat, order, alpha=0.5)

    results: List[Dict[str, float]] = []
    for idx, master in enumerate(masters):
        curvature = curv_per_node.get(idx, 0.0)
        xyz = hybrid_brain_xyz(master, curvature)
        results.append(xyz)
    return results