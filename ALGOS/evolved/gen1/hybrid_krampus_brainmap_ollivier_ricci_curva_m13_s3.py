# DARWIN HAMMER — match 13, survivor 3
# gen: 1
# parent_a: krampus_brainmap.py (gen0)
# parent_b: ollivier_ricci_curvature.py (gen0)
# born: 2026-05-29T23:19:18Z

from __future__ import annotations

import math
import random
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

def extract_full_features(text: str) -> Dict[str, float]:
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

def lazy_rw_distribution(adj: Dict[int, List[int]], node: int, alpha: float = 0.5) -> Dict[int, float]:
    neighbours = adj.get(node, [])
    deg = len(neighbours)
    dist = {node: alpha}
    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            dist[nb] = dist.get(nb, 0.0) + spread
    return dist

def bfs_distances(adj: Dict[int, List[int]]) -> Tuple[np.ndarray, List[int]]:
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

def earth_mover_distance(p: Dict[int, float], q: Dict[int, float]) -> float:
    assert set(p.keys()) == set(q.keys())
    return sum(abs(p[k] - q[k]) for k in p)

def ollivier_ricci_curvature(adj: Dict[int, List[int]], node1: int, node2: int) -> float:
    D, _ = bfs_distances(adj)
    d = D[node1, node2]
    if d == np.inf or d == 0:
        return 0.0
    p = lazy_rw_distribution(adj, node1)
    q = lazy_rw_distribution(adj, node2)
    W1 = earth_mover_distance(p, q)
    return 1 - W1 / d

def hybrid_build_adj(master_vectors: List[Dict[str, float]], threshold: float = 1.0) -> Dict[int, List[int]]:
    adj = {}
    for i, v1 in enumerate(master_vectors):
        for j, v2 in enumerate(master_vectors[i+1:], start=i+1):
            d = np.linalg.norm(np.array(list(v1.values())) - np.array(list(v2.values())))
            if d < threshold:
                adj.setdefault(i, []).append(j)
                adj.setdefault(j, []).append(i)
    return adj

def hybrid_node_curvature(master_vectors: List[Dict[str, float]]) -> List[float]:
    adj = hybrid_build_adj(master_vectors)
    curvatures = []
    for node in range(len(master_vectors)):
        total_curvature = 0.0
        count = 0
        for neighbour in adj.get(node, []):
            curvature = ollivier_ricci_curvature(adj, node, neighbour)
            total_curvature += curvature
            count += 1
        if count > 0:
            curvatures.append(total_curvature / count)
        else:
            curvatures.append(0.0)
    return curvatures

def hybrid_brain_xyz(master: Dict[str, float], curvature_score: float) -> Dict[str, float]:
    xyz = brain_xyz(master)
    xyz["x"] += curvature_score * 0.1
    xyz["y"] += curvature_score * 0.2
    xyz["z"] += curvature_score * 0.1
    return xyz

def main():
    text1 = "This is a test text."
    text2 = "This is another test text."
    master1 = extract_master_vector(text1)
    master2 = extract_master_vector(text2)
    curvature_score1 = hybrid_node_curvature([master1, master2])[0]
    curvature_score2 = hybrid_node_curvature([master1, master2])[1]
    xyz1 = hybrid_brain_xyz(master1, curvature_score1)
    xyz2 = hybrid_brain_xyz(master2, curvature_score2)
    print(xyz1)
    print(xyz2)

if __name__ == "__main__":
    main()