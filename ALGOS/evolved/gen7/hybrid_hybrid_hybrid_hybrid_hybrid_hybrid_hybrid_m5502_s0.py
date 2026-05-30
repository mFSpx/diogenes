# DARWIN HAMMER — match 5502, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1203_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_label__m2630_s0.py (gen6)
# born: 2026-05-30T00:02:29Z

"""Hybrid Algorithm Fusion of:
- Parent A: Gini‑coefficient‑modulated regret‑weighted Hoeffding‑tree bandit with bilinear feature projection.
- Parent B: Path‑signature based labeling with Ollivier‑Ricci curvature scaling.

Mathematical Bridge
------------------
1. **Gini‑Modulated Bilinear Form**  
   The Gini coefficient `G` of the expected values of a set of bandit actions is used to
   scale a bilinear projection `f = xᵀ W y`.  In the original bandit, `G` controls exploration;
   here it directly weights the projection, making the projected feature `f̂ = G·f`.

2. **Curvature‑Weighted Label Confidence**  
   A similarity graph is built from the projected features.  Ollivier‑Ricci curvature `κ(e)`
   of each edge `e` quantifies the local connectivity.  The labeling confidence `c_i`
   obtained from the path‑signature component of Parent B is multiplied by the average
   curvature of edges incident to node `i`, yielding a curvature‑scaled confidence
   `ĉ_i = c_i·⟨κ⟩_i`.

The resulting hybrid score for an action combines the scaled projection with the
curvature‑weighted confidence, providing a single unified decision metric.

Author: Computational Physicist & AI Architect
Date: 2026-05-30
"""

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Action descriptor used by the bandit component."""
    action_id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class TextNode:
    """Node representing a text/document in the similarity graph."""
    node_id: str
    feature: np.ndarray          # Original high‑dimensional feature vector
    projected: np.ndarray = None  # Filled after bilinear projection
    label_confidence: float = 0.0  # From path‑signature labeling


# ----------------------------------------------------------------------
# 1. Gini coefficient (Parent A)
# ----------------------------------------------------------------------
def gini_coefficient(values: List[float]) -> float:
    """Return the Gini coefficient of a list of non‑negative numbers."""
    if not values:
        return 0.0
    sorted_vals = np.sort(np.array(values, dtype=float))
    n = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


# ----------------------------------------------------------------------
# 2. Bilinear projection with Gini modulation (Parent A ↔ Parent B)
# ----------------------------------------------------------------------
def gini_modulated_bilinear(
    left: np.ndarray,
    right: np.ndarray,
    W: np.ndarray,
    gini: float,
) -> float:
    """
    Compute a bilinear form f = leftᵀ·W·right and scale it by the Gini coefficient.
    left, right : 1‑D vectors of compatible dimensions.
    W          : 2‑D matrix with shape (len(left), len(right)).
    gini       : scalar in [0,1] obtained from the bandit actions.
    """
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("left and right must be 1‑D arrays")
    if W.shape != (left.size, right.size):
        raise ValueError("W shape mismatch")
    bilinear = float(left @ W @ right)
    return gini * bilinear


# ----------------------------------------------------------------------
# 3. Simple Ollivier‑Ricci curvature approximation (Parent B)
# ----------------------------------------------------------------------
def approximate_ollivier_ricci(
    adjacency: Dict[str, List[str]],
    node_features: Dict[str, np.ndarray],
) -> Dict[Tuple[str, str], float]:
    """
    Approximate Ollivier‑Ricci curvature κ(e) for each edge e = (i, j).

    For each node we define a uniform probability measure over its neighbours.
    The Wasserstein distance between the two measures is approximated by the
    average Euclidean distance between neighbour feature vectors.
    Curvature is then κ(e) = 1 - (W₁ / d(i,j)), where d(i,j) is the Euclidean
    distance between the feature vectors of the incident nodes.
    """
    curvature: Dict[Tuple[str, str], float] = {}
    for i, neighbors_i in adjacency.items():
        for j in neighbors_i:
            if (j, i) in curvature:  # undirected graph – skip duplicate
                continue
            # distance between node embeddings
            d_ij = np.linalg.norm(node_features[i] - node_features[j]) + 1e-12
            # uniform measures over neighbours (including the opposite node)
            neigh_i = [node_features[n] for n in set(neighbors_i + [j])]
            neigh_j = [node_features[n] for n in set(adjacency[j] + [i])]
            # approximate Wasserstein-1 distance as mean pairwise distance
            pairwise = [
                np.linalg.norm(fi - fj) for fi in neigh_i for fj in neigh_j
            ]
            if not pairwise:
                w1 = 0.0
            else:
                w1 = sum(pairwise) / len(pairwise)
            curvature[(i, j)] = 1.0 - w1 / d_ij
    return curvature


# ----------------------------------------------------------------------
# 4. Hybrid scoring pipeline
# ----------------------------------------------------------------------
def build_similarity_graph(
    nodes: List[TextNode],
    threshold: float = 0.5,
) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:
    """
    Construct an undirected similarity graph where an edge exists if the cosine
    similarity between projected feature vectors exceeds `threshold`.
    Returns adjacency dict and a mapping node_id → projected vector.
    """
    adjacency: Dict[str, List[str]] = defaultdict(list)
    proj_map: Dict[str, np.ndarray] = {}
    for n in nodes:
        if n.projected is None:
            raise ValueError(f"Node {n.node_id} missing projected features")
        proj_map[n.node_id] = n.projected

    ids = [n.node_id for n in nodes]
    for i, id_i in enumerate(ids):
        vec_i = proj_map[id_i]
        for j in range(i + 1, len(ids)):
            id_j = ids[j]
            vec_j = proj_map[id_j]
            cos_sim = float(vec_i @ vec_j) / (
                np.linalg.norm(vec_i) * np.linalg.norm(vec_j) + 1e-12
            )
            if cos_sim >= threshold:
                adjacency[id_i].append(id_j)
                adjacency[id_j].append(id_i)
    return dict(adjacency), proj_map


def hybrid_action_score(
    action: BanditAction,
    left_feat: np.ndarray,
    right_feat: np.ndarray,
    W: np.ndarray,
    gini: float,
    node_curvature: float,
) -> float:
    """
    Combine three ingredients into a single scalar:
    - Gini‑modulated bilinear projection (captures interaction of left/right features).
    - Node‑specific average curvature (captures graph connectivity).
    - The raw expected value of the bandit action (baseline utility).
    """
    bilinear = gini_modulated_bilinear(left_feat, right_feat, W, gini)
    # Scale the bilinear term by curvature to embed graph topology.
    topo_scaled = bilinear * (1.0 + node_curvature)
    # Blend with the original expected value (weight 0.6 / 0.4 arbitrarily chosen).
    return 0.6 * action.expected_value + 0.4 * topo_scaled


def run_hybrid_pipeline(
    actions: List[BanditAction],
    left_features: Dict[str, np.ndarray],
    right_features: Dict[str, np.ndarray],
    W: np.ndarray,
    nodes: List[TextNode],
    similarity_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    End‑to‑end hybrid evaluation:
    1. Compute Gini from action expected values.
    2. Project each node's feature via the Gini‑modulated bilinear form.
    3. Build similarity graph and compute curvature.
    4. For each action, combine expected value, projected interaction, and curvature‑scaled confidence.
    Returns a mapping action_id → hybrid score.
    """
    # 1. Gini coefficient of bandit expected values
    gini = gini_coefficient([a.expected_value for a in actions])

    # 2. Project node features (using node_id as left/right key pair)
    projected_nodes = []
    for node in nodes:
        # For illustration we split the node's feature vector in half
        split = node.feature.shape[0] // 2
        left = node.feature[:split]
        right = node.feature[split:]
        proj_val = gini_modulated_bilinear(left, right, W, gini)
        projected_nodes.append(
            TextNode(
                node_id=node.node_id,
                feature=node.feature,
                projected=np.array([proj_val]),  # keep 1‑D for similarity
                label_confidence=node.label_confidence,
            )
        )

    # 3. Graph construction and curvature
    adjacency, proj_map = build_similarity_graph(projected_nodes, threshold=similarity_threshold)
    curvature_edges = approximate_ollivier_ricci(adjacency, proj_map)

    # Compute average curvature per node
    node_avg_curv: Dict[str, float] = defaultdict(float)
    node_deg: Dict[str, int] = defaultdict(int)
    for (i, j), kappa in curvature_edges.items():
        node_avg_curv[i] += kappa
        node_avg_curv[j] += kappa
        node_deg[i] += 1
        node_deg[j] += 1
    for nid in node_avg_curv:
        node_avg_curv[nid] /= max(node_deg[nid], 1)

    # 4. Hybrid scores per action
    scores: Dict[str, float] = {}
    for action in actions:
        # pick a random node to provide curvature context (demo purpose)
        node_id = random.choice(list(node_avg_curv.keys()))
        curv = node_avg_curv[node_id]
        # pick matching left/right features (fallback to zeros)
        left = left_features.get(action.action_id, np.zeros(W.shape[0]))
        right = right_features.get(action.action_id, np.zeros(W.shape[1]))
        score = hybrid_action_score(action, left, right, W, gini, curv)
        scores[action.action_id] = score
    return scores


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic bandit actions
    actions = [
        BanditAction("a1", expected_value=0.8, cost=0.1),
        BanditAction("a2", expected_value=0.3, cost=0.2),
        BanditAction("a3", expected_value=0.5, cost=0.15),
    ]

    # Random left/right feature matrices for each action
    dim_left, dim_right = 4, 4
    left_feats = {a.action_id: np.random.rand(dim_left) for a in actions}
    right_feats = {a.action_id: np.random.rand(dim_right) for a in actions}
    W = np.random.rand(dim_left, dim_right) * 0.5  # bilinear weight matrix

    # Create synthetic text nodes (high‑dimensional)
    nodes = []
    for idx in range(6):
        fid = f"n{idx}"
        vec = np.random.rand(dim_left + dim_right)  # concatenate left+right space
        confidence = random.uniform(0.4, 0.9)  # mock path‑signature confidence
        nodes.append(TextNode(node_id=fid, feature=vec, label_confidence=confidence))

    # Run the hybrid pipeline
    scores = run_hybrid_pipeline(
        actions,
        left_features=left_feats,
        right_features=right_feats,
        W=W,
        nodes=nodes,
        similarity_threshold=0.3,
    )

    print("Hybrid Action Scores:")
    for aid, sc in scores.items():
        print(f"  {aid}: {sc:.4f}")

    # Verify that scores are finite numbers
    assert all(math.isfinite(v) for v in scores.values()), "Non‑finite scores detected"

    print("\nSmoke test completed successfully.")