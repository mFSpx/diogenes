# DARWIN HAMMER — match 4747, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m482_s0.py (gen5)
# parent_b: hybrid_hybrid_ternary_route_hybrid_hybrid_ternar_m439_s0.py (gen4)
# born: 2026-05-29T23:58:06Z

import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

Point = Tuple[float, float]
Edge = Tuple[str, str]          # directed edge (parent, child)

# ----------------------------------------------------------------------
# Helper mathematics (parents)
# ----------------------------------------------------------------------
def euclidean(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, lik: float, fp: float) -> float:
    """P(E) = P(E|H)·P(H) + P(E|¬H)·P(¬H)."""
    return lik * prior + fp * (1.0 - prior)

def bayes_update(prior: float, lik: float, marginal: float) -> float:
    """Posterior P(H|E) = P(E|H)·P(H) / P(E)."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * lik / marginal

# ----------------------------------------------------------------------
# 1. Health‑aware Gini impurity (vectorised & numerically stable)
# ----------------------------------------------------------------------
def _health_weights(morphs: List[Morphology]) -> np.ndarray:
    """Return an array of health weights w_i = mass / (l·w·h) with safeguards."""
    vols = np.array(
        [max(m.length * m.width * m.height, 1e-12) for m in morphs],
        dtype=np.float64,
    )
    masses = np.array([m.mass for m in morphs], dtype=np.float64)
    w = masses / vols
    # Clip extreme values to avoid overflow in later sums
    w = np.clip(w, 0.0, 1e6)
    return w

def health_weighted_gini(labels: List[int], morphologies: List[Morphology]) -> float:
    """
    Compute a health‑weighted Gini impurity.

    w_i = mass_i / (length_i·width_i·height_i)
    G = 1 - Σ_k (p_k)²   where p_k = Σ_{i∈k} w_i / Σ_i w_i
    """
    if len(labels) != len(morphologies):
        raise ValueError("labels and morphologies must have same length")
    if not labels:                     # empty node → impurity 0
        return 0.0

    w = _health_weights(morphologies)
    total_w = w.sum()
    if total_w == 0.0:
        return 0.0

    # accumulate weighted counts per class
    label_arr = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(label_arr)
    weighted_counts = np.array(
        [w[label_arr == lbl].sum() for lbl in unique_labels],
        dtype=np.float64,
    )
    pk = weighted_counts / total_w
    gini = 1.0 - np.dot(pk, pk)
    return float(gini)

# ----------------------------------------------------------------------
# 2. Empirical‑Bernstein Hoeffding split decision (tighter bound)
# ----------------------------------------------------------------------
def _gain_variance(
    parent_labels: List[int],
    left_labels: List[int],
    right_labels: List[int],
    parent_morph: List[Morphology],
    left_morph: List[Morphology],
    right_morph: List[Morphology],
) -> float:
    """
    Estimate the variance of the Gini gain using the law of total variance.
    This is a cheap approximation that works with the health‑weighted Gini.
    """
    # compute Gini for each node
    g_parent = health_weighted_gini(parent_labels, parent_morph)
    g_left = health_weighted_gini(left_labels, left_morph)
    g_right = health_weighted_gini(right_labels, right_morph)

    n = len(parent_labels)
    n_l = len(left_labels)
    n_r = len(right_labels)

    # weighted gain for each sample (binary indicator of being in left/right)
    # variance ≈ Σ w_i (Δ_i - gain)² / (Σ w_i)²
    w_parent = _health_weights(parent_morph)
    total_w = w_parent.sum()
    if total_w == 0.0:
        return 0.0

    # contribution of each sample to the gain
    # Δ_i = g_parent - g_child (child depends on side)
    child_gini = np.empty_like(w_parent)
    child_gini[:n_l] = g_left
    child_gini[n_l:] = g_right
    side_weight = np.concatenate(
        [_health_weights(left_morph), _health_weights(right_morph)]
    )
    # weighted proportion of each side
    prop_left = n_l / n
    prop_right = n_r / n
    delta_i = g_parent - (prop_left * g_left + prop_right * g_right)
    # variance of a constant is zero; we use a tiny epsilon to avoid zero division
    variance = (w_parent * (delta_i - delta_i) ** 2).sum() / (total_w ** 2 + 1e-12)
    return float(variance)

def hoeffding_split_decision(
    parent_labels: List[int],
    left_labels: List[int],
    right_labels: List[int],
    parent_morph: List[Morphology],
    left_morph: List[Morphology],
    right_morph: List[Morphology],
    delta: float = 1e-7,
) -> Tuple[bool, float]:
    """
    Decide whether to split a node using an empirical‑Bernstein bound
    applied to the health‑weighted Gini gain.

    Returns (split, epsilon) where split is True if gain > ε.
    """
    n = len(parent_labels)
    if n == 0:
        return False, 0.0

    # compute weighted Gini values
    g_parent = health_weighted_gini(parent_labels, parent_morph)
    g_left = health_weighted_gini(left_labels, left_morph)
    g_right = health_weighted_gini(right_labels, right_morph)

    n_l = len(left_labels)
    n_r = len(right_labels)

    # weighted gain (parent impurity minus weighted child impurity)
    gain = g_parent - (n_l / n) * g_left - (n_r / n) * g_right

    # range of possible gain: Gini ∈ [0, 1] ⇒ gain ∈ [-1, 1]
    R = 1.0

    # empirical variance of the gain (cheap upper bound)
    var = _gain_variance(
        parent_labels, left_labels, right_labels,
        parent_morph, left_morph, right_morph,
    )

    # Empirical Bernstein bound
    epsilon = math.sqrt(2 * var * math.log(1.0 / delta) / n) + (3 * R * math.log(1.0 / delta) / n)

    return gain > epsilon, epsilon

# ----------------------------------------------------------------------
# 3. Bayesian ternary routing with deeper health integration
# ----------------------------------------------------------------------
def _directed_edge_key(a: str, b: str) -> Edge:
    """Canonical directed edge representation."""
    return (a, b)

def bayesian_ternary_route_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],                     # directed edges (parent → child)
    root: str,
    edge_priors: Dict[Edge, float],
    evidence: Dict[Edge, Tuple[float, float]],
    path_weight: float = 0.2,
    leaf_partitions: Optional[List[Tuple[List[int], List[Morphology]]]] = None,
) -> float:
    """
    Compute the cost of a ternary routing tree.

    1. Update each directed edge's prior with Bayesian evidence.
    2. Multiply the posterior by a health factor derived from the incident nodes.
    3. Material cost = Σ length(e) * posterior(e) * health_factor(e)
    4. Expected path cost = path_weight * Σ distance(root → node) using the same
       expected edge lengths.
    5. Optional Gini penalty = material_cost * average health‑weighted Gini of leaves.
    """
    # ------------------------------------------------------------------
    # 1. Posterior computation (per directed edge)
    # ------------------------------------------------------------------
    posteriors: Dict[Edge, float] = {}
    for a, b in edges:
        prior = edge_priors.get(_directed_edge_key(a, b), 0.5)
        lik, fp = evidence.get(_directed_edge_key(a, b), (0.5, 0.5))
        marginal = bayes_marginal(prior, lik, fp)
        post = bayes_update(prior, lik, marginal)
        posteriors[_directed_edge_key(a, b)] = post

    # ------------------------------------------------------------------
    # 2. Health factor for each edge (average of endpoint health weights)
    # ------------------------------------------------------------------
    node_health: Dict[str, float] = {}
    for nid, morph in nodes.items():
        # dummy Morphology for nodes without explicit geometry (fallback weight 1)
        if isinstance(morph, Morphology):
            vol = max(morph.length * morph.width * morph.height, 1e-12)
            node_health[nid] = min(morph.mass / vol, 1e6)
        else:   # type: ignore – nodes may be given as plain Point
            node_health[nid] = 1.0

    edge_health_factor: Dict[Edge, float] = {}
    for a, b in edges:
        hf = (node_health.get(a, 1.0) + node_health.get(b, 1.0)) / 2.0
        edge_health_factor[_directed_edge_key(a, b)] = hf

    # ------------------------------------------------------------------
    # 3. Material cost (expected length weighted by posterior & health)
    # ------------------------------------------------------------------
    material_cost = 0.0
    for a, b in edges:
        edge_len = euclidean(nodes[a], nodes[b])
        post = posteriors[_directed_edge_key(a, b)]
        hf = edge_health_factor[_directed_edge_key(a, b)]
        material_cost += edge_len * post * hf

    # ------------------------------------------------------------------
    # 4. Expected path lengths from root (DFS)
    # ------------------------------------------------------------------
    adjacency: Dict[str, List[Tuple[str, Edge]]] = {nid: [] for nid in nodes}
    for a, b in edges:
        adjacency[a].append((b, _directed_edge_key(a, b)))
        adjacency[b].append((a, _directed_edge_key(b, a)))   # reverse direction for traversal

    dist: Dict[str, float] = {root: 0.0}
    stack: List[str] = [root]
    visited: Set[str] = {root}

    while stack:
        cur = stack.pop()
        for nb, e_key in adjacency[cur]:
            if nb in visited:
                continue
            # Expected length uses the posterior of the direction we travel
            exp_len = euclidean(nodes[cur], nodes[nb]) * posteriors.get(e_key, 0.5) * edge_health_factor.get(e_key, 1.0)
            dist[nb] = dist[cur] + exp_len
            visited.add(nb)
            stack.append(nb)

    expected_path_cost = path_weight * sum(dist.values())

    # ------------------------------------------------------------------
    # 5. Gini penalty from leaf partitions (if any)
    # ------------------------------------------------------------------
    gini_penalty = 0.0
    if leaf_partitions:
        leaf_ginis = [
            health_weighted_gini(lbls, morphs)
            for lbls, morphs in leaf_partitions
            if lbls and morphs
        ]
        if leaf_ginis:
            avg_leaf_gini = sum(leaf_ginis) / len(leaf_ginis)
            gini_penalty = material_cost * avg_leaf_gini

    return material_cost + expected_path_cost + gini_penalty

# ----------------------------------------------------------------------
# Example high‑level hybrid operation (now using the improved components)
# ----------------------------------------------------------------------
def hybrid_split_and_route(
    data: np.ndarray,
    labels: List[int],
    morphologies: List[Morphology],
    nodes: Dict[str, Point],
    edges: List[Edge],
    edge_priors: Dict[Edge, float],
    delta: float = 1e-7,
) -> Tuple[bool, float, float]:
    """
    Perform a single binary split using health‑weighted Gini + empirical‑Bernstein
    Hoeffding bound, then evaluate the ternary routing cost of the resulting tree.

    Returns:
        split_ok   – whether the split satisfies the bound,
        epsilon    – bound value used for the decision,
        total_cost – Bayesian ternary routing cost (including Gini penalty).
    """
    # Very naive split: choose a random feature and threshold for demonstration.
    # In practice this would be replaced by a proper search over candidate splits.
    if data.shape[0] == 0:
        return False, 0.0, 0.0

    feat_idx = np.random.randint(data.shape[1])
    threshold = np.median(data[:, feat_idx])

    left_mask = data[:, feat_idx] <= threshold
    right_mask = ~left_mask

    left_labels = [labels[i] for i, v in enumerate(left_mask) if v]
    right_labels = [labels[i] for i, v in enumerate(right_mask) if v]

    left_morph = [morphologies[i] for i, v in enumerate(left_mask) if v]
    right_morph = [morphologies[i] for i, v in enumerate(right_mask) if v]

    split_ok, epsilon = hoeffding_split_decision(
        parent_labels=labels,
        left_labels=left_labels,
        right_labels=right_labels,
        parent_morph=morphologies,
        left_morph=left_morph,
        right_morph=right_morph,
        delta=delta,
    )

    # Build leaf partitions for the cost function (only if split is accepted)
    leaf_partitions = None
    if split_ok:
        leaf_partitions = [
            (left_labels, left_morph),
            (right_labels, right_morph),
        ]

    total_cost = bayesian_ternary_route_cost(
        nodes=nodes,
        edges=edges,
        root=list(nodes.keys())[0],
        edge_priors=edge_priors,
        evidence={},                     # no evidence in this demo
        path_weight=0.2,
        leaf_partitions=leaf_partitions,
    )

    return split_ok, epsilon, total_cost