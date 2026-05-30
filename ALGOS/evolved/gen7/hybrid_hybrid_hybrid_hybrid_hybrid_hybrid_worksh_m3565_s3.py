# DARWIN HAMMER — match 3565, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_sheaf__hybrid_hybrid_hybrid_m2030_s3.py (gen6)
# parent_b: hybrid_hybrid_workshare_all_hybrid_krampus_brain_m23_s4.py (gen2)
# born: 2026-05-29T23:50:44Z

"""Hybrid Sheaf‑Cohomology & Workshare Allocation Engine
=====================================================

Parents
-------
* **Parent A** – *hybrid_hybrid_hybrid_sheaf__hybrid_hybrid_hybrid_m2030_s3.py*  
  Builds a coboundary matrix Δ from a graph, weighting each edge by the
  average *regret‑weight* of its incident nodes and then prunes edges with
  an exponential probability `p(t)=λ·exp(-α·t)`.  The null‑space dimension
  `dim ker(Δ)` reflects topological connectivity together with decision‑theoretic
  information.

* **Parent B** – *hybrid_hybrid_workshare_all_hybrid_krampus_brain_m23_s4.py*  
  Computes a deterministic work‑share factor from the Doomsday weekday,
  distributes a residual share among model groups using a normalized
  24‑dimensional feature vector, and optionally adjusts the deterministic
  target with a curvature scalar derived from the outer‑product of the
  feature vector.

Mathematical Bridge
-------------------
Both parents operate on **linear objects** (a matrix or a resource vector)
scaled by a **probability‑like weight vector** that sums to one:

* Parent A → *regret‑weight vector* `w_r ∈ Δⁿ` (node‑wise).
* Parent B → *feature‑weight vector* `w_f ∈ Δᵐ` (group‑wise).

The hybrid algorithm fuses these two weight families by **tensor‑multiplying**
them onto the edges of the sheaf graph:


edge_weight(e = (i, j, g)) = ½ (w_r[i] + w_r[j]) · w_f[g]


The resulting weighted coboundary matrix `Δ̂` is then pruned exactly as in
Parent A.  Its null‑space dimension `ν = dim ker(Δ̂)` is used as a *topological
confidence* scalar that modulates the deterministic Doomsday factor from
Parent B.  The curvature scalar `κ = trace(F·Fᵀ) = ‖F‖²` (with `F` the raw
24‑dimensional feature vector) provides an additional epistemic adjustment.

The final allocation for each model group `g` is


deterministic   = total_units * (1 + doomsday/7) * (1 + κ·ν̂)
residual_share  = total_units - deterministic
allocation[g]   = deterministic·γ_g + residual_share·w_f[g]


where `γ_g` is a normalized group‑specific factor derived from the
edge‑wise contributions of `Δ̂`.  This yields a single unified system that
captures **topology**, **decision‑theoretic regret**, **feature‑driven
probabilities**, and **curvature‑based epistemic confidence**.

The module below implements the hybrid pipeline with clear, testable
functions.
"""

import math
import random
import sys
from pathlib import Path
from datetime import date
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Decision‑theoretic data structures (Parent A)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
_EPISTEMIC_FACTOR = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.2,
    "SURE_MAYBE": 0.6,
}


class Action:
    """Action with cost, baseline probability, and epistemic certainty."""

    __slots__ = ("cost", "probability", "epistemic_certainty")

    def __init__(self, cost: float, probability: float, epistemic_certainty: str):
        if epistemic_certainty not in EPISTEMIC_FLAGS:
            raise ValueError(f"Invalid epistemic flag: {epistemic_certainty}")
        self.cost = float(cost)
        self.probability = float(probability)
        self.epistemic_certainty = epistemic_certainty

    def epistemic_factor(self) -> float:
        """Map certainty flag to a scalar in [0,1]."""
        return _EPISTEMIC_FACTOR[self.epistemic_certainty]


# ----------------------------------------------------------------------
# Feature‑driven structures (Parent B)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return the Doomsday weekday index for a given Gregorian date.
    Monday → 0, …, Sunday → 6.
    """
    return (date(year, month, day).weekday() + 1) % 7


# ----------------------------------------------------------------------
# Hybrid Core
# ----------------------------------------------------------------------


def compute_regret_weights(actions: List[Action]) -> Dict[str, float]:
    """
    Produce a normalized regret‑weight for each distinct node identifier.

    The node identifier is derived from the string representation of an
    action (``str(action)``) – this keeps the function agnostic to the
    surrounding graph construction.

    Returns
    -------
    dict
        Mapping ``node_id -> weight`` with ``∑ weight == 1``.
    """
    raw = {}
    for act in actions:
        nid = str(act)  # deterministic identifier
        # Regret contribution = cost * (1 - probability) * epistemic factor
        regret = act.cost * (1.0 - act.probability) * act.epistemic_factor()
        raw[nid] = raw.get(nid, 0.0) + regret

    total = sum(raw.values())
    if total == 0.0:
        # Avoid division by zero – fallback to uniform distribution
        n = len(raw) or 1
        return {k: 1.0 / n for k in raw}
    return {k: v / total for k, v in raw.items()}


def compute_feature_weights(feature_vec: np.ndarray) -> Dict[str, float]:
    """
    Convert a 24‑dimensional raw feature vector into a probability
    distribution over the four model groups.

    The first six entries drive ``codex``, the next six ``groq``,
    then ``cohere`` and finally ``local_models``.
    """
    if feature_vec.shape != (24,):
        raise ValueError("feature_vec must be a 24‑dimensional vector")
    # Sum each 6‑element block
    block_sums = np.array([
        feature_vec[0:6].sum(),
        feature_vec[6:12].sum(),
        feature_vec[12:18].sum(),
        feature_vec[18:24].sum(),
    ])
    total = block_sums.sum()
    if total == 0.0:
        # Uniform fallback
        return {g: 0.25 for g in GROUPS}
    probs = block_sums / total
    return {g: float(p) for g, p in zip(GROUPS, probs)}


def prune_edges(
    edges: List[Tuple[str, str, str]],
    lam: float,
    alpha: float,
    steps: int,
) -> List[Tuple[str, str, str]]:
    """
    Stochastically prune edges according to the exponential schedule
    `p(t)=λ·exp(-α·t)`.  For each step we independently decide to keep an edge
    with probability `1 - p(t)`.  The function returns the edge list that
    survived *all* steps.

    Parameters
    ----------
    edges
        List of `(node_i, node_j, group)` triples.
    lam, alpha
        Schedule parameters (both non‑negative).
    steps
        Number of pruning iterations.

    Returns
    -------
    list
        Retained edges.
    """
    retained = edges.copy()
    for t in range(steps):
        prob = lam * math.exp(-alpha * t)
        prob = min(max(prob, 0.0), 1.0)  # clamp to [0,1]
        retained = [
            e for e in retained if random.random() > prob
        ]
        if not retained:
            break
    return retained


def build_hybrid_sheaf(
    nodes: List[str],
    edges: List[Tuple[str, str, str]],
    regret_weights: Dict[str, float],
    feature_weights: Dict[str, float],
    lam: float = 0.3,
    alpha: float = 0.05,
    prune_steps: int = 5,
) -> Tuple[np.ndarray, List[Tuple[str, str, str]]]:
    """
    Construct the weighted coboundary matrix Δ̂.

    For each edge `e = (i, j, g)` we compute

        w_e = ½ (w_r[i] + w_r[j]) · w_f[g]

    where `w_r` are regret weights and `w_f` are feature weights.
    The matrix has shape `(|E_ret|, |V|)` with entries
    `Δ̂[e, i] =  w_e` and `Δ̂[e, j] = -w_e`.

    The edge set is first pruned using `prune_edges`.
    """
    # Ensure every node appears in regret_weights; missing nodes get zero weight.
    for n in nodes:
        regret_weights.setdefault(n, 0.0)

    pruned = prune_edges(edges, lam, alpha, prune_steps)

    node_index = {n: idx for idx, n in enumerate(nodes)}
    m = len(pruned)
    n = len(nodes)
    delta = np.zeros((m, n), dtype=float)

    for row, (i, j, g) in enumerate(pruned):
        w_r_i = regret_weights.get(i, 0.0)
        w_r_j = regret_weights.get(j, 0.0)
        w_f_g = feature_weights.get(g, 0.0)
        w_edge = 0.5 * (w_r_i + w_r_j) * w_f_g
        delta[row, node_index[i]] = w_edge
        delta[row, node_index[j]] = -w_edge

    return delta, pruned


def hybrid_nullspace_dimension(delta: np.ndarray, eps: float = 1e-10) -> int:
    """
    Compute `dim ker(Δ̂)` via singular‑value decomposition.
    Singular values smaller than `eps` are treated as zero.
    """
    if delta.size == 0:
        # Empty matrix → full rank = 0, nullspace = number of columns
        return delta.shape[1]
    s = np.linalg.svd(delta, compute_uv=False)
    nullity = np.sum(s < eps)
    return int(nullity)


def curvature_scalar(feature_vec: np.ndarray) -> float:
    """
    Curvature metric defined as the trace of the outer product `F·Fᵀ`,
    which equals the squared Euclidean norm of the feature vector.
    """
    return float(np.linalg.norm(feature_vec) ** 2)


def allocate_resources_hybrid(
    total_units: float,
    date_tuple: Tuple[int, int, int],
    delta: np.ndarray,
    feature_vec: np.ndarray,
    group_edge_map: Dict[str, List[int]],
) -> Dict[str, float]:
    """
    Produce a final allocation per model group.

    Steps
    -----
    1. Compute topological confidence `ν = dim ker(Δ̂)`.
    2. Compute curvature `κ = ‖F‖²`.
    3. Deterministic share = `total_units * (1 + d/7) * (1 + κ·ν̂)`,
       where `d` is the Doomsday weekday index and `ν̂ = ν / max(ν,1)`.
    4. Residual share = `total_units - deterministic`.
    5. Distribute residual proportionally to the *feature weight* vector.
    6. Additionally, spread the deterministic share among groups according to
       the total absolute edge weight contributed by each group in Δ̂.

    Returns
    -------
    dict
        Mapping ``group -> allocated units`` (sums to `total_units` within
        floating‑point tolerance).
    """
    year, month, day = date_tuple
    d = doomsday(year, month, day)

    # 1. Topological confidence
    nu = hybrid_nullspace_dimension(delta)
    nu_hat = nu / max(nu, 1)

    # 2. Curvature
    kappa = curvature_scalar(feature_vec)

    # 3. Deterministic component
    deterministic = total_units * (1.0 + d / 7.0) * (1.0 + kappa * nu_hat)

    deterministic = min(deterministic, total_units)  # cannot exceed total
    residual = total_units - deterministic

    # 4. Feature‑driven residual distribution
    raw_feat = feature_vec.reshape(4, 6).sum(axis=1)  # 4 groups
    feat_weights = raw_feat / raw_feat.sum()
    residual_alloc = {g: residual * w for g, w in zip(GROUPS, feat_weights)}

    # 5. Deterministic share split via edge‑weight contribution
    # Compute total absolute edge weight per group from Δ̂
    group_weights = {g: 0.0 for g in GROUPS}
    for row, (i, j, g) in enumerate(group_edge_map["edges"]):
        w = abs(delta[row, :]).sum()  # sum of absolute entries in the row
        group_weights[g] += w
    total_gw = sum(group_weights.values())
    if total_gw == 0.0:
        # fallback to uniform split
        det_alloc = {g: deterministic / len(GROUPS) for g in GROUPS}
    else:
        det_alloc = {
            g: deterministic * (group_weights[g] / total_gw) for g in GROUPS
        }

    # 6. Combine
    final_alloc = {g: det_alloc[g] + residual_alloc[g] for g in GROUPS}
    # Normalise tiny rounding errors
    scale = total_units / sum(final_alloc.values())
    final_alloc = {g: v * scale for g, v in final_alloc.items()}
    return final_alloc


# ----------------------------------------------------------------------
# Utility wrappers for the hybrid pipeline
# ----------------------------------------------------------------------


def hybrid_pipeline(
    actions: List[Action],
    nodes: List[str],
    edges: List[Tuple[str, str, str]],
    feature_vec: np.ndarray,
    total_units: float,
    date_tuple: Tuple[int, int, int],
    lam: float = 0.3,
    alpha: float = 0.05,
    prune_steps: int = 5,
) -> Dict[str, float]:
    """
    End‑to‑end execution:

    1. Compute regret and feature weights.
    2. Build the hybrid sheaf matrix and prune it.
    3. Allocate resources based on the resulting topology and curvature.

    Returns a mapping ``group -> allocated units``.
    """
    regret_w = compute_regret_weights(actions)
    feature_w = compute_feature_weights(feature_vec)

    delta, retained_edges = build_hybrid_sheaf(
        nodes,
        edges,
        regret_w,
        feature_w,
        lam=lam,
        alpha=alpha,
        prune_steps=prune_steps,
    )

    # Package edge information needed for