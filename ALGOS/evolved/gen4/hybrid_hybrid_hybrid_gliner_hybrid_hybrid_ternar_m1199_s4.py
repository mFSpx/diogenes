# DARWIN HAMMER — match 1199, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_gliner_zero_s_hybrid_krampus_brain_m30_s5.py (gen3)
# parent_b: hybrid_hybrid_ternary_route_hybrid_workshare_all_m171_s0.py (gen2)
# born: 2026-05-29T23:33:29Z

import numpy as np
import math
import random
import sys
from datetime import date, datetime, timezone
from pathlib import Path
import json
import os
import signal
import time
from typing import Tuple, Dict, List, Any

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
EPS = 1e-12                     # safeguard against division‑by‑zero
C1 = 0.01 ** 2                  # SSIM constants (can be tuned)
C2 = 0.03 ** 2

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places for human‑readable percentages."""
    return round(float(value), 6)

def _safe_std(arr: np.ndarray) -> float:
    """Standard deviation with a tiny epsilon to avoid zero‑variance pitfalls."""
    return max(np.std(arr), EPS)

# ----------------------------------------------------------------------
# Core mathematical building blocks
# ----------------------------------------------------------------------
def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two 1‑D vectors.
    This implementation is robust to zero variance by injecting a small epsilon.
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)

    sigma_x = _safe_std(x)
    sigma_y = _safe_std(y)

    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2)

    return float(numerator / denominator)

def compute_group_similarities(
    prototype: np.ndarray,
    group_prototypes: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Compute a per‑group SSIM score between a global prototype and each group's
    prototype vector. The scores are normalised to sum to 1.
    """
    raw_scores = {
        name: compute_ssim(prototype, vec)
        for name, vec in group_prototypes.items()
    }
    total = sum(raw_scores.values()) + EPS
    return {name: score / total for name, score in raw_scores.items()}

def prim_mst(cost_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Return a Minimum Spanning Tree (MST) of a fully connected graph using Prim's
    algorithm. The graph is defined by a symmetric cost matrix where
    cost_matrix[i, j] = cost(i, j). The result is a list of edges (u, v, cost).
    """
    n = cost_matrix.shape[0]
    in_tree = [False] * n
    min_edge = [(math.inf, -1)] * n  # (cost, parent)
    min_edge[0] = (0.0, -1)

    mst_edges: List[Tuple[int, int, float]] = []

    for _ in range(n):
        # pick the vertex with the smallest connecting edge
        u = min((i for i in range(n) if not in_tree[i]), key=lambda i: min_edge[i][0])
        in_tree[u] = True
        parent, cost = min_edge[u][1], min_edge[u][0]
        if parent != -1:
            mst_edges.append((parent, u, cost))

        # update the cheapest edges to the remaining vertices
        for v in range(n):
            if not in_tree[v]:
                w = cost_matrix[u, v]
                if w < min_edge[v][0]:
                    min_edge[v] = (w, u)

    return mst_edges

# ----------------------------------------------------------------------
# Allocation strategies
# ----------------------------------------------------------------------
def allocate_workshare_ssim(
    x: np.ndarray,
    y: np.ndarray,
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
    group_prototypes: Dict[str, np.ndarray] | None = None,
) -> Dict[str, Any]:
    """
    Allocate work units among groups using a *global* SSIM score.
    This is a thin wrapper kept for backward compatibility.
    """
    ssim = compute_ssim(x, y)
    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    per_group = llm_units / len(groups)

    lanes = [
        {
            "group": g,
            "llm_units": _pct(per_group * ssim),
            "llm_share_pct": _pct(100.0 / len(groups)),
            "proof_required": True,
        }
        for g in groups
    ]

    return {
        "total_units": _pct(total_units),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "lanes": lanes,
    }

def allocate_workshare_day(
    target_weekday: int,
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
) -> Dict[str, Any]:
    """
    Allocate work units purely based on the current weekday.
    If the current day does not match ``target_weekday`` an empty allocation
    (zero deterministic units) is returned instead of an error.
    """
    today = date.today().weekday()
    deterministic_units = (
        total_units * deterministic_target_pct / 100.0
        if today == target_weekday
        else 0.0
    )
    llm_units = total_units - deterministic_units
    per_group = llm_units / len(groups)

    lanes = [
        {
            "group": g,
            "llm_units": _pct(per_group),
            "llm_share_pct": _pct(100.0 / len(groups)),
            "proof_required": True,
        }
        for g in groups
    ]

    return {
        "total_units": _pct(total_units),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "lanes": lanes,
        "weekday_matched": today == target_weekday,
    }

def allocate_workshare_hybrid(
    x: np.ndarray,
    y: np.ndarray,
    day: int,
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
    group_prototypes: Dict[str, np.ndarray] | None = None,
) -> Dict[str, Any]:
    """
    Deep hybrid allocation:
      • Deterministic units are released only on the specified ``day``.
      • LLM units are split among groups proportionally to the *per‑group*
        SSIM similarity between the global prototype (computed from ``x`` and ``y``)
        and each group's own prototype vector.
      • An MST built from the pairwise dissimilarities (1‑SSIM) of group prototypes
        is exposed for downstream routing decisions, illustrating a true fusion of
        “allocation” (SSIM) and “minimum‑cost tree” concepts.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Compute the global prototype (simple average of the two inputs)
    # ------------------------------------------------------------------
    prototype = (x.astype(float) + y.astype(float)) / 2.0

    # ------------------------------------------------------------------
    # 2️⃣ Per‑group similarity weights
    # ------------------------------------------------------------------
    if group_prototypes is None:
        # Fallback: use the same prototype for every group (behaviour of the
        # original implementation) – this keeps the function usable without
        # extra data.
        group_prototypes = {g: prototype.copy() for g in groups}

    group_weights = compute_group_similarities(prototype, group_prototypes)

    # ------------------------------------------------------------------
    # 3️⃣ Deterministic vs. LLM split based on the day
    # ------------------------------------------------------------------
    today = date.today().weekday()
    deterministic_units = (
        total_units * deterministic_target_pct / 100.0
        if today == day
        else 0.0
    )
    llm_units = total_units - deterministic_units

    # ------------------------------------------------------------------
    # 4️⃣ Allocate LLM units according to similarity weights
    # ------------------------------------------------------------------
    lanes = []
    for g in groups:
        weight = group_weights.get(g, 0.0)
        allocated = llm_units * weight
        lanes.append(
            {
                "group": g,
                "llm_units": _pct(allocated),
                "llm_share_pct": _pct(weight * 100.0),
                "proof_required": True,
                "similarity": _pct(weight * 100.0),
            }
        )

    # ------------------------------------------------------------------
    # 5️⃣ Build a Minimum Spanning Tree over groups (optional but demonstrates
    #    the deeper integration of the “minimum cost tree” idea)
    # ------------------------------------------------------------------
    n = len(groups)
    cost_mat = np.zeros((n, n))
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            if i == j:
                continue
            # Dissimilarity = 1 - SSIM between the two group prototypes
            cost_mat[i, j] = 1.0 - compute_ssim(group_prototypes[gi], group_prototypes[gj])

    mst_edges = prim_mst(cost_mat)

    mst_representation = [
        {"from": groups[u], "to": groups[v], "cost": _pct(cost)} for u, v, cost in mst_edges
    ]

    # ------------------------------------------------------------------
    # 6️⃣ Assemble final payload
    # ------------------------------------------------------------------
    return {
        "total_units": _pct(total_units),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "lanes": lanes,
        "mst": mst_representation,
        "weekday_matched": today == day,
    }

# ----------------------------------------------------------------------
# Simple demo when run as a script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example vectors
    x = np.array([1, 2, 3], dtype=float)
    y = np.array([4, 5, 6], dtype=float)

    # Example per‑group prototypes (randomly generated for demo)
    rng = np.random.default_rng(42)
    group_protos = {
        g: rng.random(3) for g in GROUPS
    }

    day = 3                     # Wednesday (0 = Monday)
    total_units = 100.0

    print("=== SSIM‑only allocation ===")
    print(allocate_workshare_ssim(x, y, total_units=total_units))

    print("\n=== Day‑only allocation ===")
    print(allocate_workshare_day(day, total_units=total_units))

    print("\n=== Deep hybrid allocation ===")
    print(
        allocate_workshare_hybrid(
            x,
            y,
            day,
            total_units=total_units,
            group_prototypes=group_protos,
        )
    )