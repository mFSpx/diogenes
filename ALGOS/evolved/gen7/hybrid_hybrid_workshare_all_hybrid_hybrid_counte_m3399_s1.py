# DARWIN HAMMER — match 3399, survivor 1
# gen: 7
# parent_a: hybrid_workshare_allocator_doomsday_calendar_m14_s1.py (gen1)
# parent_b: hybrid_hybrid_counterfactua_hybrid_hybrid_endpoi_m1283_s1.py (gen6)
# born: 2026-05-29T23:49:49Z

"""
Hybrid Workshare-Graph Allocator

Parents:
- hybrid_workshare_allocator_doomsday_calendar_m14_s1.py (Algorithm A)
- hybrid_hybrid_counterfactua_hybrid_hybrid_endpoi_m1283_s1.py (Algorithm B)

Mathematical Bridge:
Algorithm A provides a deterministic workshare split that is modulated by the
weekday (doomsday) factor:  

    deterministic_units = total_units * deterministic_pct/100 * (1 + d/7)

where d ∈ {0,…,6} is the weekday index.

Algorithm B treats information as nodes in a graph and defines a similarity
kernel based on Euclidean distance with a Gaussian weighting:

    s_ij = exp( - (ε * ‖v_i - v_j‖)^2 )

We fuse the two by constructing a graph whose nodes are the *groups* from
Algorithm A.  Each node carries a 2‑dimensional allocation vector
v_i = [deterministic_share_i, llm_share_i].  The doomsday‑scaled deterministic
share determines the magnitude of the first component, while the second
component is the residual LLM allocation.  Pairwise similarities are computed
with the Gaussian kernel from Algorithm B, yielding a similarity matrix that
can be used for entropy‑based uncertainty measurement and for a diffusion‑style
adjustment of the original allocation.

The resulting hybrid system simultaneously respects the calendar‑driven
deterministic scaling and the graph‑based uncertainty quantification.
"""

import sys
import random
import math
import pathlib
from datetime import date
from dataclasses import dataclass
from typing import Sequence, Tuple, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Core utilities (from Algorithm A)
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return the weekday index used by the original doomsday calendar:
    Monday → 1, ..., Sunday → 0 (mod 7).
    """
    return (date(year, month, day).weekday() + 1) % 7


def allocate_workshare_with_doomsday(
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models"),
    year: int = date.today().year,
    month: int = date.today().month,
    day: int = date.today().day,
) -> Dict[str, Dict[str, float]]:
    """
    Compute a deterministic vs. LLM split for each group.
    The deterministic portion is scaled by the weekday factor.
    Returns a dict mapping each group to its allocation vector:
        {
            "deterministic": ...,
            "llm": ...,
        }
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0 <= deterministic_target_pct <= 100):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("at least one group required")

    d_factor = doomsday(year, month, day)
    # Scale deterministic portion by (1 + d/7) as in the parent algorithm
    deterministic_units = total_units * deterministic_target_pct / 100.0 * (1 + d_factor / 7)
    llm_units = total_units - deterministic_units

    # Distribute each portion equally among groups (simple uniform split)
    det_per_group = deterministic_units / len(groups)
    llm_per_group = llm_units / len(groups)

    allocation: Dict[str, Dict[str, float]] = {}
    for grp in groups:
        allocation[grp] = {
            "deterministic": _pct(det_per_group),
            "llm": _pct(llm_per_group),
        }
    return allocation


# ----------------------------------------------------------------------
# Graph‑based similarity utilities (from Algorithm B)
# ----------------------------------------------------------------------
Vector = Sequence[float]


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a_arr - b_arr))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel used as similarity measure."""
    return math.exp(-((epsilon * r) ** 2))


def build_similarity_matrix(
    allocation: Dict[str, Dict[str, float]],
    epsilon: float = 1.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Construct a symmetric similarity matrix S where
        S_ij = exp( - (ε * ||v_i - v_j||)^2 )
    with v_i = [deterministic_i, llm_i].

    Returns the matrix and the ordered list of group names.
    """
    groups = list(allocation.keys())
    n = len(groups)
    S = np.zeros((n, n), dtype=float)

    vectors = [np.array([allocation[g]["deterministic"], allocation[g]["llm"]]) for g in groups]

    for i in range(n):
        for j in range(i, n):
            dist = euclidean(vectors[i], vectors[j])
            sim = gaussian(dist, epsilon)
            S[i, j] = sim
            S[j, i] = sim  # symmetry
    return S, groups


def rowwise_entropy(matrix: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy for each row of a probability matrix.
    Rows are first normalized to sum to 1 (handling zero rows safely).
    """
    # Protect against division by zero
    row_sums = matrix.sum(axis=1, keepdims=True)
    # If a row sums to zero, replace with uniform distribution
    safe_sums = np.where(row_sums == 0, 1.0, row_sums)
    prob = matrix / safe_sums

    # Entropy: - Σ p * log(p)  (base e)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log(prob, where=(prob > 0))
    entropy = -np.sum(prob * log_p, axis=1)
    return entropy


def compute_entropy_per_group(
    similarity_matrix: np.ndarray,
    groups: List[str],
) -> Dict[str, float]:
    """
    Return a mapping group → entropy value derived from the similarity matrix.
    """
    entropies = rowwise_entropy(similarity_matrix)
    return {grp: float(ent) for grp, ent in zip(groups, entropies)}


# ----------------------------------------------------------------------
# Hybrid adjustment using entropy (new hybrid operation)
# ----------------------------------------------------------------------
def adjust_allocation_by_entropy(
    allocation: Dict[str, Dict[str, float]],
    entropies: Dict[str, float],
    entropy_weight: float = 0.2,
) -> Dict[str, Dict[str, float]]:
    """
    Reduce the LLM share of groups with high entropy (high uncertainty)
    and increase their deterministic share proportionally.
    The adjustment respects the total_units invariant.

    Parameters
    ----------
    allocation : dict
        Original allocation per group.
    entropies : dict
        Entropy per group (higher → more uncertainty).
    entropy_weight : float
        Scaling factor for the adjustment (0 ≤ weight ≤ 1).

    Returns
    -------
    dict
        New allocation dict with the same total units.
    """
    if not (0 <= entropy_weight <= 1):
        raise ValueError("entropy_weight must be between 0 and 1")

    groups = list(allocation.keys())
    total_det = sum(v["deterministic"] for v in allocation.values())
    total_llm = sum(v["llm"] for v in allocation.values())
    total_units = total_det + total_llm

    # Normalise entropies to [0,1]
    max_ent = max(entropies.values()) if entropies else 1.0
    min_ent = min(entropies.values()) if entropies else 0.0
    range_ent = max_ent - min_ent if max_ent != min_ent else 1.0
    norm_ent = {g: (e - min_ent) / range_ent for g, e in entropies.items()}

    new_allocation: Dict[str, Dict[str, float]] = {}
    for g in groups:
        e = norm_ent[g]
        # Transfer a fraction of LLM units to deterministic based on entropy
        transfer = entropy_weight * e * allocation[g]["llm"]
        new_det = allocation[g]["deterministic"] + transfer
        new_llm = allocation[g]["llm"] - transfer
        new_allocation[g] = {
            "deterministic": _pct(new_det),
            "llm": _pct(new_llm),
        }

    # Re‑normalize to preserve the global total (tiny rounding errors may appear)
    corrected_total = sum(v["deterministic"] + v["llm"] for v in new_allocation.values())
    correction_factor = total_units / corrected_total if corrected_total != 0 else 1.0

    for g in groups:
        new_allocation[g]["deterministic"] = _pct(new_allocation[g]["deterministic"] * correction_factor)
        new_allocation[g]["llm"] = _pct(new_allocation[g]["llm"] * correction_factor)

    return new_allocation


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    TOTAL_UNITS = 1000.0
    DETERMINISTIC_PCT = 85.0
    GROUPS = ("codex", "groq", "cohere", "local_models", "anthropic")

    # Step 1: Base allocation with doomsday factor
    base_alloc = allocate_workshare_with_doomsday(
        total_units=TOTAL_UNITS,
        deterministic_target_pct=DETERMINISTIC_PCT,
        groups=GROUPS,
        year=2026,
        month=5,
        day=29,
    )
    print("Base allocation:")
    for g, vals in base_alloc.items():
        print(f"  {g}: deterministic={vals['deterministic']}, llm={vals['llm']}")

    # Step 2: Build similarity matrix from allocation vectors
    S, order = build_similarity_matrix(base_alloc, epsilon=0.5)
    print("\nSimilarity matrix (ordered groups):")
    print(order)
    print(np.round(S, 4))

    # Step 3: Compute entropy per group
    ent = compute_entropy_per_group(S, order)
    print("\nEntropy per group:")
    for g in order:
        print(f"  {g}: {ent[g]:.4f}")

    # Step 4: Adjust allocation based on entropy
    adjusted = adjust_allocation_by_entropy(base_alloc, ent, entropy_weight=0.3)
    print("\nAdjusted allocation:")
    for g, vals in adjusted.items():
        print(f"  {g}: deterministic={vals['deterministic']}, llm={vals['llm']}")

    # Verify total units conserved (within rounding tolerance)
    total_before = sum(v["deterministic"] + v["llm"] for v in base_alloc.values())
    total_after = sum(v["deterministic"] + v["llm"] for v in adjusted.values())
    print(f"\nTotal units before: {total_before:.6f}, after: {total_after:.6f}")
    assert abs(total_before - total_after) < 1e-3, "Total units not conserved"
    print("Smoke test completed successfully.")