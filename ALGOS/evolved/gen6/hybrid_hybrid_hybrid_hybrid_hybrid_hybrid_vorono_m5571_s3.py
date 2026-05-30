# DARWIN HAMMER — match 5571, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_ternary_lens__m2604_s1.py (gen4)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m841_s3.py (gen5)
# born: 2026-05-30T00:02:53Z

"""Hybrid Allocation‑Sheaf‑Voronoi‑RBF‑Audit‑Prune

Parents
-------
* **Parent A** – `hybrid_hybrid_hybrid_hybrid_hybrid_ternary_lens__m2604_s1.py`
  Provides weekday‑weighted allocation, builds a coboundary matrix Δ,
  computes the 1‑cochain residual `r = Δ s` and its Euclidean norm
  `‖r‖₂` (global risk factor).

* **Parent B** – `hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m841_s3.py`
  Supplies a Voronoi partition of points around seed centroids,
  Gaussian radial‑basis‑function (RBF) similarity from Euclidean distance,
  and a collection of per‑seed associative‑memory matrices (a *sheaf*).

Mathematical Bridge
-------------------
Both families operate on the same linear‑algebraic objects:

* The allocation vector `s ∈ ℝⁿ` lives in the same space as the seed‑centroid
  weights of the Voronoi sheaf: each group/seed corresponds to a row of the
  coboundary matrix Δ and to a memory matrix `M_i`.

* The residual norm `ρ = ‖Δ s‖₂` quantifies global inconsistency of the
  allocation.  We reuse `ρ` as a *scale* for the Gaussian RBF:


    w_i(query) = exp( - (‖q - c_i‖²) / (2·(ε·(1+β·ρ))²) )


  where `c_i` is seed *i*, `ε` a base width and `β` a coupling constant.
  Larger residual → broader RBF → more diffused memory readout,
  mirroring the aggressive pruning behaviour of Parent A.

* The same `ρ` also modulates the pruning probability of audit candidates:


    p_i(k) = min(1, (p₀/(1+k))·(1+γ·ρ) )


  (`γ` may differ from `β`.)

The module therefore fuses allocation/sheaf residuals with Voronoi‑RBF
retrieval and audit‑prune scheduling in a single unified pipeline.

"""

import datetime as dt
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – allocation & sheaf utilities
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1

def weekday_weight_vector(reference: dt.date = None) -> np.ndarray:
    """Return a 7‑element weight vector whose entries depend on the weekday.

    The weight for Monday is 1.0, each subsequent day is multiplied by 0.9.
    The vector is cyclically shifted so that the entry corresponding to
    ``reference.weekday()`` (Monday=0) is first.
    """
    if reference is None:
        reference = dt.date.today()
    base = np.array([1.0 * (0.9 ** i) for i in range(7)], dtype=float)
    shift = reference.weekday()
    return np.roll(base, -shift)


def build_incidence_matrix(num_groups: int, num_edges: int) -> np.ndarray:
    """Create a random coboundary matrix Δ ∈ {‑1,0,1}^{m×n}."""
    rng = np.random.default_rng(seed=42)
    Δ = rng.integers(-1, 2, size=(num_edges, num_groups))
    # Ensure each row has at least one non‑zero entry
    for i in range(num_edges):
        if not np.any(Δ[i]):
            Δ[i, rng.integers(0, num_groups)] = 1
    return Δ.astype(int)


def allocate_and_residual(demand: np.ndarray,
                          weight_vec: np.ndarray,
                          beta: float = 0.5) -> Tuple[np.ndarray, float]:
    """Allocate resources proportionally to ``demand`` weighted by weekdays.

    Returns the allocation vector ``s`` and the residual norm ``ρ = ‖Δ s‖₂``.
    """
    # Simple proportional allocation
    total = demand.sum()
    if total == 0:
        s = np.zeros_like(demand, dtype=float)
    else:
        s = demand * weight_vec[: len(demand)] / weight_vec[: len(demand)].sum()
        s *= total / s.sum()  # renormalise to original total

    # Build a modest coboundary matrix and compute residual
    Δ = build_incidence_matrix(num_groups=len(demand), num_edges=2 * len(demand))
    r = Δ @ s
    rho = np.linalg.norm(r, 2)
    # Optionally scale allocation by residual (demonstrates coupling)
    s = s * (1 + beta * rho)
    return s, rho


# ----------------------------------------------------------------------
# Parent B – Voronoi / RBF / associative‑memory utilities
# ----------------------------------------------------------------------
def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


def nearest(point: np.ndarray, seeds: np.ndarray) -> int:
    """Index of the closest seed to *point*."""
    if seeds.size == 0:
        raise ValueError("seeds required")
    return int(np.argmin(np.linalg.norm(seeds - point, axis=1)))


def assign(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Binary region matrix R (n_seeds × n_points)."""
    n_seeds = seeds.shape[0]
    n_pts = points.shape[0]
    regions = np.zeros((n_seeds, n_pts), dtype=int)
    for j, p in enumerate(points):
        i = nearest(p, seeds)
        regions[i, j] = 1
    return regions


def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian radial‑basis function."""
    return math.exp(-((epsilon * r) ** 2))


def rbf_weights(distances: np.ndarray,
                epsilon: float = 1.0,
                rho: float = 0.0,
                beta: float = 0.3) -> np.ndarray:
    """Compute Gaussian RBF weights modulated by the residual norm ρ.

    Effective width = ε·(1+β·ρ).
    """
    effective_eps = epsilon * (1.0 + beta * rho)
    # Vectorised Gaussian
    return np.exp(-((effective_eps * distances) ** 2))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_retrieve(query: np.ndarray,
                    seeds: np.ndarray,
                    sheaf: Dict[int, np.ndarray],
                    rho: float,
                    epsilon: float = 1.0,
                    beta: float = 0.3) -> np.ndarray:
    """Retrieve a vector from the Voronoi‑RBF‑Sheaf using the residual norm.

    Steps
    -----
    1. Compute Euclidean distances from ``query`` to each seed.
    2. Transform distances into RBF weights using ``rho`` (bridge).
    3. Weight each seed's memory matrix (a column vector) by its RBF weight.
    4. Return the weighted sum.
    """
    dists = np.linalg.norm(seeds - query, axis=1)
    weights = rbf_weights(dists, epsilon=epsilon, rho=rho, beta=beta)
    # Normalise weights to avoid scaling explosion
    if weights.sum() > 0:
        weights = weights / weights.sum()
    # Assemble retrieval
    result = np.zeros_like(next(iter(sheaf.values())))
    for i, w in enumerate(weights):
        mem_vec = sheaf.get(i)
        if mem_vec is not None:
            result += w * mem_vec
    return result


def audit_and_prune(candidates: List[Dict[str, Any]],
                    base_p0: float,
                    rho: float,
                    gamma: float = 0.2) -> List[Dict[str, Any]]:
    """Audit candidates and probabilistically prune them.

    Each candidate must contain an integer field ``'violations'``.
    The pruning probability for iteration ``k`` (starting at 0) is

        p(k) = min(1, (base_p0/(1+k)) * (1 + gamma * rho))

    The function runs a single pass (k = 0) and returns the survivors.
    """
    survivors = []
    k = 0
    for cand in candidates:
        violations = cand.get("violations", 0)
        # Simple risk factor: more violations → higher chance to be pruned
        risk_factor = 1 + violations
        p = min(1.0, (base_p0 / (1 + k)) * (1 + gamma * rho) * risk_factor)
        if random.random() > p:
            survivors.append(cand)
        k += 1
    return survivors


def full_hybrid_pipeline(demand: np.ndarray,
                         points: np.ndarray,
                         query: np.ndarray,
                         candidates: List[Dict[str, Any]],
                         epsilon: float = 1.0) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Execute the end‑to‑end hybrid algorithm.

    Returns
    -------
    * ``retrieved`` – vector obtained from the Voronoi‑RBF‑Sheaf.
    * ``pruned_candidates`` – list of candidates surviving the audit‑prune step.
    """
    # 1. Allocation & residual
    w_vec = weekday_weight_vector()
    allocation, rho = allocate_and_residual(demand, w_vec)

    # 2. Build seeds from allocation (one seed per group)
    #    For demonstration we embed the scalar allocation into a 3‑D seed.
    seeds = np.column_stack((
        np.arange(len(allocation)),
        allocation,
        np.full_like(allocation, fill_value=0.5)
    )).astype(float)

    # 3. Construct a trivial sheaf: each seed stores its allocation as a vector
    sheaf = {i: np.array([allocation[i], allocation[i] * 0.1, 1.0]) for i in range(len(allocation))}

    # 4. Retrieve using the query point
    retrieved = hybrid_retrieve(query, seeds, sheaf, rho, epsilon=epsilon)

    # 5. Audit & prune candidates
    pruned = audit_and_prune(candidates, base_p0=0.7, rho=rho)

    return retrieved, pruned


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic demand for four groups
    demand_vec = np.array([120, 80, 150, 50], dtype=float)

    # Random points representing lens candidates (for Voronoi)
    rng = np.random.default_rng(seed=123)
    points = rng.normal(loc=0.0, scale=1.0, size=(20, 3))

    # Query point (e.g., a new candidate's feature vector)
    query_pt = np.array([1.0, 0.5, -0.2])

    # Dummy audit candidates
    audit_candidates = [
        {"id": i, "violations": rng.integers(0, 4)} for i in range(10)
    ]

    retrieved_vec, survivors = full_hybrid_pipeline(
        demand=demand_vec,
        points=points,
        query=query_pt,
        candidates=audit_candidates,
        epsilon=0.8
    )

    print("Retrieved vector:", retrieved_vec)
    print("Surviving candidates:", survivors)