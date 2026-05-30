# DARWIN HAMMER — match 1820, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_distributed_leader_e_m730_s1.py (gen5)
# parent_b: hybrid_hybrid_privacy_model_hybrid_hybrid_fisher_m33_s0.py (gen4)
# born: 2026-05-29T23:38:59Z

"""Hybrid Algorithm integrating:
- Parent A: ternary‑vector based maximal independent set leader election.
- Parent B: Fisher score weighted privacy risk and SSIM‑adjusted resource matrix.

Mathematical bridge:
The ternary vector **t** (∈{-1,0,1}ⁿ) generated from the command payload is
used as a *structural signature* for each node.  In Parent B the Fisher
score **F(θ)** provides a data‑driven weighting factor.  The hybrid
algorithm multiplies the broadcast probability of the MIS algorithm by a
Fisher‑derived factor **(1+F/κ)** (κ is a normalising constant) and also
uses the ternary signature to weight the SSIM similarity between a model
resource matrix **A** and the signature.  Thus the two topologies are
fused through a common scalar weighting that simultaneously respects the
graph‑based leader election and the statistical privacy/model‑resource
considerations.
"""

import argparse
import collections
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

# -------------------------- Parent A components --------------------------

TERNARY_DIMS = 12  # dimension of ternary signature


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def payload_hash(raw_command: str, normalized_intent: str, context: Dict[str, Any]) -> str:
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ternary_vector(
    raw_command: str, normalized_intent: str, context: Dict[str, Any]
) -> np.ndarray:
    """Generate a deterministic ternary vector from payload data."""
    payload_hash_value = payload_hash(raw_command, normalized_intent, context)
    hash_int = int(payload_hash_value, 16)
    vec = np.zeros(TERNARY_DIMS, dtype=int)
    for i in range(TERNARY_DIMS):
        vec[i] = (hash_int % 3) - 1  # maps {0,1,2} → {-1,0,1}
        hash_int //= 3
    return vec


def base_broadcast_probability(phase: int, step: int, ternary_vec: np.ndarray) -> float:
    """Original probability from Parent A (without Fisher weighting)."""
    prob = min(1.0, 1.0 / (2 ** max(0, phase - step)))
    prob *= 1 + np.sum(ternary_vec) / TERNARY_DIMS
    return prob


def maximal_independent_set(
    graph: Dict[Any, Set[Any]],
    phases: int = 8,
    seed: int | str | None = None,
    ternary_vec: np.ndarray | None = None,
    fisher_factor: float = 1.0,
) -> Set[Any]:
    """MIS algorithm where broadcast probability is scaled by fisher_factor."""
    rng = random.Random(seed)
    undecided: Set[Any] = set(graph)
    leaders: Set[Any] = set()
    blocked: Set[Any] = set()

    if ternary_vec is None:
        ternary_vec = np.random.randint(-1, 2, size=TERNARY_DIMS)

    for phase in range(1, phases + 1):
        if not undecided:
            break
        p = base_broadcast_probability(phases, phase, ternary_vec) * fisher_factor
        p = min(1.0, p)  # keep a valid probability
        broadcasts = {n for n in undecided if rng.random() < p}
        new_leaders = {n for n in broadcasts if not (graph.get(n, set()) & broadcasts)}
        leaders |= new_leaders
        if new_leaders:
            neigh = set().union(*(graph.get(n, set()) for n in new_leaders), new_leaders)
            blocked |= neigh
        undecided -= blocked

    # any remaining undecided nodes that are not adjacent to leaders become leaders
    for n in sorted(undecided, key=str):
        if not (graph.get(n, set()) & leaders):
            leaders.add(n)

    return leaders


# -------------------------- Parent B components --------------------------

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index between two 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Normalized reconstruction risk in [0,1]."""
    if total_records <= 0:
        return 0.0
    risk = unique_quasi_identifiers / total_records
    return max(0.0, min(1.0, risk))


# -------------------------- Hybrid Operations --------------------------

def hybrid_compute_ternary_signature(
    raw_command: str, normalized_intent: str, context: Dict[str, Any]
) -> np.ndarray:
    """Public wrapper that returns the ternary signature used throughout the hybrid."""
    return ternary_vector(raw_command, normalized_intent, context)


def hybrid_leader_election(
    graph: Dict[Any, Set[Any]],
    raw_command: str,
    normalized_intent: str,
    context: Dict[str, Any],
    theta: float,
    center: float,
    width: float,
    phases: int = 8,
    seed: int | str | None = None,
) -> Set[Any]:
    """
    Perform leader election on ``graph``.
    The broadcast probability of the MIS algorithm is scaled by a Fisher
    factor derived from (theta, center, width).  The ternary signature
    generated from the payload provides the structural term.
    """
    t_vec = ternary_vector(raw_command, normalized_intent, context)
    fisher = fisher_score(theta, center, width)
    # Normalise Fisher to a modest scaling factor to keep probabilities ≤1
    fisher_factor = 1 + fisher / (1.0 + fisher)  # maps to (1,2)
    return maximal_independent_set(
        graph,
        phases=phases,
        seed=seed,
        ternary_vec=t_vec,
        fisher_factor=fisher_factor,
    )


def hybrid_model_resource_matrix(
    base_matrix: np.ndarray,
    raw_command: str,
    normalized_intent: str,
    context: Dict[str, Any],
    theta: float,
    center: float,
    width: float,
) -> np.ndarray:
    """
    Adjust ``base_matrix`` using SSIM similarity between each column and the
    ternary signature.  The similarity weights are further scaled by the
    Fisher score, providing a data‑driven modulation.
    """
    if base_matrix.ndim != 2:
        raise ValueError("base_matrix must be 2‑D")
    t_vec = ternary_vector(raw_command, normalized_intent, context).astype(float)

    # Pad / truncate ternary vector to match column length
    col_len = base_matrix.shape[0]
    if t_vec.size < col_len:
        pad = np.zeros(col_len - t_vec.size)
        t_vec_padded = np.concatenate([t_vec, pad])
    else:
        t_vec_padded = t_vec[:col_len]

    fisher = fisher_score(theta, center, width)
    fisher_weight = 1 + fisher / (1.0 + fisher)  # ∈ (1,2)

    adjusted = np.empty_like(base_matrix, dtype=float)
    for j in range(base_matrix.shape[1]):
        col = base_matrix[:, j]
        similarity = ssim(col, t_vec_padded)
        weight = similarity * fisher_weight
        adjusted[:, j] = col * weight
    return adjusted


def hybrid_privacy_risk_vector(
    nodes: List[Any],
    unique_quasi_identifiers: int,
    total_records: int,
    raw_command: str,
    normalized_intent: str,
    context: Dict[str, Any],
    theta: float,
    center: float,
    width: float,
) -> Dict[Any, float]:
    """
    Produce a privacy‑risk score per node.
    The base risk is the reconstruction risk; it is multiplied by a factor
    that combines the ternary signature magnitude and the Fisher score.
    """
    base_risk = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    t_vec = ternary_vector(raw_command, normalized_intent, context)
    ternary_factor = 1 + np.abs(np.sum(t_vec)) / TERNARY_DIMS  # in [0,2]
    fisher = fisher_score(theta, center, width)
    fisher_factor = 1 + fisher / (1.0 + fisher)  # in (1,2)

    overall_factor = ternary_factor * fisher_factor
    node_risk = {n: min(1.0, base_risk * overall_factor) for n in nodes}
    return node_risk


# -------------------------- Smoke Test --------------------------

if __name__ == "__main__":
    # Minimal graph for leader election
    example_graph = {
        "A": {"B", "C"},
        "B": {"A", "D"},
        "C": {"A"},
        "D": {"B"},
        "E": set(),
    }

    cmd = "select * from users"
    intent = "query"
    ctx = {"user": "alice", "session": "xyz"}

    theta_val = 0.5
    center_val = 0.0
    width_val = 1.0

    leaders = hybrid_leader_election(
        example_graph,
        cmd,
        intent,
        ctx,
        theta=theta_val,
        center=center_val,
        width=width_val,
        phases=6,
        seed=42,
    )
    print("Leaders:", leaders)

    # Base resource matrix (5 rows × 3 columns)
    base_mat = np.array(
        [
            [10, 20, 30],
            [15, 25, 35],
            [20, 30, 40],
            [25, 35, 45],
            [30, 40, 50],
        ],
        dtype=float,
    )
    adjusted_mat = hybrid_model_resource_matrix(
        base_mat, cmd, intent, ctx, theta_val, center_val, width_val
    )
    print("Adjusted resource matrix:\n", adjusted_mat)

    # Privacy risk per node
    nodes = list(example_graph.keys())
    risk_dict = hybrid_privacy_risk_vector(
        nodes,
        unique_quasi_identifiers=80,
        total_records=200,
        raw_command=cmd,
        normalized_intent=intent,
        context=ctx,
        theta=theta_val,
        center=center_val,
        width=width_val,
    )
    print("Privacy risk per node:", risk_dict)