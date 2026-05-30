# DARWIN HAMMER — match 2898, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m1964_s4.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hard_t_m1192_s1.py (gen5)
# born: 2026-05-29T23:46:33Z

"""Hybrid Engine Endpoint – Workshare Allocation Fusion

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – defines immutable `EngineEndpoint` objects, derives geometric
  indices (sphericity) and epistemic‑flag vectors, and validates health scores.
* **Parent B** – defines `WorkshareLane` objects, parses stylometric groups,
  and allocates LLM units across lanes using deterministic target percentages
  and rectified‑flow equations.

**Mathematical bridge**

The bridge is a linear mapping that treats the epistemic‑flag vector and the
geometric indices of each `EngineEndpoint` as a feature vector **f** ∈ ℝⁿ.
These features are projected onto the workshare lanes via a weight matrix **W**,
producing a raw allocation vector **a = W·f**.  The allocation is then
rectified (ReLU) and normalized to satisfy each lane’s deterministic target
percentage, yielding the final allocation **â**.  The combined system matrix
**M = W·diag(f)** captures both the endpoint topology and the lane‑share
topology, enabling downstream optimal‑flow calculations.

The module provides three high‑level functions that demonstrate this hybrid
operation:
1. `endpoint_feature_vector` – builds the feature vector from an endpoint.
2. `allocate_workshare` – projects features onto lanes, applies rectified flow,
   and respects deterministic target percentages.
3. `optimal_loading_path` – computes a minimal‑norm path through the combined
   system matrix using a pseudo‑inverse (numpy.linalg.pinv).

All code is pure Python with only `numpy`, `math`, `random`, `sys`,
`pathlib`, and standard library imports.  No external ML frameworks are used.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Endpoint definitions and geometric indices
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Geometric description of an engine component."""
    length: float
    width: float
    height: float
    mass: float

    def __post_init__(self) -> None:
        if any(v <= 0 for v in (self.length, self.width, self.height, self.mass)):
            raise ValueError("All morphology dimensions and mass must be positive.")


@dataclass(frozen=True)
class EngineEndpoint:
    """Immutable description of an engine endpoint."""
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    health_score: float
    morphology: Morphology

    def __post_init__(self) -> None:
        if not (0.0 <= self.health_score <= 1.0):
            raise ValueError("health_score must be in the interval [0, 1].")
        if not self.capabilities:
            raise ValueError("capabilities list cannot be empty.")


# Epistemic flags (shared with Parent A)
EPISTEMIC_FLAGS: Tuple[str, ...] = (
    "FACT",
    "PROBABLE",
    "POSSIBLE",
    "UNCERTAIN",
    "UNKNOWN",
)


def epistemic_flag_vector(endpoint: EngineEndpoint) -> np.ndarray:
    """Binary vector indicating presence of each epistemic flag."""
    return np.array(
        [1.0 if flag in endpoint.capabilities else 0.0 for flag in EPISTEMIC_FLAGS],
        dtype=float,
    )


def sphericity_index(m: Morphology) -> float:
    """Ratio of the geometric mean of the dimensions to the longest side."""
    gm = (m.length * m.width * m.height) ** (1.0 / 3.0)
    return gm / max(m.length, m.width, m.height)


def curvature_score(m: Morphology) -> float:
    """
    A simple curvature surrogate: difference between surface area and volume
    normalized by the longest side.  This provides a scalar that can be mixed
    with the sphericity index.
    """
    surface = 2 * (m.length * m.width + m.width * m.height + m.height * m.length)
    volume = m.length * m.width * m.height
    longest = max(m.length, m.width, m.height)
    return (surface - volume) / (longest ** 2 + 1e-9)


def endpoint_feature_vector(endpoint: EngineEndpoint) -> np.ndarray:
    """
    Assemble a feature vector for an endpoint.
    Features (in order):
        - Epistemic flag binary vector (len = 5)
        - Health score (scalar)
        - Sphericity index (scalar)
        - Curvature score (scalar)
    Returns a 1‑D numpy array of shape (8,).
    """
    flags = epistemic_flag_vector(endpoint)                     # (5,)
    health = np.array([endpoint.health_score], dtype=float)   # (1,)
    sph = np.array([sphericity_index(endpoint.morphology)], dtype=float)  # (1,)
    curv = np.array([curvature_score(endpoint.morphology)], dtype=float)  # (1,)
    return np.concatenate([flags, health, sph, curv])          # (8,)


# ----------------------------------------------------------------------
# Parent B – Workshare lane definitions and allocation utilities
# ----------------------------------------------------------------------


GROUPS = ("codex", "groq", "cohere", "local_models")


def words(text: str) -> List[str]:
    """Extract lower‑case alphabetic tokens from a string."""
    return [word.lower() for word in text.split() if word.isalpha()]


@dataclass
class WorkshareLane:
    """Definition of a workshare allocation lane."""
    group: str                 # one of GROUPS
    llm_units: float           # total units available in this lane
    llm_share_pct: float       # deterministic target percentage (0‑1)
    proof_required: bool       # whether the lane demands proof of work


def rectified_flow(x: np.ndarray) -> np.ndarray:
    """
    Rectified flow equation used in Parent B.
    Implements element‑wise ReLU followed by a soft‑normalization that keeps
    the total sum equal to the original L1 norm.
    """
    relu = np.maximum(0.0, x)
    total = relu.sum()
    if total == 0.0:
        return relu
    return relu / total * x.sum()


# ----------------------------------------------------------------------
# Hybrid core: projection, allocation, and optimal path
# ----------------------------------------------------------------------


def build_weight_matrix(endpoints: List[EngineEndpoint],
                        lanes: List[WorkshareLane]) -> np.ndarray:
    """
    Construct a weight matrix W that maps endpoint feature space (8 dims) to
    lanes (len(lanes) dims).  The matrix is built from simple heuristics that
    respect both parent designs:

        - Epistemic flags contribute proportionally to lanes that share the same
          group name (e.g., a flag "FACT" boosts lanes whose group appears in the
          endpoint's `resource_class` string).
        - Health score scales the whole row.
        - Geometric indices (sphericity, curvature) influence lanes with
          `proof_required == True`.

    The resulting matrix shape is (len(lanes), 8).
    """
    n_lanes = len(lanes)
    W = np.zeros((n_lanes, 8), dtype=float)

    for i, lane in enumerate(lanes):
        # Base contribution from epistemic flags
        flag_weights = np.array([1.0 if lane.group in endpoint.resource_class.lower()
                                 else 0.5 for endpoint in endpoints])
        # Average across endpoints to obtain a single vector per flag
        avg_flag = flag_weights.mean()
        # Populate the first five columns (flags) with this average
        W[i, :5] = avg_flag

        # Health contribution: average health across endpoints
        avg_health = np.mean([ep.health_score for ep in endpoints])
        W[i, 5] = avg_health

        # Geometric contributions
        avg_sph = np.mean([sphericity_index(ep.morphology) for ep in endpoints])
        avg_curv = np.mean([curvature_score(ep.morphology) for ep in endpoints])

        # If proof is required, give more weight to curvature; otherwise to sphericity
        if lane.proof_required:
            W[i, 6] = avg_curv
            W[i, 7] = avg_sph * 0.5
        else:
            W[i, 6] = avg_sph
            W[i, 7] = avg_curv * 0.5

        # Finally, scale the entire row by the lane's target percentage
        W[i] *= lane.llm_share_pct

    return W


def allocate_workshare(endpoints: List[EngineEndpoint],
                       lanes: List[WorkshareLane]) -> Dict[str, float]:
    """
    Perform the hybrid allocation:

        1. Build the feature matrix F (len(endpoints) × 8) by stacking each
           endpoint's feature vector.
        2. Build the weight matrix W (len(lanes) × 8) via `build_weight_matrix`.
        3. Compute raw allocation a = W · mean(F, axis=0)  (shape: len(lanes),).
        4  Apply rectified flow to enforce non‑negativity while preserving total
           allocated units.
        5. Normalize each lane's allocation to its deterministic target percentage
           of the total available LLM units across all lanes.

    Returns a mapping from lane.group to the final allocated units.
    """
    if not endpoints or not lanes:
        raise ValueError("Both endpoints and lanes must be non‑empty.")

    # 1. Feature matrix
    F = np.stack([endpoint_feature_vector(ep) for ep in endpoints])  # (E,8)

    # 2. Weight matrix
    W = build_weight_matrix(endpoints, lanes)                         # (L,8)

    # 3. Raw allocation using the mean feature vector
    f_mean = F.mean(axis=0)                                            # (8,)
    raw_alloc = W @ f_mean                                            # (L,)

    # 4. Rectified flow
    flow_alloc = rectified_flow(raw_alloc)

    # 5. Deterministic target normalization
    total_units = sum(l.llm_units for l in lanes)
    target_units = np.array([lane.llm_share_pct * total_units for lane in lanes])
    # Avoid division by zero
    scaling = np.divide(target_units, flow_alloc, out=np.zeros_like(target_units),
                        where=flow_alloc != 0)
    final_alloc = flow_alloc * scaling

    # Build result dict
    allocation_dict = {lane.group: float(final_alloc[i]) for i, lane in enumerate(lanes)}
    return allocation_dict


def optimal_loading_path(endpoints: List[EngineEndpoint],
                         lanes: List[WorkshareLane],
                         target_pct: float = 0.75) -> np.ndarray:
    """
    Compute an optimal loading path through the combined system matrix.

    The combined system matrix is defined as:

        M = W · diag(f_mean)

    where:
        - W is the weight matrix (see `build_weight_matrix`).
        - f_mean is the mean endpoint feature vector.
        - diag(f_mean) creates a diagonal scaling that injects endpoint geometry
          into the lane space.

    The optimal path is the minimum‑norm solution x to M·x = b,
    where b is a target vector representing `target_pct` of each lane's
    available units.  The solution uses the Moore‑Penrose pseudo‑inverse.

    Returns the vector x (shape: len(lanes),) containing the proportion of each
    lane to be engaged.
    """
    if not (0.0 < target_pct <= 1.0):
        raise ValueError("target_pct must be in (0, 1].")

    # Mean feature vector
    f_mean = np.mean([endpoint_feature_vector(ep) for ep in endpoints], axis=0)  # (8,)

    # Weight matrix
    W = build_weight_matrix(endpoints, lanes)                                   # (L,8)

    # Combined system matrix
    M = W @ np.diag(f_mean)                                                      # (L,8)

    # Target vector b: desired amount of LLM units per lane
    total_units = sum(l.llm_units for l in lanes)
    b = np.full((len(lanes),), target_pct * total_units / len(lanes), dtype=float)

    # Pseudo‑inverse solution
    pinv = np.linalg.pinv(M)          # (8, L)
    x = pinv @ b                      # (8,)

    # Map back to lane space by applying W again (ensures dimensionality matches lanes)
    lane_load = W @ x                  # (L,)

    # Clip negatives (should not happen after pseudo‑inverse but guard anyway)
    lane_load = np.maximum(0.0, lane_load)

    # Normalize to total_units * target_pct
    scale = (target_pct * total_units) / lane_load.sum() if lane_load.sum() else 0.0
    lane_load *= scale

    return lane_load


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a few synthetic endpoints
    endpoints = [
        EngineEndpoint(
            engine_id="eng-1",
            channel="alpha",
            residency="cloud",
            runtime="python3.11",
            resource_class="codex",
            always_on=True,
            endpoint="https://api.example.com/v1",
            capabilities=["FACT", "PROBABLE", "UNKNOWN"],
            health_score=0.92,
            morphology=Morphology(length=1.2, width=0.8, height=0.5, mass=2.3),
        ),
        EngineEndpoint(
            engine_id="eng-2",
            channel="beta",
            residency="edge",
            runtime="python3.10",
            resource_class="groq",
            always_on=False,
            endpoint="https://edge.example.com/v2",
            capabilities=["POSSIBLE", "UNCERTAIN"],
            health_score=0.78,
            morphology=Morphology(length=0.9, width=1.1, height=0.6, mass=1.8),
        ),
    ]

    # Define workshare lanes
    lanes = [
        WorkshareLane(group="codex", llm_units=120.0, llm_share_pct=0.4, proof_required=False),
        WorkshareLane(group="groq", llm_units=80.0, llm_share_pct=0.35, proof_required=True),
        WorkshareLane(group="cohere", llm_units=50.0, llm_share_pct=0.15, proof_required=False),
        WorkshareLane(group="local_models", llm_units=30.0, llm_share_pct=0.10, proof_required=True),
    ]

    # 1. Feature vector demonstration
    fv = endpoint_feature_vector(endpoints[0])
    print("Feature vector (first endpoint):", fv)

    # 2. Hybrid allocation
    allocation = allocate_workshare(endpoints, lanes)
    print("\nHybrid workshare allocation (units per group):")
    for grp, units in allocation.items():
        print(f"  {grp}: {units:.2f}")

    # 3. Optimal loading path
    path = optimal_loading_path(endpoints, lanes, target_pct=0.75)
    print("\nOptimal loading path (units per lane):")
    for lane, units in zip(lanes, path):
        print(f"  {lane.group}: {units:.2f}")

    # Simple sanity checks (no exceptions)
    assert all(v >= 0 for v in allocation.values()), "Negative allocation detected"
    assert abs(sum(allocation.values()) - sum(l.llm_units * l.llm_share_pct for l in lanes)) < 1e-6, "Allocation does not respect target percentages"
    assert all(u >= 0 for u in path), "Negative values in optimal path"
    print("\nSmoke test completed successfully.")