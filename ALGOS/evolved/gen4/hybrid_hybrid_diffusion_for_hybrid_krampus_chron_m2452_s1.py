# DARWIN HAMMER — match 2452, survivor 1
# gen: 4
# parent_a: hybrid_diffusion_forcing_hybrid_hybrid_minimu_m918_s3.py (gen3)
# parent_b: hybrid_krampus_chrono_hybrid_possum_filter_m34_s1.py (gen3)
# born: 2026-05-29T23:42:22Z

"""Hybrid Diffusion‑Forcing & Temporal‑Spatial Resource Allocation

Parents:
- `diffusion_forcing.py` (Algorithm A) – provides a diffusion noise schedule
  α̅ₜ (alpha_bars) that quantifies uncertainty per token/step.
- `hybrid_krampus_chrono_hybrid_possum_filter_m34_s1.py` (Algorithm B) – builds a
  resource matrix **A** whose columns are spatial distance, privacy load, and
  temporal weight, and enforces linear budget constraints Aᵀ·x ≤ budgets.

Mathematical bridge:
The noise schedule α̅ₜ is interpreted as a *temporal confidence factor* ϕₜ∈[0,1].
Each edge in the minimum‑cost tree (Algorithm A) receives a base cost cᵢ (e.g.
haversine distance).  A certainty flag supplies a Bayesian confidence
pᵢ∈[0,1].  The hybrid edge weight wᵢ is defined as

    wᵢ = cᵢ · (1 – α̅ₜᵢ) · pᵢ

Thus the diffusion uncertainty attenuates the spatial cost, while epistemic
confidence amplifies it.  The resulting vector of hybrid weights becomes the
*spatial* column of the resource matrix **A**.  The *privacy* column is derived
from the certainty flags (higher confidence ⇒ higher privacy load), and the
*temporal* column is directly the noise schedule α̅ₜ.  Consequently the same
α̅ₜ that drives diffusion also drives the temporal budget in the linear system.

The module implements the full pipeline:
1. Generate α̅ₜ via `noise_schedule`.
2. Map textual labels to `CertaintyFlag` objects and to probabilities.
3. Compute hybrid edge weights.
4. Assemble the resource matrix **A** for entities + a synthetic “model tier”.
5. Find a feasible allocation vector x that respects spatial, privacy and temporal
   budgets.
6. Evaluate the total hybrid tree cost.

All operations rely only on `numpy`, `math`, `random`, `sys`, and `pathlib`."""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = (
    "FACT",
    "PROBABLE",
    "POSSIBLE",
    "BULLSHIT",
    "SURE_MAYBE",
)

@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # basis points, 0‑10000 => 0‑1 probability
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def probability(self) -> float:
        return self.confidence_bps / 10000.0


@dataclass(frozen=True)
class Entity:
    """Spatial‑temporal entity used by the resource‑budget sub‑system."""
    id: str
    lat: float
    lon: float
    category: str
    timestamp: str  # ISO‑8601 string
    address_signature: str = ""
    score: float = 0.0


# ----------------------------------------------------------------------
# Algorithm A – Diffusion schedule & hybrid edge weighting
# ----------------------------------------------------------------------
def noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    """Return the diffusion noise schedule α̅ₜ for t=0…T."""
    if schedule != "cosine":
        raise ValueError("Only 'cosine' schedule is implemented.")
    s = 0.008
    steps = np.arange(T + 1, dtype=np.float64)
    f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
    alpha_bars = f / f[0]
    return np.clip(alpha_bars, 0.0, 1.0)


def label_to_certainty(label: str) -> CertaintyFlag:
    """Map a textual label to a deterministic CertaintyFlag."""
    mapping: Dict[str, Tuple[int, str]] = {
        "FACT": (9500, "high"),
        "PROBABLE": (8000, "medium"),
        "POSSIBLE": (5000, "low"),
        "BULLSHIT": (1000, "very_low"),
        "SURE_MAYBE": (3000, "uncertain"),
    }
    if label not in mapping:
        raise ValueError(f"Unknown certainty label: {label}")
    confidence, authority = mapping[label]
    return CertaintyFlag(
        label=label,
        confidence_bps=confidence,
        authority_class=authority,
        rationale=f"Derived from label {label}",
    )


def hybrid_edge_weights(
    edge_costs: List[float],
    certainty_flags: List[CertaintyFlag],
    alpha_bars: np.ndarray,
) -> np.ndarray:
    """
    Compute hybrid edge weights wᵢ = cᵢ·(1‑α̅ₜᵢ)·pᵢ.

    Parameters
    ----------
    edge_costs: list of base spatial costs (e.g. haversine distances)
    certainty_flags: list of CertaintyFlag objects, one per edge
    alpha_bars: diffusion schedule; length must be ≥ len(edge_costs)

    Returns
    -------
    ndarray of hybrid weights (float64)
    """
    if len(edge_costs) != len(certainty_flags):
        raise ValueError("edge_costs and certainty_flags must have equal length")
    if len(alpha_bars) < len(edge_costs):
        raise ValueError("alpha_bars too short for number of edges")
    costs = np.asarray(edge_costs, dtype=np.float64)
    probs = np.asarray([cf.probability() for cf in certainty_flags], dtype=np.float64)
    temporal_factors = 1.0 - alpha_bars[: len(edge_costs)]
    return costs * temporal_factors * probs


def hybrid_tree_cost(edge_weights: np.ndarray) -> float:
    """Simple sum of hybrid edge weights – stands in for a minimum‑cost tree."""
    return float(np.sum(edge_weights))


# ----------------------------------------------------------------------
# Algorithm B – Spatial‑temporal resource matrix & budget feasibility
# ----------------------------------------------------------------------
def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat,lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000 * math.sqrt(h)


def temporal_weight_from_alpha(alpha: float) -> float:
    """Interpret diffusion α̅ₜ as a temporal weight (higher α̅ ⇒ higher weight)."""
    return float(alpha)


def privacy_load_from_certainty(cf: CertaintyFlag) -> float:
    """
    Map epistemic confidence to a privacy‑load scalar.
    Higher confidence → higher load (more sensitive).
    """
    # Linear map: p∈[0,1] → load∈[0.1,1.0]
    return 0.1 + 0.9 * cf.probability()


def build_resource_matrix(
    entities: List[Entity],
    reference_location: Tuple[float, float],
    alpha_bars: np.ndarray,
    certainty_map: Dict[str, CertaintyFlag],
) -> np.ndarray:
    """
    Assemble matrix **A** with rows = entities (plus a synthetic model tier)
    and columns = [spatial, privacy, temporal].

    Spatial column = haversine distance to reference location.
    Privacy column = derived from certainty flag (using entity.category as key).
    Temporal column = α̅ₜ taken cyclically over the rows.
    """
    rows = []
    for idx, ent in enumerate(entities):
        spatial = haversine_m((ent.lat, ent.lon), reference_location)
        cf = certainty_map.get(ent.category, CertaintyFlag("UNKNOWN", 0, "none", ""))
        privacy = privacy_load_from_certainty(cf)
        temporal = temporal_weight_from_alpha(alpha_bars[idx % len(alpha_bars)])
        rows.append([spatial, privacy, temporal])
    # Add a synthetic model tier row (e.g. RAM load, privacy‑adjusted factor)
    model_spatial = 0.0  # models do not contribute spatial distance
    model_privacy = 0.5  # placeholder
    model_temporal = np.mean(alpha_bars)  # average temporal load
    rows.append([model_spatial, model_privacy, model_temporal])
    return np.asarray(rows, dtype=np.float64)


def feasible_allocation(
    A: np.ndarray, budgets: Tuple[float, float, float]
) -> np.ndarray:
    """
    Find a simple feasible allocation vector x (length = rows of A)
    such that Aᵀ·x ≤ budgets.

    The algorithm uses a greedy scaling: start with x = 1 for all rows,
    compute violation, and uniformly scale down until constraints are met.
    """
    rows = A.shape[0]
    x = np.ones(rows, dtype=np.float64)
    for _ in range(1000):
        lhs = A.T @ x
        if np.all(lhs <= budgets):
            break
        # Compute scaling factor per constraint
        factors = np.where(lhs > budgets, budgets / lhs, 1.0)
        scale = np.min(factors)  # most restrictive
        x *= scale * 0.99  # a tiny margin to avoid oscillation
    else:
        raise RuntimeError("Failed to find feasible allocation within iteration limit")
    return x


# ----------------------------------------------------------------------
# High‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_pipeline(
    T: int,
    edge_costs: List[float],
    edge_labels: List[str],
    entities: List[Entity],
    reference_location: Tuple[float, float],
    budgets: Tuple[float, float, float],
) -> Dict[str, any]:
    """
    Execute the full hybrid workflow.

    Returns a dictionary containing:
        - alpha_bars
        - hybrid_weights
        - total_tree_cost
        - resource_matrix
        - allocation_vector
    """
    # 1. Diffusion schedule
    alpha = noise_schedule(T)

    # 2. Certainty flags for edges
    certainty_flags = [label_to_certainty(lbl) for lbl in edge_labels]

    # 3. Hybrid edge weights
    hybrid_w = hybrid_edge_weights(edge_costs, certainty_flags, alpha)

    # 4. Total tree cost
    total_cost = hybrid_tree_cost(hybrid_w)

    # 5. Certainty map for entities (use category as key)
    cert_map = {lbl: label_to_certainty(lbl) for lbl in set(e.category for e in entities)}

    # 6. Resource matrix
    A = build_resource_matrix(entities, reference_location, alpha, cert_map)

    # 7. Feasible allocation respecting budgets
    x = feasible_allocation(A, budgets)

    return {
        "alpha_bars": alpha,
        "hybrid_weights": hybrid_w,
        "total_tree_cost": total_cost,
        "resource_matrix": A,
        "allocation_vector": x,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy diffusion length
    T = 10

    # Edge costs (metres) and arbitrary certainty labels
    edge_costs = [random.uniform(1e3, 5e4) for _ in range(7)]
    edge_labels = random.choices(EPISTEMIC_FLAGS, k=7)

    # Create synthetic entities
    ref_loc = (37.7749, -122.4194)  # San Francisco
    entities = []
    for i in range(5):
        lat = ref_loc[0] + random.uniform(-0.1, 0.1)
        lon = ref_loc[1] + random.uniform(-0.1, 0.1)
        cat = random.choice(EPISTEMIC_FLAGS)
        ts = "2023-01-01T12:00:00Z"
        entities.append(Entity(id=f"E{i}", lat=lat, lon=lon, category=cat, timestamp=ts))

    # Budgets: spatial (metres), privacy (unitless), temporal (unitless)
    budgets = (2e5, 3.0, 5.0)

    result = hybrid_pipeline(
        T=T,
        edge_costs=edge_costs,
        edge_labels=edge_labels,
        entities=entities,
        reference_location=ref_loc,
        budgets=budgets,
    )

    # Simple sanity prints (no external libraries)
    print("Alpha schedule (first 5):", result["alpha_bars"][:5])
    print("Hybrid edge weights:", result["hybrid_weights"])
    print("Total hybrid tree cost:", result["total_tree_cost"])
    print("Resource matrix shape:", result["resource_matrix"].shape)
    print("Allocation vector:", result["allocation_vector"])
    print("All constraints satisfied:",
          np.all(result["resource_matrix"].T @ result["allocation_vector"] <= budgets))