# DARWIN HAMMER — match 2760, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m1482_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2132_s0.py (gen5)
# born: 2026-05-29T23:45:40Z

"""
Hybrid Algorithm: fusion of
- hybrid_hybrid_hybrid_gliner_hybrid_possum_filter_m323_s2.py (Parent A)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2132_s0.py (Parent B)

Mathematical Bridge
-------------------
Parent A scores a (Span, Entity) pair as

    S_A = span.score × entity_score × A(d)

where A(d) = exp(‑α·d) is a haversine‑based attenuation factor.
Parent B provides a feature‑count vector **w** (derived from hygiene regexes) and a
Sheaf **𝓢** whose node‑sections sₙ ∈ ℝᵏ act as multiplicative weights on pheromone
signals.  Fisher information **I(p)** of a pheromone probability vector **p**
and a Bayesian update rule are also defined.

The hybrid algorithm fuses these structures by:

1. Using the feature‑count vector **w** to weight the product
   `span.score × entity_score` componentwise.
2. Modulating each pheromone entry’s signal value with the corresponding
   Sheaf section `sₙ`.
3. Adjusting the pheromone decay using the Fisher information of the
   current probability distribution and performing a Bayesian update.

The three core functions below implement this unified system.
"""

import math
import random
import sys
import pathlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (adapted from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """A textual span with a confidence score."""
    start: int
    end: int
    text: str
    label: str
    score: float
    # optional geographic coordinates for distance attenuation
    lat: float = 0.0
    lon: float = 0.0
    backend: str = "literal_fallback"


@dataclass(frozen=True)
class Entity:
    """A generic entity that can be paired with a Span."""
    identifier: str
    score: float
    lat: float = 0.0
    lon: float = 0.0


class PheromoneEntry:
    """Tracks a pheromone signal on a surface (Parent B)."""
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    def decay_factor(self, current_time: datetime) -> float:
        """Exponential decay based on half‑life."""
        elapsed = (current_time - self.created_at).total_seconds()
        return 0.5 ** (elapsed / self.half_life_seconds)


# ----------------------------------------------------------------------
# Sheaf structure (Parent B)
# ----------------------------------------------------------------------
class Sheaf:
    """A simple sheaf whose sections are vectors attached to nodes."""
    def __init__(self, node_dims: Dict[Any, int], edge_list: List[Tuple[Any, Any]]):
        self.node_dims = dict(node_dims)          # node → dimension
        self.edges = list(edge_list)              # list of (u, v)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[Any, Any],
                        src_map: Iterable[float],
                        dst_map: Iterable[float]) -> None:
        u, v = edge
        self._restrictions[(u, v)] = (np.array(src_map, dtype=float),
                                      np.array(dst_map, dtype=float))

    def set_section(self, node: Any, value: Iterable[float]) -> None:
        dim = self.node_dims.get(node, None)
        if dim is None:
            raise ValueError(f"Node {node} not declared in node_dims.")
        arr = np.array(value, dtype=float)
        if arr.shape[0] != dim:
            raise ValueError(f"Section dimension mismatch for node {node}.")
        self._sections[node] = arr

    def get_section(self, node: Any) -> np.ndarray:
        return self._sections.get(node, np.ones(self.node_dims.get(node, 1)))


# ----------------------------------------------------------------------
# Helper mathematics
# ----------------------------------------------------------------------
def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """Return great‑circle distance in kilometres between two lat/lon points."""
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_span_entity_score(span: Span,
                             entity: Entity,
                             feature_weights: np.ndarray,
                             alpha: float = 0.01) -> float:
    """
    Compute the fused score for a (Span, Entity) pair.

    - The raw product is `span.score * entity.score`.
    - It is weighted component‑wise by `feature_weights` (length must be 1 or
      match the dimensionality of the feature space; we use its L2 norm as a
      scalar factor).
    - A haversine‑based attenuation `exp(-α·d)` is applied, where `d` is the
      geographic distance between span and entity.

    Returns a scalar hybrid score.
    """
    # 1. Base product
    base = span.score * entity.score

    # 2. Feature weighting
    if feature_weights.size == 0:
        weight_factor = 1.0
    else:
        # Use the L2 norm to collapse the vector into a scalar weight
        weight_factor = np.linalg.norm(feature_weights)
    weighted = base * weight_factor

    # 3. Distance attenuation
    d = haversine_distance(span.lat, span.lon, entity.lat, entity.lon)
    attenuation = math.exp(-alpha * d)

    return weighted * attenuation


def pheromone_sheaf_update(entry: PheromoneEntry,
                          sheaf: Sheaf,
                          node: Any,
                          current_time: datetime = None) -> None:
    """
    Update a pheromone entry in place using:

    * Multiplicative modulation by the Sheaf section attached to `node`.
    * Decay accelerated by the Fisher information of the entry's current
      probability vector (here approximated by the scalar `signal_value`).

    The function mutates `entry.signal_value`.
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)

    # 1. Sheaf modulation
    section = sheaf.get_section(node)          # vector
    mod_factor = np.mean(section)              # simple scalar summary
    entry.signal_value *= mod_factor

    # 2. Fisher‑information‑based decay
    # For a scalar probability p, Fisher information I(p) = 1 / (p * (1-p))
    p = max(min(entry.signal_value, 0.999999), 1e-9)  # clamp to (0,1)
    fisher_info = 1.0 / (p * (1.0 - p))

    # Accelerate decay: effective half‑life = original / (1 + I)
    effective_half_life = entry.half_life_seconds / (1.0 + fisher_info)
    elapsed = (current_time - entry.created_at).total_seconds()
    decay = 0.5 ** (elapsed / effective_half_life)

    entry.signal_value *= decay
    entry.last_decay = current_time


def bayesian_fisher_update(prior: np.ndarray,
                           likelihood: np.ndarray,
                           epsilon: float = 1e-12) -> np.ndarray:
    """
    Perform a Bayesian update of a discrete probability distribution
    `prior` with a `likelihood` vector, then adjust the posterior using
    Fisher information as a regulariser.

    Steps:
    1. Standard Bayes: posterior ∝ prior ⊙ likelihood
    2. Normalise.
    3. Compute Fisher information I = Σ ( (∂ log p_i / ∂θ)^2 p_i )
       for a dummy parameter θ; for a categorical distribution this reduces
       to the variance of the posterior.
    4. Blend posterior with a uniform distribution weighted by I
       (higher I → stronger regularisation).

    Returns the regularised posterior distribution.
    """
    # 1. Unnormalised posterior
    unnorm = prior * likelihood
    if unnorm.sum() == 0:
        # avoid division by zero – fall back to uniform
        posterior = np.full_like(prior, 1.0 / prior.size)
    else:
        posterior = unnorm / unnorm.sum()

    # 2. Fisher information (variance proxy)
    variance = np.var(posterior)
    fisher_info = variance + epsilon  # ensure positivity

    # 3. Regularisation blending
    uniform = np.full_like(posterior, 1.0 / posterior.size)
    blended = (posterior + fisher_info * uniform) / (1.0 + fisher_info)

    # 4. Final normalisation
    blended /= blended.sum()
    return blended


# ----------------------------------------------------------------------
# Demonstration utilities
# ----------------------------------------------------------------------
def _dummy_feature_vector(dim: int = 5) -> np.ndarray:
    """Generate a random positive feature‑count vector."""
    vec = np.random.poisson(lam=3, size=dim).astype(float) + 1.0
    return vec


def _build_demo_sheaf() -> Sheaf:
    """Create a small Sheaf with two nodes for the smoke test."""
    node_dims = {"A": 3, "B": 3}
    edges = [("A", "B")]
    sheaf = Sheaf(node_dims, edges)
    sheaf.set_section("A", [0.9, 1.1, 1.0])
    sheaf.set_section("B", [1.2, 0.8, 1.0])
    sheaf.set_restriction(("A", "B"),
                          src_map=[1, 0, 0],
                          dst_map=[0, 1, 0])
    return sheaf


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Create dummy Span and Entity with geographic coordinates
    span = Span(start=0, end=10, text="example", label="TEST",
                score=0.78, lat=40.7128, lon=-74.0060)   # New York
    entity = Entity(identifier="E1", score=0.85,
                    lat=34.0522, lon=-118.2437)          # Los Angeles

    # Feature weights from Parent B (e.g., hygiene regex counts)
    w = _dummy_feature_vector(dim=4)

    # Hybrid score
    hybrid_score = hybrid_span_entity_score(span, entity, w, alpha=0.0005)
    print(f"Hybrid Span‑Entity Score: {hybrid_score:.6f}")

    # Pheromone entry and Sheaf update
    pher = PheromoneEntry(surface_key="surf1",
                          signal_kind="usage",
                          signal_value=0.6,
                          half_life_seconds=3600)
    sheaf = _build_demo_sheaf()
    pheromone_sheaf_update(pher, sheaf, node="A")
    print(f"Updated pheromone signal value: {pher.signal_value:.6f}")

    # Bayesian + Fisher update on a categorical distribution
    prior = np.array([0.2, 0.5, 0.3])
    likelihood = np.array([0.7, 0.2, 0.1])
    posterior = bayesian_fisher_update(prior, likelihood)
    print(f"Posterior after Bayesian‑Fisher update: {posterior}")