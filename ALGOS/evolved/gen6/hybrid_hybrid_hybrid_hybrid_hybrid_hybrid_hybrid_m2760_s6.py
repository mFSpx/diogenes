# DARWIN HAMMER — match 2760, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m1482_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2132_s0.py (gen5)
# born: 2026-05-29T23:45:40Z

"""Hybrid Algorithm integrating:
- Parent A: hybrid_hybrid_hybrid_gliner_hybrid_possum_filter_m323_s2.py
- Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2132_s0.py

Mathematical Bridge:
Parent A scores Span‑Entity pairs using a product of their scores attenuated by a
distance factor and weighted by decayed pheromone signals.  
Parent B provides sheaf‑cohomology sections that act as multiplicative weights
for pheromone probabilities, together with Fisher‑information‑based
information‑theoretic scaling and a Bayesian update of the pheromone
distribution.

The hybrid algorithm therefore:
1. Uses the sheaf section norm as a weight for each pheromone entry.
2. Applies exponential decay to pheromone signals (Parent A) and aggregates them.
3. Modulates the aggregated pheromone weight by a Fisher‑information scalar
   derived from the pheromone value distribution (Parent B).
4. Updates a prior probability vector with the pheromone likelihood via a
   Bayesian rule, yielding a Bayes factor that further scales the final score.
5. Combines the above with the original Span‑Entity product and the
   distance‑attenuation factor to produce a unified hybrid score.
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
import re

# ----------------------------------------------------------------------
# Core data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """A textual span with optional geospatial hint."""
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"
    # optional latitude/longitude for distance calculations
    lat: float = None
    lon: float = None


class PheromoneEntry:
    """Tracks a pheromone signal with exponential decay."""
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

    def decayed_value(self) -> float:
        """Return the signal value after exponential half‑life decay."""
        age = self.age_seconds()
        if self.half_life_seconds <= 0:
            return self.signal_value
        decay_factor = 0.5 ** (age / self.half_life_seconds)
        return self.signal_value * decay_factor


# ----------------------------------------------------------------------
# Core data structures (from Parent B)
# ----------------------------------------------------------------------
class Sheaf:
    """Simple sheaf storing a vector section per node."""
    def __init__(self, node_dims: Dict[Any, int], edge_list: List[Tuple[Any, Any]]):
        self.node_dims = dict(node_dims)          # dimension per node
        self.edges = list(edge_list)              # list of edges (u, v)
        self._restrictions = {}                   # edge -> (src_map, dst_map)
        self._sections = {}                       # node -> np.ndarray

    def set_restriction(self, edge: Tuple[Any, Any], src_map: List[float], dst_map: List[float]):
        u, v = edge
        self._restrictions[(u, v)] = (np.array(src_map, dtype=float),
                                      np.array(dst_map, dtype=float))

    def set_section(self, node: Any, value: List[float]):
        self._sections[node] = np.array(value, dtype=float)

    def get_section_weight(self, node: Any) -> float:
        """Return a scalar weight derived from the norm of the section vector."""
        vec = self._sections.get(node)
        if vec is None:
            return 1.0
        return np.linalg.norm(vec) + 1e-12   # avoid zero


# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great‑circle distance between two lat/lon points (kilometers)."""
    if None in (lat1, lon1, lat2, lon2):
        return 0.0
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def distance_attenuation(span_a: Span, span_b: Span, scale: float = 10.0) -> float:
    """
    Exponential attenuation based on spatial (or positional) distance.
    If geographic coordinates are missing, fall back to index distance.
    """
    if span_a.lat is not None and span_b.lat is not None:
        dist = haversine(span_a.lat, span_a.lon, span_b.lat, span_b.lon)
    else:
        dist = abs(span_a.start - span_b.start)
    return math.exp(-dist / scale)


def aggregate_pheromone(span: Span,
                        pheromones: List[PheromoneEntry],
                        sheaf: Sheaf) -> float:
    """
    Sum of decayed pheromone values, each multiplied by a sheaf‑section weight.
    The span's surface key (here we reuse span.label) selects the sheaf node.
    """
    total = 0.0
    node = span.label
    sheaf_weight = sheaf.get_section_weight(node)
    for p in pheromones:
        total += p.decayed_value() * sheaf_weight
    return total


def fisher_information(values: np.ndarray) -> float:
    """
    Approximate Fisher information for a discrete distribution.
    We use the variance of the log‑probabilities as a proxy.
    """
    if values.size == 0:
        return 0.0
    probs = values / values.sum() if values.sum() != 0 else np.ones_like(values) / values.size
    logp = np.log(probs + 1e-12)
    return float(np.var(logp))


def bayesian_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """
    Simple Bayesian update: posterior ∝ prior * likelihood.
    The vectors must be the same length.
    """
    if prior.shape != likelihood.shape:
        raise ValueError("Prior and likelihood must have the same shape.")
    unnorm = prior * likelihood
    total = unnorm.sum()
    if total == 0:
        return prior
    return unnorm / total


# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def hybrid_score(span: Span,
                 entity_score: float,
                 counterpart: Span,
                 pheromones: List[PheromoneEntry],
                 sheaf: Sheaf,
                 prior: np.ndarray) -> float:
    """
    Compute the unified hybrid score for a Span‑Entity pair.
    Steps:
    1. Base product of span.score and entity_score.
    2. Distance attenuation with counterpart span.
    3. Aggregated pheromone weight (sheaf‑modulated).
    4. Fisher‑information scaling.
    5. Bayesian factor derived from pheromone likelihood.
    """
    # 1. Base product
    base = span.score * entity_score

    # 2. Distance attenuation
    att = distance_attenuation(span, counterpart)

    # 3. Pheromone aggregation (sheaf weighting)
    pher_weight = aggregate_pheromone(span, pheromones, sheaf) + 1e-9

    # 4. Fisher information factor
    pher_vals = np.array([p.signal_value for p in pheromones], dtype=float)
    fisher_factor = 1.0 + fisher_information(pher_vals)

    # 5. Bayesian posterior factor
    if pher_vals.size > 0:
        likelihood = pher_vals / pher_vals.sum()
        # truncate or pad prior to match likelihood length
        if prior.size < likelihood.size:
            prior = np.pad(prior, (0, likelihood.size - prior.size), constant_values=1.0)
        elif prior.size > likelihood.size:
            prior = prior[:likelihood.size]
        posterior = bayesian_update(prior, likelihood)
        bayes_factor = posterior.mean()
    else:
        bayes_factor = 1.0

    # Final hybrid score
    return base * att * pher_weight * fisher_factor * bayes_factor


def compute_pheromone_distribution(pheromones: List[PheromoneEntry]) -> np.ndarray:
    """
    Return a normalized probability distribution over pheromone signal values.
    """
    if not pheromones:
        return np.array([], dtype=float)
    vals = np.array([p.signal_value for p in pheromones], dtype=float)
    total = vals.sum()
    if total == 0:
        return np.full_like(vals, 1.0 / vals.size)
    return vals / total


def update_pheromones(pheromones: List[PheromoneEntry],
                     observation: float,
                     learning_rate: float = 0.1) -> None:
    """
    Simple reinforcement: increase each pheromone's signal_value proportionally
    to the observed evidence, then let decay happen naturally.
    """
    for p in pheromones:
        p.signal_value = (1 - learning_rate) * p.signal_value + learning_rate * observation


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create dummy spans
    span_a = Span(start=10, end=20, text="alpha", label="nodeA",
                  score=0.78, lat=37.7749, lon=-122.4194)
    span_b = Span(start=15, end=25, text="beta", label="nodeB",
                  score=0.65, lat=34.0522, lon=-118.2437)

    # Create pheromone entries
    pheromones = [
        PheromoneEntry(surface_key="nodeA", signal_kind="click", signal_value=0.9, half_life_seconds=30),
        PheromoneEntry(surface_key="nodeB", signal_kind="view", signal_value=0.4, half_life_seconds=45),
        PheromoneEntry(surface_key="nodeA", signal_kind="share", signal_value=0.2, half_life_seconds=60)
    ]

    # Build a simple sheaf with sections per node label
    sheaf = Sheaf(node_dims={"nodeA": 3, "nodeB": 3}, edge_list=[])
    sheaf.set_section("nodeA", [0.5, 0.3, 0.2])   # norm ≈ 0.616
    sheaf.set_section("nodeB", [0.1, 0.1, 0.8])   # norm ≈ 0.812

    # Prior (uniform) for Bayesian update
    prior = np.full(3, 1.0 / 3)

    # Compute hybrid score
    score = hybrid_score(
        span=span_a,
        entity_score=0.55,
        counterpart=span_b,
        pheromones=pheromones,
        sheaf=sheaf,
        prior=prior
    )
    print(f"Hybrid score (span A vs B): {score:.6f}")

    # Demonstrate auxiliary functions
    dist = haversine(span_a.lat, span_a.lon, span_b.lat, span_b.lon)
    print(f"Haversine distance (km): {dist:.3f}")

    pher_dist = compute_pheromone_distribution(pheromones)
    print(f"Pheromone distribution: {pher_dist}")

    # Simulate an observation and update pheromones
    update_pheromones(pheromones, observation=0.7)
    updated_vals = [p.signal_value for p in pheromones]
    print(f"Pheromone values after update: {updated_vals}")