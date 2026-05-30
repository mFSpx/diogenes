# DARWIN HAMMER — match 1776, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_hybrid_m720_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hoeffding_tre_m44_s1.py (gen3)
# born: 2026-05-29T23:38:44Z

"""Hybrid Algorithm: Pheromone‑Tropical‑Hoeffding Fusion

Parents:
- **Algorithm A** – hybrid_hybrid_hybrid_gliner_hybrid_hybrid_hybrid_m720_s1.py  
  Provides a pheromone store where each *Span* creates a decaying signal.  
- **Algorithm B** – hybrid_hybrid_hybrid_endpoi_hybrid_hoeffding_tre_m44_s1.py  
  Provides a tropical (max‑plus) network that produces candidate gains and a
  Hoeffding bound that decides when a split (or decision) is statistically
  justified.

Mathematical Bridge:
Each *endpoint* from Algorithm B is treated as a *state dimension* whose
feature value is the **aggregated pheromone signal** coming from the spans
that refer to that endpoint.  The vector of aggregated pheromones forms the
input to the tropical network.  The network’s output (gain candidates) is
compared against a Hoeffding bound whose confidence parameter *δ* is derived
from the **certainty flags** (here the `score` of a span, normalised).  Thus the
fusion couples the decay‑driven pheromone dynamics with the max‑plus algebra
and statistical guarantee of Algorithm B.

The resulting system can be used for online decision making: as new spans
arrive they update the pheromone store, the aggregated signals drive the
tropical evaluation, and the Hoeffding bound decides whether a split (or any
action) should be taken.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures from Parent A
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    """A labeled text segment that carries a raw confidence score."""
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


class PheromoneEntry:
    """A decaying pheromone signal attached to a surface (e.g., an endpoint)."""

    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(self, surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int):
        self.uuid = str(pathlib.Path.cwd())  # placeholder unique id
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """Global repository for pheromone entries."""

    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add_entry(cls, surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int) -> None:
        entry = PheromoneEntry(surface_key, signal_kind, signal_value, half_life_seconds)
        cls._entries[entry.uuid] = entry

    @classmethod
    def decay_all(cls) -> None:
        for entry in list(cls._entries.values()):
            entry.apply_decay()
            if entry.signal_value < 1e-9:  # prune near‑zero signals
                del cls._entries[entry.uuid]

    @classmethod
    def aggregate_by_surface(cls) -> Dict[str, float]:
        """Sum decayed signal values for each surface key."""
        agg: Dict[str, float] = {}
        for entry in cls._entries.values():
            agg.setdefault(entry.surface_key, 0.0)
            agg[entry.surface_key] += entry.signal_value
        return agg


# ----------------------------------------------------------------------
# Core data structures from Parent B
# ----------------------------------------------------------------------


class StateDimension:
    """Represents an endpoint with health‑related attributes."""

    def __init__(self, endpoint: str, failure_rate: float, recovery_priority: float):
        self.endpoint = endpoint
        self.failure_rate = failure_rate
        self.recovery_priority = recovery_priority


class TropicalNetwork:
    """A max‑plus (ReLU‑like) network."""

    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        assert weights.shape[0] == biases.shape[0], "weights rows must match bias length"
        self.weights = weights
        self.biases = biases

    def evaluate(self, input_vector: np.ndarray) -> np.ndarray:
        """max(0, W·x + b) applied element‑wise (tropical ReLU)."""
        raw = self.weights @ input_vector + self.biases
        return np.maximum(0.0, raw)


def hoeffding_bound(delta: float, n: int) -> float:
    """Classic Hoeffding bound for binary outcomes."""
    if n <= 0:
        return float("inf")
    return math.sqrt((math.log(2.0 / delta)) / (2 * n))


# ----------------------------------------------------------------------
# Hybrid Operations (at least three functions)
# ----------------------------------------------------------------------


def add_pheromones_from_spans(spans: List[Span]) -> None:
    """
    Convert a list of Span objects into pheromone entries.
    The half‑life is heuristically derived from the label length;
    the signal kind is the span label.
    """
    for sp in spans:
        half_life = max(1, len(sp.label) * 10)  # simple heuristic
        PheromoneStore.add_entry(
            surface_key=sp.label,
            signal_kind="span_score",
            signal_value=sp.score,
            half_life_seconds=half_life,
        )


def aggregate_pheromone_vector(state_dims: List[StateDimension]) -> np.ndarray:
    """
    Build the input vector for the tropical network.
    Each component i corresponds to the aggregated pheromone signal for the
    endpoint identified by `state_dims[i].endpoint`.  If no pheromone exists,
    the component is zero.
    """
    agg = PheromoneStore.aggregate_by_surface()
    vec = np.zeros(len(state_dims))
    for i, dim in enumerate(state_dims):
        vec[i] = agg.get(dim.endpoint, 0.0)
    return vec


def hybrid_split_decision(
    state_dims: List[StateDimension],
    tropical_net: TropicalNetwork,
    delta: float,
    sample_counts: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one hybrid decision step.

    1. Decay all pheromones (simulating passage of time).
    2. Build the input vector from aggregated pheromones.
    3. Evaluate the tropical network → gain candidates.
    4. Compute a Hoeffding bound per dimension using the supplied sample
       counts (derived from certainty flags in the original system).
    5. Return a tuple (gain_vector, bound_vector); the caller may split where
       gain > bound.
    """
    # 1. decay
    PheromoneStore.decay_all()

    # 2. input vector
    input_vec = aggregate_pheromone_vector(state_dims)

    # 3. tropical evaluation
    gains = tropical_net.evaluate(input_vec)

    # 4. Hoeffding bounds per dimension
    bounds = np.array([hoeffding_bound(delta, n) for n in sample_counts])

    return gains, bounds


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create synthetic spans
    sample_spans = [
        Span(start=0, end=5, text="alpha", label="endpoint_A", score=0.8),
        Span(start=6, end=10, text="beta", label="endpoint_B", score=0.3),
        Span(start=11, end=15, text="gamma", label="endpoint_A", score=0.5),
    ]

    # Populate pheromone store
    add_pheromones_from_spans(sample_spans)

    # Define endpoints / state dimensions
    endpoints = [
        StateDimension(endpoint="endpoint_A", failure_rate=0.02, recovery_priority=0.9),
        StateDimension(endpoint="endpoint_B", failure_rate=0.05, recovery_priority=0.6),
        StateDimension(endpoint="endpoint_C", failure_rate=0.01, recovery_priority=0.95),
    ]

    # Random tropical network compatible with 3 dimensions
    rng = np.random.default_rng(42)
    weights = rng.normal(size=(3, 3))
    biases = rng.normal(size=3)
    tropical_net = TropicalNetwork(weights=weights, biases=biases)

    # Sample counts (e.g., number of observations per endpoint)
    sample_counts = [20, 15, 5]

    # Perform hybrid decision
    gains, bounds = hybrid_split_decision(
        state_dims=endpoints,
        tropical_net=tropical_net,
        delta=0.05,
        sample_counts=sample_counts,
    )

    # Simple reporting
    for i, dim in enumerate(endpoints):
        decision = "SPLIT" if gains[i] > bounds[i] else "hold"
        print(
            f"Endpoint {dim.endpoint}: gain={gains[i]:.4f}, bound={bounds[i]:.4f} → {decision}"
        )