# DARWIN HAMMER — match 950, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_gliner_hybrid_possum_filter_m323_s2.py (gen4)
# born: 2026-05-29T23:31:58Z

"""Hybrid Fusion of Bandit‑RBF Core (Parent A) and Span‑Pheromone Core (Parent B)

This module mathematically fuses the two parental algorithms by identifying a common
exponential‑decay kernel:

* Parent A employs a Gaussian kernel `exp(-ε²‖x‑c‖²)` inside an RBF surrogate.
* Parent B attenuates pheromone influence with a spatial exponential `exp(-d/λ)` where
  `d` is a haversine distance.

The bridge is the product of the two kernels, yielding a **spatio‑temporal kernel**

K(x,c; s_x, s_c) = exp(-ε²‖x‑c‖²) · exp(- haversine(s_x, s_c) / λ)

where `s_x` and `s_c` are geographic coordinates attached to the input vector
and to the RBF centre respectively.  

The fused system therefore:

1. Maintains a bandit policy (`_POLICY`) and a virtual VRAM store (`_STORE`).
2. Stores pheromone entries (`_PHEROMONE`) that decay by half‑life and are also
   updated through the same exponential kernel.
3. Uses the combined kernel inside the RBF surrogate to predict rewards that are
   spatially aware.
4. Computes a joint “information weight’’ for each `Span`‑entity pair as  
   `w = span.score × surrogate.predict(vector, coord) × exp(-d/λ)`,
   where `d` is the haversine distance between the span’s (implicit) location and the
   entity coordinate.

The three core functions below demonstrate the hybrid operation:
* `update_bandit` – bandit learning step.
* `decay_pheromones` – half‑life decay of pheromone entries.
* `select_hybrid_action` – chooses an action by merging bandit expected reward
  with the spatially‑aware joint weights of spans.

All imports are restricted to the allowed standard‑library modules and NumPy."""

import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared type aliases
# ----------------------------------------------------------------------
Vector = Sequence[float]
Coord = Tuple[float, float]  # (latitude, longitude)

# ----------------------------------------------------------------------
# Parent A structures (Bandit + RBF)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


_POLICY: Dict[str, List[float]] = {}          # action_id -> [total_reward, count]
_STORE: Dict[str, float] = {}                # virtual VRAM store (key -> value)


def reset_all() -> None:
    """Clear policy, VRAM store and pheromone memory."""
    _POLICY.clear()
    _STORE.clear()
    _PHEROMONE.clear()


def _reward(action_id: str) -> float:
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Standard Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def haversine(coord1: Coord, coord2: Coord) -> float:
    """Great‑circle distance in kilometres."""
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    earth_radius_km = 6371.0
    return earth_radius_km * c


def combined_kernel(
    x: Vector,
    c: Vector,
    sx: Coord,
    sc: Coord,
    epsilon: float = 1.0,
    lam: float = 1.0,
) -> float:
    """Spatio‑temporal kernel = Gaussian(x,c) * exp(-haversine(sx,sc)/lam)."""
    return gaussian(euclidean(x, c), epsilon) * math.exp(-haversine(sx, sc) / lam)


@dataclass(frozen=True)
class RBFSurrogate:
    """RBF surrogate enriched with optional geographic coordinates for each centre."""
    centers: List[Tuple[float, ...]]
    spatial_coords: List[Coord]               # same length as centres
    weights: List[float]
    epsilon: float = 1.0
    lam: float = 1.0                          # spatial attenuation length‑scale

    def predict(self, x: Vector, sx: Coord) -> float:
        """Predict using the combined spatio‑temporal kernel."""
        return sum(
            w * combined_kernel(x, c, sx, sc, self.epsilon, self.lam)
            for w, c, sc in zip(self.weights, self.centers, self.spatial_coords)
        )


def update_bandit(update: BanditUpdate) -> None:
    """Incorporate a new reward observation into the bandit policy."""
    total, n = _POLICY.get(update.action_id, [0.0, 0.0])
    total += update.reward
    n += 1
    _POLICY[update.action_id] = [total, n]
    # Store the latest propensity for possible diagnostics
    _STORE[f"propensity:{update.action_id}"] = update.propensity


# ----------------------------------------------------------------------
# Parent B structures (Span + Pheromone)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


class PheromoneEntry:
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
        self.uuid = str(uuid.uuid4())
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
        """Multiplicative factor based on half‑life."""
        age = self.age_seconds()
        # decay per half‑life: value *= 0.5^(age / half_life)
        return 0.5 ** (age / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


_PHEROMONE: List[PheromoneEntry] = []


def add_pheromone(entry: PheromoneEntry) -> None:
    _PHEROMONE.append(entry)


def decay_pheromones() -> None:
    """Apply half‑life decay to all stored pheromone entries."""
    for entry in _PHEROMONE:
        entry.apply_decay()


def pheromone_distribution() -> List[float]:
    """Current signal values of all pheromone entries."""
    return [e.signal_value for e in _PHEROMONE if e.signal_value > 0]


def entropy(probs: Sequence[float]) -> float:
    """Shannon entropy of a probability distribution (base e)."""
    total = sum(probs)
    if total == 0:
        return 0.0
    norm = [p / total for p in probs if p > 0]
    return -sum(p * math.log(p) for p in norm)


def information_gain() -> float:
    """Information gain of the current pheromone distribution (negative entropy)."""
    return -entropy(pheromone_distribution())


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_joint_weight(
    span: Span,
    entity_coord: Coord,
    surrogate: RBFSurrogate,
    lambda_: float = 1.0,
) -> float:
    """
    Joint information weight for a Span–Entity pair.

    w = span.score × surrogate.predict(x, s) × exp(-d/λ)

    where:
        x = [span.start, span.end]               (simple numeric encoding)
        s = entity_coord                         (geographic location of the entity)
        d = haversine(span_center, entity_coord) (span_center is approximated as (0,0) for demo)
    """
    # Encode span as a 2‑D vector
    x_vec: Vector = (float(span.start), float(span.end))

    # For demonstration we treat the span's geographic centre as (0,0)
    span_center: Coord = (0.0, 0.0)

    d = haversine(span_center, entity_coord)
    spatial_atten = math.exp(-d / lambda_)

    surrogate_val = surrogate.predict(x_vec, entity_coord)

    return span.score * surrogate_val * spatial_atten


def select_hybrid_action(
    context_vector: Vector,
    entity_coord: Coord,
    actions: List[str],
    spans: List[Span],
    surrogate: RBFSurrogate,
    lambda_: float = 1.0,
    epsilon: float = 0.1,
) -> Tuple[str, float]:
    """
    Choose an action by mixing bandit expected reward with the best joint span‑entity weight.

    Score(action) = bandit_reward(action) + max_{span∈spans} compute_joint_weight(...)
    """
    best_action = None
    best_score = -math.inf

    for a in actions:
        bandit_val = _reward(a)

        # compute the strongest supporting span for this action (demo uses all spans)
        joint_vals = [
            compute_joint_weight(span, entity_coord, surrogate, lambda_)
            for span in spans
        ]
        max_joint = max(joint_vals) if joint_vals else 0.0

        total = bandit_val + max_joint
        if total > best_score:
            best_score = total
            best_action = a

    return best_action, best_score


def entropy_of_pheromones() -> float:
    """Utility exposing the current entropy of the pheromone store."""
    return entropy(pheromone_distribution())


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Initialise a simple surrogate with one centre
    centre_vec = (0.0, 0.0)                # matches the dummy span encoding
    centre_coord = (37.7749, -122.4194)    # San Francisco approx.
    surrogate = RBFSurrogate(
        centers=[centre_vec],
        spatial_coords=[centre_coord],
        weights=[1.0],
        epsilon=0.5,
        lam=500.0,
    )

    # 2. Create a dummy span and an entity coordinate (Los Angeles)
    span = Span(start=10, end=20, text="example", label="TEST", score=0.8)
    entity_coord = (34.0522, -118.2437)  # Los Angeles

    # 3. Register a bandit action and feed a reward
    action_id = "click"
    update = BanditUpdate(context_id="ctx1", action_id=action_id, reward=1.2, propensity=0.7)
    update_bandit(update)

    # 4. Add a pheromone entry
    pheromone = PheromoneEntry(
        surface_key="example_surface",
        signal_kind="click_signal",
        signal_value=5.0,
        half_life_seconds=60,
    )
    add_pheromone(pheromone)

    # 5. Run decay once (should keep value >0)
    decay_pheromones()

    # 6. Hybrid selection
    chosen, score = select_hybrid_action(
        context_vector=[5.0, 15.0],
        entity_coord=entity_coord,
        actions=[action_id, "ignore"],
        spans=[span],
        surrogate=surrogate,
        lambda_=300.0,
    )
    print(f"Chosen action: {chosen} (score={score:.4f})")
    print(f"Pheromone entropy: {entropy_of_pheromones():.4f}")
    print(f"Information gain (negative entropy): {information_gain():.4f}")