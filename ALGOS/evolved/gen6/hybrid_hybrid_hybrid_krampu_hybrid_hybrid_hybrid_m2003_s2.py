# DARWIN HAMMER — match 2003, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_krampus_stick_hybrid_hybrid_hybrid_m1008_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m599_s1.py (gen5)
# born: 2026-05-29T23:40:22Z

"""Hybrid Pheromone‑Endpoint Bandit Algorithm

Parents:
- **Parent A** – PheromoneStore with exponential decay (half‑life model).
- **Parent B** – Endpoint health scores, Hoeffding‑bounded bandit selection, SSIM similarity.

Mathematical bridge:
The decayed pheromone signal vector on a given surface is interpreted as a
*context vector* C ∈ ℝⁿ.  Each Endpoint i possesses a health_score h_i that is
combined with C to form a *context‑aware score* s_i = h_i · (C·w_i) where w_i
is a learned weight (here taken as the normalized pheromone vector).  The
bandit algorithm uses Hoeffding’s inequality to compute an upper confidence
bound U_i = s_i + B_i (B_i = hoeffding_bound(R, δ, n_i)).  The endpoint with
max U_i is selected.  The selected endpoint then drives a ternary router whose
input‑output similarity is measured by SSIM; this similarity is fed back as a
reward that updates both the bandit statistics and the pheromone store (a new
PheromoneEntry with signal_value proportional to the SSIM reward).

The module implements:
1. Pheromone decay and retrieval (Parent A).
2. Endpoint health handling, Hoeffding bound, SSIM (Parent B).
3. Hybrid functions that fuse the two models as described above.
"""

import math
import random
import threading
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import sys

# --------------------------------------------------------------------------- #
# Parent A – Pheromone infrastructure
# --------------------------------------------------------------------------- #

Vector = List[float]


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

    def __init__(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: int,
    ) -> None:
        self.uuid = f"{random.getrandbits(128):032x}"
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = float(signal_value)
        self.half_life_seconds = max(1, half_life_seconds)  # avoid division by zero
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    # --------------------------------------------------------------------- #
    # Decay utilities
    # --------------------------------------------------------------------- #
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Exponential decay based on half‑life."""
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """Thread‑safe global store for pheromone entries."""

    _entries: Dict[str, PheromoneEntry] = {}
    _lock = threading.RLock()

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        with cls._lock:
            cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        with cls._lock:
            return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_surface(cls, surface_key: str) -> List[Dict[str, float]]:
        """Apply decay to all entries of a surface and return a log."""
        log = []
        with cls._lock:
            for e in cls._entries.values():
                if e.surface_key == surface_key:
                    before = e.signal_value
                    e.apply_decay()
                    log.append(
                        {
                            "uuid": e.uuid,
                            "before": before,
                            "after": e.signal_value,
                            "decay_factor": e.decay_factor(),
                        }
                    )
        return log

    @classmethod
    def vector_for_surface(cls, surface_key: str) -> np.ndarray:
        """Return a normalized vector of current signal values for a surface."""
        entries = cls.get_by_surface(surface_key)
        if not entries:
            return np.array([], dtype=float)
        values = np.array([e.signal_value for e in entries], dtype=float)
        # Apply decay before reading
        for e in entries:
            e.apply_decay()
        # Normalize to unit L2 norm (avoid division by zero)
        norm = np.linalg.norm(values)
        return values / norm if norm > 0 else values


# --------------------------------------------------------------------------- #
# Parent B – Endpoint, Hoeffding, SSIM, Bandit
# --------------------------------------------------------------------------- #

@dataclass
class Endpoint:
    identifier: str
    health_score: float
    failure_rate: float
    recovery_priority: float


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for bounded reward r ∈ [0, R]."""
    if n == 0:
        return float("inf")
    return math.sqrt((r ** 2 * math.log(2 / delta)) / (2 * n))


def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index Measure for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    cov_xy = np.mean((x - mean_x) * (y - mean_y))
    var_x = np.mean((x - mean_x) ** 2)
    var_y = np.mean((y - mean_y) ** 2)
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    luminance = (2 * mean_x * mean_y + c1) / (mean_x ** 2 + mean_y ** 2 + c1)
    contrast = (2 * np.sqrt(var_x) * np.sqrt(var_y) + c2) / (var_x + var_y + c2)
    structure = (cov_xy + c2 / 2) / (np.sqrt(var_x) * np.sqrt(var_y) + c2 / 2)
    return luminance * contrast * structure


class SimpleBandit:
    """Epsilon‑greedy bandit with Hoeffding‑based confidence bounds."""

    def __init__(self, endpoints: List[Endpoint], delta: float = 0.05):
        self.delta = delta
        self.counts: Dict[str, int] = defaultdict(int)
        self.values: Dict[str, float] = defaultdict(float)
        self.endpoints = {e.identifier: e for e in endpoints}

    def expected_reward(self, identifier: str) -> float:
        return self.values[identifier] / self.counts[identifier] if self.counts[identifier] > 0 else 0.0

    def upper_confidence(self, identifier: str) -> float:
        n = self.counts[identifier]
        mu = self.expected_reward(identifier)
        bound = hoeffding_bound(r=1.0, delta=self.delta, n=n)  # reward bounded in [0,1]
        return mu + bound

    def select(self, context_vector: np.ndarray) -> Endpoint:
        """Select endpoint using context‑aware scores + Hoeffding UCB."""
        # Compute raw context‑aware scores: h_i * (w·C)
        # w is the normalized pheromone vector for the surface associated with the endpoint.
        # For simplicity we reuse the same context_vector for all endpoints.
        scores = {}
        for ep in self.endpoints.values():
            # health_score already normalised in [0,1] in typical use‑cases
            scores[ep.identifier] = ep.health_score * np.mean(context_vector) if context_vector.size else ep.health_score

        # Blend with Hoeffding upper confidence bound
        ucb_scores = {
            eid: scores[eid] + self.upper_confidence(eid) for eid in scores
        }
        chosen_id = max(ucb_scores, key=ucb_scores.get)
        return self.endpoints[chosen_id]

    def update(self, identifier: str, reward: float) -> None:
        self.counts[identifier] += 1
        self.values[identifier] += reward


# --------------------------------------------------------------------------- #
# Hybrid functions (core of the fusion)
# --------------------------------------------------------------------------- #

def compute_context_vector(surface_key: str) -> np.ndarray:
    """
    Retrieve the decayed pheromone vector for a surface and normalise it.
    This vector serves as the context C for the bandit decision.
    """
    vec = PheromoneStore.vector_for_surface(surface_key)
    # If no pheromones exist, fall back to a uniform vector of length 1.
    if vec.size == 0:
        return np.ones(1, dtype=float)
    return vec


def select_endpoint_hybrid(surface_key: str, bandit: SimpleBandit) -> Endpoint:
    """
    Combine pheromone context with endpoint health scores and select an endpoint.
    """
    context = compute_context_vector(surface_key)
    chosen = bandit.select(context)
    return chosen


def route_and_update(
    surface_key: str,
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    bandit: SimpleBandit,
    reward_scale: float = 10.0,
) -> Tuple[Endpoint, float]:
    """
    Perform a ternary routing step:
    1. Choose an endpoint via the hybrid bandit.
    2. Compute SSIM similarity between input and output signals.
    3. Feed the similarity as a reward to the bandit.
    4. Insert a new pheromone entry whose value is proportional to the reward.
    Returns the selected endpoint and the reward used.
    """
    # 1. Endpoint selection
    endpoint = select_endpoint_hybrid(surface_key, bandit)

    # 2. Similarity (reward) computation
    reward = ssim(input_signal, output_signal)  # in [0,1] for typical signals
    scaled_reward = reward * reward_scale

    # 3. Bandit update
    bandit.update(endpoint.identifier, reward)

    # 4. Pheromone reinforcement
    entry = PheromoneEntry(
        surface_key=surface_key,
        signal_kind="reward",
        signal_value=scaled_reward,
        half_life_seconds=30,  # arbitrary half‑life for reward pheromones
    )
    PheromoneStore.add(entry)

    return endpoint, reward


def decay_all_surfaces(surface_keys: List[str]) -> None:
    """
    Apply decay to every listed surface.  This function demonstrates bulk
    management of the pheromone store, a necessary maintenance step before
    subsequent routing cycles.
    """
    for key in surface_keys:
        PheromoneStore.decay_surface(key)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Create a few dummy pheromone entries
    for i in range(5):
        PheromoneStore.add(
            PheromoneEntry(
                surface_key="surface_A",
                signal_kind="init",
                signal_value=random.uniform(0.5, 1.5),
                half_life_seconds=random.randint(20, 60),
            )
        )

    # Define endpoints
    endpoints = [
        Endpoint(identifier=f"ep_{i}", health_score=random.uniform(0.2, 1.0),
                 failure_rate=random.uniform(0.0, 0.3),
                 recovery_priority=random.uniform(0.1, 0.9))
        for i in range(3)
    ]

    # Initialise hybrid bandit
    bandit = SimpleBandit(endpoints)

    # Dummy input/output signals
    inp = np.linspace(0, 255, 100) + np.random.normal(0, 5, 100)
    outp = inp * 0.9 + np.random.normal(0, 5, 100)  # slightly degraded version

    # Run a routing cycle
    chosen_ep, reward = route_and_update(
        surface_key="surface_A",
        input_signal=inp,
        output_signal=outp,
        bandit=bandit,
    )
    print(f"Chosen endpoint: {chosen_ep.identifier}, SSIM reward: {reward:.4f}")

    # Decay surfaces to show maintenance
    decay_all_surfaces(["surface_A"])
    # Verify that pheromone vectors still exist
    vec = compute_context_vector("surface_A")
    print(f"Context vector (size {vec.size}): {vec[:5]} ...")