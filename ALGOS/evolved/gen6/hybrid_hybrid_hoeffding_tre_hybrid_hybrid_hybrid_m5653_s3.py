# DARWIN HAMMER — match 5653, survivor 3
# gen: 6
# parent_a: hybrid_hoeffding_tree_gini_coefficient_m13_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m2606_s3.py (gen5)
# born: 2026-05-30T00:03:58Z

"""Hybrid algorithm combining Hoeffding‑Tree split statistics with pheromone‑driven
entropy modulation.

Parent A:   hoeffding_tree + Gini coefficient (splitting criterion)
Parent B:   surface pheromone store with exponential decay (entropy‑like signal)

Mathematical bridge:
The Hoeffding bound  ε = sqrt( (R²·ln(1/δ)) / (2·n) )  depends on the confidence
parameter δ.  In a pheromone‑controlled environment the confidence can be
adapted dynamically: a fresh, strong pheromone signal (high signal_value) implies
high certainty (small δ), whereas an aged, decayed signal implies lower certainty
(larger δ).  Consequently the bound, the Gini‑weighted gain gap and the split
decision become functions of both the data distribution (Gini) and the pheromone
state (decay‑modulated δ).

The three core functions below illustrate this fusion:
* `hoeffding_bound_pheromone` – computes ε using a δ derived from a pheromone entry.
* `should_split_hybrid` – decides whether to split a node using Gini‑weighted
  gain and the pheromone‑adjusted bound.
* `update_pheromone_from_split` – reinforces the pheromone associated with a
  surface when a split occurs, otherwise lets it decay.

The module is self‑contained and executable."""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from collections.abc import Iterable

# ----------------------------------------------------------------------
# Pheromone infrastructure (derived from Parent B)
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    _uid_counter = 0

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        # simple deterministic uid without external libs
        PheromoneEntry._uid_counter += 1
        self.uuid = f"ph-{PheromoneEntry._uid_counter}"
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = float(signal_value)
        self.half_life_seconds = int(half_life_seconds)
        now = pathlib.Path(__file__).stat().st_mtime
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (pathlib.Path(__file__).stat().st_mtime - self.last_decay)

    def decay_factor(self) -> float:
        """Multiplicative decay factor since last_decay (exponential half‑life)."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = pathlib.Path(__file__).stat().st_mtime


class PheromoneStore:
    """In‑memory singleton store for pheromone entries."""
    _entries: dict = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> list[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_surface(cls, surface_key: str) -> None:
        for entry in cls.get_by_surface(surface_key):
            entry.apply_decay()


# ----------------------------------------------------------------------
# Gini coefficient (derived from Parent A)
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0.0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


# ----------------------------------------------------------------------
# Hybrid Hoeffding bound that incorporates pheromone strength
# ----------------------------------------------------------------------
def hoeffding_bound_pheromone(r: float, base_delta: float,
                              n: int, pheromone: PheromoneEntry) -> float:
    """
    Compute Hoeffding bound ε where the confidence δ is scaled by the current
    pheromone signal.  Stronger signal ⇒ smaller effective δ ⇒ tighter bound.
    """
    if r <= 0 or not (0 < base_delta < 1) or n <= 0:
        raise ValueError("Invalid parameters for Hoeffding bound")
    # Apply decay first so the signal reflects current strength
    pheromone.apply_decay()
    # Map signal_value ∈ (0, ∞) to a scaling factor in (0,1]
    # We cap the scaling at 1.0 to avoid δ > base_delta.
    scale = min(1.0, pheromone.signal_value / (pheromone.signal_value + 1.0))
    effective_delta = base_delta * (1.0 - scale) + 1e-9  # avoid log(0)
    return math.sqrt((r * r * math.log(1.0 / effective_delta)) / (2.0 * n))


# ----------------------------------------------------------------------
# Decision data class
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    weighted_gap: float
    reason: str


# ----------------------------------------------------------------------
# Hybrid split decision using Gini and pheromone‑adjusted bound
# ----------------------------------------------------------------------
def should_split_hybrid(best_gain: float, second_best_gain: float,
                        r: float, base_delta: float, n: int,
                        values: Iterable[float],
                        pheromone: PheromoneEntry,
                        tie_threshold: float = 0.05) -> SplitDecision:
    """
    Determine whether a Hoeffding tree node should split.
    - Gini coefficient measures inequality of the attribute distribution.
    - The gain gap is weighted by (1 - Gini) to favour homogeneous splits.
    - Hoeffding bound ε uses a pheromone‑scaled confidence.
    """
    gini = gini_coefficient(values)
    eps = hoeffding_bound_pheromone(r, base_delta, n, pheromone)
    gap = best_gain - second_best_gain
    weighted_gap = gap * (1.0 - gini)

    split = weighted_gap > eps or eps < tie_threshold
    if weighted_gap > eps:
        reason = "weighted_gap_exceeds_bound"
    elif eps < tie_threshold:
        reason = "epsilon_below_tie_threshold"
    else:
        reason = "await_more_data"

    return SplitDecision(split, eps, weighted_gap, reason)


# ----------------------------------------------------------------------
# Reinforce pheromone when a split succeeds
# ----------------------------------------------------------------------
def update_pheromone_from_split(surface_key: str,
                                succeeded: bool,
                                reward: float = 0.1,
                                decay_half_life: int = 30) -> None:
    """
    If a split occurs (succeeded=True) we either create a new pheromone entry
    or boost the existing one.  If no split occurs we simply let the existing
    pheromones decay (handled elsewhere).
    """
    entries = PheromoneStore.get_by_surface(surface_key)
    if succeeded:
        if entries:
            # reinforce the strongest existing entry
            entry = max(entries, key=lambda e: e.signal_value)
            entry.signal_value += reward
        else:
            # create a fresh entry
            entry = PheromoneEntry(surface_key=surface_key,
                                   signal_kind="split_success",
                                   signal_value=reward,
                                   half_life_seconds=decay_half_life)
            PheromoneStore.add(entry)
    else:
        # No split – let all entries decay naturally
        PheromoneStore.decay_surface(surface_key)


# ----------------------------------------------------------------------
# Example utility: compute a pheromone‑weighted split point
# ----------------------------------------------------------------------
def pheromone_weighted_split_point(values: Iterable[float],
                                   r: float, base_delta: float, n: int,
                                   pheromone: PheromoneEntry) -> float:
    """
    Return a scalar that could serve as a split threshold.
    It combines the Gini coefficient with the pheromone‑adjusted Hoeffding bound.
    """
    gini = gini_coefficient(values)
    eps = hoeffding_bound_pheromone(r, base_delta, n, pheromone)
    # Simple heuristic: move the split point proportionally to (1‑gini)·eps
    return (1.0 - gini) * eps


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # synthetic data
    np.random.seed(0)
    values = np.random.exponential(scale=1.0, size=100)

    # parameters
    r = 1.0
    base_delta = 0.05
    n = len(values)

    # create a pheromone entry for a hypothetical surface
    surface = "node-42"
    pheromone = PheromoneEntry(surface_key=surface,
                               signal_kind="init",
                               signal_value=0.8,
                               half_life_seconds=60)
    PheromoneStore.add(pheromone)

    # simulate random gains
    best_gain = random.random()
    second_gain = random.random() * 0.5

    decision = should_split_hybrid(best_gain, second_gain,
                                   r, base_delta, n,
                                   values, pheromone)

    print("SplitDecision:", decision)

    # update pheromone based on decision
    update_pheromone_from_split(surface, decision.should_split)

    # compute a split point
    split_pt = pheromone_weighted_split_point(values, r, base_delta, n, pheromone)
    print("Pheromone‑weighted split point:", split_pt)

    # ensure decay works without error
    PheromoneStore.decay_surface(surface)
    print("Post‑decay signal value:",
          PheromoneStore.get_by_surface(surface)[0].signal_value)