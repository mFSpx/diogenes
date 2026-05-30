# DARWIN HAMMER — match 4565, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_semant_m431_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2010_s3.py (gen4)
# born: 2026-05-29T23:56:32Z

"""Hybrid Geometric‑Temporal Bandit Fusion

Parents:
- hybrid_hybrid_geometric_pro_hybrid_hybrid_semant_m431_s0.py (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2010_s3.py (Algorithm B)

Mathematical bridge:
Algorithm A provides a Euclidean squared distance between two grade‑1 multivectors
(2‑D points) and a temporal motif support value.  Algorithm B updates a
contextual bandit where each propensity is scaled by an epistemic certainty flag.
The fusion treats the motif support as an epistemic certainty factor and uses the
spatial distance as an additional scaling term.  Concretely, for a bandit update
we compute

    w = d²(p_i, p_j) · (1 + motif.support)⁻¹

and modify the original propensity π as

    π′ = π · exp(‑w) · (1 + store.dance)

Thus the geometric topology (distance) and the temporal‑semantic topology
(motif support) jointly re‑weight the bandit learning dynamics, while the store
state supplies a bounded control signal that further modulates the update.

The module implements this hybrid system and provides three exemplar functions:
`weighted_distance`, `update_bandit_with_spatial`, and `generate_hybrid_motif`. """

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field, frozen, FrozenInstanceError
from typing import List, Tuple, Dict, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Geometry / Multivector utilities (from Algorithm A)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel duplicate pair (e_i*e_i = 1)
                del lst[j : j + 2]
                n -= 2
                # sign unchanged
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


# ----------------------------------------------------------------------
# Domain dataclasses (shared between parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int  # epistemic certainty analogue


@dataclass(frozen=True)
class HybridMotif:
    """Spatio‑temporal motif enriched with morphology and a geometric vector."""
    pattern: Tuple[str, ...]
    support: int
    centroid_lat: float
    centroid_lon: float
    morphology: Morphology
    vector: Tuple[float, ...]          # semantic feature vector
    score: float                       # fused quality metric


# ----------------------------------------------------------------------
# Bandit / Store components (from Algorithm B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Honeybee‑style store that yields a bounded control signal."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Apply the store equation and recompute the dance duration."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self._last_delta = delta
        self.level = max(0.0, self.level + self.dt * delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ (computed lazily)."""
        delta = getattr(self, "_last_delta", 0.0)
        return max(0.0, min(self.limit, self.base + self.gain * delta))


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def weighted_distance(p1: Tuple[float, float],
                      p2: Tuple[float, float],
                      motif: TemporalMotif) -> float:
    """
    Euclidean squared distance between two 2‑D points scaled by the inverse
    of the motif support (treated as epistemic certainty).

    Parameters
    ----------
    p1, p2 : (x, y) coordinates.
    motif : TemporalMotif providing the support factor.

    Returns
    -------
    float : weighted distance.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    euclidean_sq = dx * dx + dy * dy
    # Avoid division by zero; support >= 0 by definition.
    scale = 1.0 / (1.0 + motif.support)
    return euclidean_sq * scale


def update_bandit_with_spatial(action: BanditAction,
                               update: BanditUpdate,
                               store: StoreState,
                               motif: TemporalMotif,
                               p_src: Tuple[float, float],
                               p_dst: Tuple[float, float]) -> BanditAction:
    """
    Fuse geometric distance and temporal motif into the bandit propensity.

    The new propensity is:
        π′ = π · exp(‑w) · (1 + store.dance)

    where w is the weighted distance from `weighted_distance`.

    Returns a new immutable BanditAction.
    """
    w = weighted_distance(p_src, p_dst, motif)
    modulation = math.exp(-w) * (1.0 + store.dance)
    new_propensity = max(0.0, action.propensity * modulation)

    # Re‑compute a simple confidence bound as inverse of the new propensity
    new_confidence = 1.0 / (new_propensity + 1e-8)

    return BanditAction(
        action_id=action.action_id,
        propensity=new_propensity,
        expected_reward=action.expected_reward,  # unchanged for this demo
        confidence_bound=new_confidence,
        algorithm=action.algorithm + "+Hybrid"
    )


def generate_hybrid_motif(point: Tuple[float, float],
                          morphology: Morphology,
                          pattern: Tuple[str, ...],
                          support: int,
                          store: StoreState) -> HybridMotif:
    """
    Construct a HybridMotif from geometric, semantic and store information.

    The score combines:
        - store.dance (control signal)
        - motif support
        - normalized morphological volume (length·width·height)

    Returns a fully populated HybridMotif.
    """
    # Simple semantic vector derived from morphology attributes
    vector = (morphology.length, morphology.width, morphology.height, morphology.mass)

    # Normalized volume for scoring
    volume = morphology.length * morphology.width * morphology.height
    norm_volume = volume / (1.0 + volume)

    score = (store.dance * (1.0 + support) * norm_volume)

    return HybridMotif(
        pattern=pattern,
        support=support,
        centroid_lat=point[0],
        centroid_lon=point[1],
        morphology=morphology,
        vector=vector,
        score=score
    )


def simulate_hybrid_step(points: List[Tuple[float, float]],
                         morphologies: List[Morphology],
                         patterns: List[Tuple[str, ...]],
                         supports: List[int],
                         actions: List[BanditAction],
                         store: StoreState) -> List[HybridMotif]:
    """
    Demonstrates the full hybrid pipeline:
    1. Create a TemporalMotif from each support.
    2. Update each BanditAction using the geometric relationship to the next point.
    3. Generate a HybridMotif that encodes the fused state.

    Returns the list of HybridMotif objects.
    """
    hybrid_motifs: List[HybridMotif] = []

    # Simple store dynamics: random inflow/outflow to keep dance non‑trivial
    inflow = [random.random() for _ in range(3)]
    outflow = [random.random() for _ in range(2)]
    store.update(inflow, outflow)

    n = len(points)
    for i in range(n):
        p_src = points[i]
        p_dst = points[(i + 1) % n]  # wrap‑around for demo
        motif = TemporalMotif(pattern=patterns[i], support=supports[i])

        # Update corresponding bandit action (assume same length)
        action = actions[i % len(actions)]
        # Fake observation for update (reward proportional to support)
        upd = BanditUpdate(
            context_id=f"ctx_{i}",
            action_id=action.action_id,
            reward=float(supports[i]),
            propensity=action.propensity
        )
        new_action = update_bandit_with_spatial(action, upd, store, motif, p_src, p_dst)

        # Produce hybrid motif using the updated store state
        hybrid = generate_hybrid_motif(
            point=p_src,
            morphology=morphologies[i % len(morphologies)],
            pattern=patterns[i],
            support=supports[i],
            store=store
        )
        hybrid_motifs.append(hybrid)

        # For demonstration, replace the old action with the new one (not used later)
        actions[i % len(actions)] = new_action

    return hybrid_motifs


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic data
    pts = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(5)]
    morphs = [Morphology(length=random.uniform(0.5, 2.0),
                         width=random.uniform(0.5, 2.0),
                         height=random.uniform(0.5, 2.0),
                         mass=random.uniform(1.0, 5.0))
              for _ in range(5)]
    pats = [("A", "B"), ("C",), ("D", "E", "F"), ("G",), ("H", "I")]
    sups = [random.randint(0, 5) for _ in range(5)]
    actions = [BanditAction(action_id=f"act_{i}",
                            propensity=1.0,
                            expected_reward=0.0,
                            confidence_bound=1.0,
                            algorithm="BaseBandit")
               for i in range(5)]
    store = StoreState()

    hybrids = simulate_hybrid_step(pts, morphs, pats, sups, actions, store)

    # Print a concise summary
    for i, hm in enumerate(hybrids):
        print(f"Hybrid {i}: centroid=({hm.centroid_lat:.2f},{hm.centroid_lon:.2f}) "
              f"score={hm.score:.4f} pattern={hm.pattern}")
    print("Smoke test completed without errors.")