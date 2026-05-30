# DARWIN HAMMER — match 3296, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_ssim_h_hybrid_hybrid_sheaf__m2667_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2171_s0.py (gen5)
# born: 2026-05-29T23:49:13Z

import uuid
import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, List, Iterable, Optional

import numpy as np


class Multivector:
    """
    Simple geometric algebra multivector implementation.

    Blades are represented as frozensets of basis indices.
    The scalar part uses the empty frozenset ().
    """

    def __init__(self, components: Dict[Tuple[int, ...], float], n: int):
        # Normalize blades to sorted tuples for deterministic ordering
        self.n = int(n)
        self.components: Dict[Tuple[int, ...], float] = {
            tuple(sorted(blade)): float(val)
            for blade, val in components.items()
            if abs(val) > 1e-15
        }

    # --------------------------------------------------------------------- #
    # Basic algebraic operations
    # --------------------------------------------------------------------- #

    def grade(self, k: int) -> "Multivector":
        """Return the grade‑k part of the multivector."""
        return Multivector(
            {
                blade: coef
                for blade, coef in self.components.items()
                if len(blade) == k
            },
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the scalar (grade‑0) component."""
        return self.components.get((), 0.0)

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #

    @staticmethod
    def _blade_sign(blade1: Tuple[int, ...], blade2: Tuple[int, ...]) -> int:
        """
        Compute the sign resulting from the geometric product of two blades.
        The sign is determined by the number of swaps needed to reorder the
        concatenated basis indices into a canonical (sorted) order.
        """
        merged = list(blade1) + list(blade2)
        sign = 1
        # Bubble‑sort counting swaps
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if merged[i] > merged[j]:
                    sign = -sign
        return sign

    @staticmethod
    def _blade_intersection(blade1: Tuple[int, ...], blade2: Tuple[int, ...]) -> Tuple[int, ...]:
        """Return the common indices (used for inner product)."""
        return tuple(sorted(set(blade1) & set(blade2)))

    @staticmethod
    def _blade_symmetric_difference(blade1: Tuple[int, ...], blade2: Tuple[int, ...]) -> Tuple[int, ...]:
        """Return the symmetric difference of two blades (used for outer product)."""
        return tuple(sorted(set(blade1) ^ set(blade2)))

    # --------------------------------------------------------------------- #
    # Algebraic operators
    # --------------------------------------------------------------------- #

    def __add__(self, other: "Multivector") -> "Multivector":
        result: Dict[Tuple[int, ...], float] = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        result: Dict[Tuple[int, ...], float] = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) - coef
        return Multivector(result, self.n)

    def __neg__(self) -> "Multivector":
        return Multivector({blade: -coef for blade, coef in self.components.items()}, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """
        Geometric product (Clifford product) of two multivectors.
        This implementation follows the standard rule:
            e_i e_j = e_i ∧ e_j + g_{ij}
        where we assume an Euclidean metric (g_{ij}=δ_{ij}).
        """
        result: Dict[Tuple[int, ...], float] = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                # Count common indices – they square to +1 in Euclidean space
                common = self._blade_intersection(blade_a, blade_b)
                # Remove common indices (they become scalars)
                reduced_a = tuple(i for i in blade_a if i not in common)
                reduced_b = tuple(i for i in blade_b if i not in common)
                new_blade = self._blade_symmetric_difference(reduced_a, reduced_b)
                sign = self._blade_sign(reduced_a, reduced_b)
                # Each common index contributes a factor of +1 (Euclidean metric)
                factor = 1.0
                result[new_blade] = result.get(new_blade, 0.0) + sign * factor * coef_a * coef_b
        return Multivector(result, self.n)

    def wedge(self, other: "Multivector") -> "Multivector":
        """Exterior (wedge) product – antisymmetric part of the geometric product."""
        result: Dict[Tuple[int, ...], float] = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                if set(blade_a) & set(blade_b):
                    # Overlapping blades wedge to zero
                    continue
                new_blade = tuple(sorted(blade_a + blade_b))
                sign = self._blade_sign(blade_a, blade_b)
                result[new_blade] = result.get(new_blade, 0.0) + sign * coef_a * coef_b
        return Multivector(result, self.n)

    def inner(self, other: "Multivector") -> "Multivector":
        """Inner product – grade‑lowering part of the geometric product."""
        result: Dict[Tuple[int, ...], float] = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                # Intersection must be non‑empty and the result grade = |a| - |b|
                common = self._blade_intersection(blade_a, blade_b)
                if not common:
                    continue
                reduced_a = tuple(i for i in blade_a if i not in common)
                reduced_b = tuple(i for i in blade_b if i not in common)
                new_blade = self._blade_symmetric_difference(reduced_a, reduced_b)
                sign = self._blade_sign(reduced_a, reduced_b)
                result[new_blade] = result.get(new_blade, 0.0) + sign * coef_a * coef_b
        return Multivector(result, self.n)

    # --------------------------------------------------------------------- #
    # Representation
    # --------------------------------------------------------------------- #

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            if blade:
                label = "e" + "".join(str(i) for i in blade)
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


class PheromoneEntry:
    """
    Represents a pheromone signal attached to a graph node or edge.
    The signal decays exponentially according to its half‑life.
    """

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
        self.uuid: str = str(uuid.uuid4())
        self.surface_key: str = surface_key
        self.signal_kind: str = signal_kind
        self.signal_value: float = float(signal_value)
        self.half_life_seconds: int = int(half_life_seconds)
        now = datetime.now(timezone.utc)
        self.created_at: datetime = now
        self.last_decay: datetime = now

    def age_seconds(self) -> float:
        """Seconds elapsed since the last decay update."""
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay(self) -> None:
        """Apply exponential decay based on elapsed time and half‑life."""
        elapsed = self.age_seconds()
        if elapsed <= 0 or self.half_life_seconds <= 0:
            return
        decay_factor = 0.5 ** (elapsed / self.half_life_seconds)
        self.signal_value *= decay_factor
        self.last_decay = datetime.now(timezone.utc)


# --------------------------------------------------------------------- #
# Fusion utilities
# --------------------------------------------------------------------- #

def sheaf_cohomology(pheromone_signals: Iterable[PheromoneEntry]) -> float:
    """
    Compute a simple sheaf‑like statistic: the average of current signal values.
    Returns 0.0 for an empty collection to avoid division by zero.
    """
    values = [sig.signal_value for sig in pheromone_signals]
    return float(np.mean(values)) if values else 0.0


def filter_signals(
    pheromone_signals: List[PheromoneEntry],
    threshold: float,
    apply_decay: bool = True,
) -> List[PheromoneEntry]:
    """
    Optionally decay each signal and keep only those whose (decayed) value exceeds
    the threshold.
    """
    filtered: List[PheromoneEntry] = []
    for sig in pheromone_signals:
        if apply_decay:
            sig.decay()
        if sig.signal_value > threshold:
            filtered.append(sig)
    return filtered


def analyze_consistency(
    pheromone_signals: List[PheromoneEntry],
    base_multivector: Multivector,
    weight_by_cohomology: bool = True,
) -> Multivector:
    """
    Fuse pheromone signals into the geometric algebra domain.

    Each signal is turned into a scalar multivector (grade‑0) after decay.
    The scalar is optionally scaled by a global sheaf‑cohomology factor to
    capture collective consistency.
    The resulting multivector is the geometric product of the base multivector
    with the sum of all signal scalars.
    """
    # Ensure signals are up‑to‑date
    for sig in pheromone_signals:
        sig.decay()

    # Global scaling factor from sheaf cohomology
    global_factor = sheaf_cohomology(pheromone_signals) if weight_by_cohomology else 1.0

    # Accumulate scalar contribution
    scalar_sum = 0.0
    for sig in pheromone_signals:
        scalar_sum += sig.signal_value

    weighted_scalar = scalar_sum * global_factor
    signal_mv = Multivector({(): weighted_scalar}, base_multivector.n)

    return base_multivector * signal_mv


# --------------------------------------------------------------------- #
# Demonstration / simple test harness
# --------------------------------------------------------------------- #

if __name__ == "__main__":
    # Define a 5‑dimensional Euclidean space multivector
    base_mv = Multivector({(1, 2): 0.5, (3, 4): 0.3}, 5)

    # Create a few pheromone entries with varying values and half‑lives
    pheromones = [
        PheromoneEntry("node_A", "type_X", 0.8, 3600),
        PheromoneEntry("node_B", "type_X", 0.4, 3600),
        PheromoneEntry("node_C", "type_X", 0.9, 3600),
    ]

    # Fuse signals into the geometric algebra domain
    fused = analyze_consistency(pheromones, base_mv)
    print("Fused multivector:", fused)

    # Filter signals with a threshold after decay
    kept = filter_signals(pheromones, threshold=0.5)
    print("Kept signal values:", [s.signal_value for s in kept])

    # Compute sheaf‑cohomology statistic
    cohom = sheaf_cohomology(pheromones)
    print("Sheaf cohomology (average signal):", cohom)