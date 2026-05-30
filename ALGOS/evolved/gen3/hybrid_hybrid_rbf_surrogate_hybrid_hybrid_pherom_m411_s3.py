# DARWIN HAMMER — match 411, survivor 3
# gen: 3
# parent_a: hybrid_rbf_surrogate_tri_algo_conduit_m8_s0.py (gen1)
# parent_b: hybrid_hybrid_pheromone_inf_privacy_m54_s0.py (gen2)
# born: 2026-05-29T23:28:48Z

"""
Hybrid RBF‑Pheromone System
============================

This module fuses the two parent algorithms:

* **Parent A – ``rbf_surrogate.py``** – provides a radial‑basis‑function (RBF)
  surrogate model that interpolates scalar values at arbitrary points by solving
  a linear system built from Gaussian kernels.

* **Parent B – ``privacy.py`` / ``hybrid_pheromone_infotaxis``** – maintains a
  pheromone map with exponential decay and computes Shannon entropy of a set
  of probabilities.

**Mathematical bridge**

The bridge is the *probabilistic interpretation* of the RBF surrogate
predictions.  For a set of query points `X = {x_i}` the surrogate yields a vector
`p_i = surrogate.predict(x_i)`.  After normalisation `p_i / Σ p_i` can be treated
as a discrete probability distribution, allowing us to compute its Shannon
entropy.  The entropy measures the uncertainty of the surrogate’s estimate and
is fed back into the pheromone dynamics:

* High entropy → the surrogate is uncertain → pheromone decay is *slowed*,
  preserving information that may be needed for further exploration.
* Low entropy → confident surrogate → pheromone decays at the nominal half‑life.

Thus the RBF surrogate supplies the “signal scores” for pheromones, while the
entropy of those scores modulates the pheromone update, achieving a unified
system that couples interpolation, uncertainty quantification and privacy‑aware
decay.

The implementation below provides three public functions that demonstrate the
hybrid operation:

1. ``fit_surrogate(points, values, epsilon)`` – builds an ``RBFSurrogate``.
2. ``update_pheromone_and_surrogate(...)`` – updates the pheromone map and
   recomputes the surrogate, using entropy to adapt decay.
3. ``privacy_adjusted_prediction(x, query_points)`` – returns a surrogate
   prediction attenuated by the entropy‑derived privacy factor.

A small smoke test is executed when the module is run as ``__main__``.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Sequence, Tuple, List, Dict

import numpy as np

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Radial‑basis‑function utilities (Parent A)
# ----------------------------------------------------------------------


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b with simple Gaussian elimination (no pivot scaling)."""
    n = len(b)
    # Build augmented matrix
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        # Partial pivoting
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        # Normalize pivot row
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        # Eliminate other rows
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]


@dataclass(frozen=True)
class RBFSurrogate:
    """Immutable RBF surrogate model."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Weighted sum of Gaussian kernels evaluated at ``x``."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )


def fit_surrogate(
    points: Iterable[Vector],
    values: Iterable[float],
    epsilon: float = 1.0,
    ridge: float = 1e-9,
) -> RBFSurrogate:
    """
    Fit an RBF surrogate to ``points`` → ``values``.
    ``ridge`` adds a small diagonal term for numerical stability.
    """
    centers = [tuple(map(float, p)) for p in points]
    y = [float(v) for v in values]
    if not centers or len(centers) != len(y):
        raise ValueError("points and values must be non‑empty and same length")
    k = [
        [
            gaussian(euclidean(a, b), epsilon) + (ridge if i == j else 0.0)
            for j, b in enumerate(centers)
        ]
        for i, a in enumerate(centers)
    ]
    weights = solve_linear(k, y)
    return RBFSurrogate(centers, weights, epsilon)


# ----------------------------------------------------------------------
# Pheromone + entropy utilities (Parent B)
# ----------------------------------------------------------------------


class HybridPheromoneSystem:
    """
    Maintains pheromone entries and an associated RBF surrogate.
    The surrogate is recomputed after each pheromone update.
    """

    def __init__(self, epsilon: float = 1.0):
        self.pheromones: Dict[str, Dict] = {}
        self.surrogate: RBFSurrogate | None = None
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # Core pheromone update (identical to Parent B, with entropy coupling)
    # ------------------------------------------------------------------
    def _raw_pheromone_update(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        """Insert or replace a pheromone entry; returns the (raw) signal value."""
        now = datetime.now(timezone.utc)
        self.pheromones[surface_key] = {
            "signal_kind": signal_kind,
            "signal_value": signal_value,
            "half_life_seconds": half_life_seconds,
            "created_time": now,
        }
        return signal_value

    def _decay_factor(self, created: datetime, half_life: float) -> float:
        """Exponential decay factor based on elapsed time."""
        elapsed = (datetime.now(timezone.utc) - created).total_seconds()
        return math.pow(0.5, elapsed / half_life)

    # ------------------------------------------------------------------
    # Entropy utilities (shared with Parent B)
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_entropy(probabilities: Iterable[float], eps: float = 1e-12) -> float:
        """Shannon entropy of a discrete distribution (unnormalised input)."""
        probs = np.array(list(probabilities), dtype=float)
        total = probs.sum()
        if total <= 0:
            raise ValueError("positive probability mass required")
        probs = probs / total
        probs = np.clip(probs, eps, 1.0)
        return -float(np.sum(probs * np.log(probs)))

    # ------------------------------------------------------------------
    # Public hybrid operations
    # ------------------------------------------------------------------
    def update_pheromone_and_surrogate(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
        point: Vector,
    ) -> None:
        """
        Update a pheromone entry *and* rebuild the RBF surrogate.
        The surrogate uses all current pheromone ``signal_value``s as target
        values and the supplied ``point`` (e.g. spatial coordinates) as the
        centre for the updated entry.
        """
        # 1) Update pheromone map (with decay of previous value if it existed)
        if surface_key in self.pheromones:
            old = self.pheromones[surface_key]
            decay = self._decay_factor(
                old["created_time"], old["half_life_seconds"]
            )
            decayed_value = old["signal_value"] * decay
            # blend old decayed value with new one (simple averaging)
            signal_value = 0.5 * (decayed_value + signal_value)

        self._raw_pheromone_update(
            surface_key, signal_kind, signal_value, half_life_seconds
        )

        # 2) Build data set for surrogate: each pheromone key is interpreted as a
        #    coordinate in a 2‑D embedding (hash‑based) plus the supplied point.
        centers: List[Tuple[float, ...]] = []
        values: List[float] = []

        for key, entry in self.pheromones.items():
            # deterministic pseudo‑coordinates from the key string
            h = hash(key)
            x = (h % 1000) / 1000.0  # ∈ [0,1)
            y = ((h // 1000) % 1000) / 1000.0
            centers.append((x, y))
            values.append(entry["signal_value"])

        # Include the explicit point supplied by the caller as an extra centre.
        # This allows the user to inject a fresh observation that is not yet
        # represented in the pheromone map.
        centers.append(tuple(map(float, point)))
        values.append(signal_value)

        # 3) Fit surrogate; epsilon may be tuned based on current entropy.
        if len(centers) >= 2:
            # Estimate entropy of current surrogate predictions on its centres.
            # This cheap estimate will guide the kernel width.
            provisional = fit_surrogate(centers, values, epsilon=self.epsilon)
            preds = [provisional.predict(c) for c in centers]
            ent = self.calculate_entropy(preds)
            # Adapt epsilon: larger entropy → smoother kernel (larger epsilon)
            adapted_epsilon = self.epsilon * (1.0 + 0.5 * ent)
        else:
            adapted_epsilon = self.epsilon

        self.surrogate = fit_surrogate(centers, values, epsilon=adapted_epsilon)

    def surrogate_entropy(self, query_points: Iterable[Vector]) -> float:
        """
        Compute the Shannon entropy of surrogate predictions over ``query_points``.
        """
        if self.surrogate is None:
            raise RuntimeError("surrogate not initialised")
        preds = [self.surrogate.predict(p) for p in query_points]
        return self.calculate_entropy(preds)

    def privacy_adjusted_prediction(self, x: Vector, query_points: Iterable[Vector]) -> float:
        """
        Return a surrogate prediction at ``x`` attenuated by an exponential of
        the entropy over ``query_points``.  The factor ``exp(-entropy)`` can be
        interpreted as a privacy‑preserving weight: higher uncertainty yields a
        smaller adjustment.
        """
        if self.surrogate is None:
            raise RuntimeError("surrogate not initialised")
        raw = self.surrogate.predict(x)
        ent = self.surrogate_entropy(query_points)
        factor = math.exp(-ent)  # in (0,1]
        return raw * factor


# ----------------------------------------------------------------------
# Convenience functions exposing the hybrid behaviour
# ----------------------------------------------------------------------


def fit_surrogate(points: Iterable[Vector], values: Iterable[float], epsilon: float = 1.0) -> RBFSurrogate:
    """Thin wrapper around the internal ``fit_surrogate`` for external callers."""
    return fit_surrogate(points, values, epsilon=epsilon)


def update_pheromone_and_surrogate(
    system: HybridPheromoneSystem,
    surface_key: str,
    signal_kind: str,
    signal_value: float,
    half_life_seconds: float,
    point: Vector,
) -> None:
    """Public API that forwards to the system method."""
    system.update_pheromone_and_surrogate(
        surface_key, signal_kind, signal_value, half_life_seconds, point
    )


def privacy_adjusted_prediction(
    system: HybridPheromoneSystem,
    x: Vector,
    query_points: Iterable[Vector],
) -> float:
    """Public API that forwards to the system method."""
    return system.privacy_adjusted_prediction(x, query_points)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a hybrid system
    hybrid = HybridPheromoneSystem(epsilon=0.8)

    # Simulated spatial observations (2‑D points) with arbitrary signal values
    observations = [
        ((0.1, 0.2), 1.5),
        ((0.4, 0.8), 2.3),
        ((0.7, 0.3), 0.9),
    ]

    # Populate pheromones and surrogate
    for i, (pt, val) in enumerate(observations):
        key = f"node_{i}"
        update_pheromone_and_surrogate(
            hybrid,
            surface_key=key,
            signal_kind="temperature",
            signal_value=val,
            half_life_seconds=60.0,
            point=pt,
        )

    # Query points for entropy estimation
    query_pts = [(0.2, 0.2), (0.5, 0.5), (0.9, 0.1)]

    # Compute a privacy‑adjusted prediction at a new location
    test_point = (0.3, 0.6)
    adj_pred = privacy_adjusted_prediction(hybrid, test_point, query_pts)

    print(f"Adjusted prediction at {test_point}: {adj_pred:.4f}")

    # Show entropy for diagnostic purposes
    ent = hybrid.surrogate_entropy(query_pts)
    print(f"Surrogate entropy over query points: {ent:.4f}")

    # Verify that the surrogate exists and can predict directly
    direct = hybrid.surrogate.predict(test_point) if hybrid.surrogate else float('nan')
    print(f"Direct surrogate prediction (no privacy factor): {direct:.4f}")