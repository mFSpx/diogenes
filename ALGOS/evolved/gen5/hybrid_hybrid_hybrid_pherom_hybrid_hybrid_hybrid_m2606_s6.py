# DARWIN HAMMER — match 2606, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_pheromone_inf_m894_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_path_s_path_signature_m501_s1.py (gen4)
# born: 2026-05-29T23:43:07Z

"""
Hybrid Pheromone–Path Signature Algorithm
========================================

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – a pheromone store where each entry decays exponentially
  according to a half‑life (`decay_factor = 0.5^{Δt/τ}`).

* **Parent B** – a path‑signature toolbox that provides a lead‑lag
  transform, B‑spline basis construction and low‑order signatures.

**Mathematical bridge**

Both parents operate on a temporal sequence.  In the parent‑B toolbox a
*path* is a sequence of vectors  :math:`x_0,\\dots,x_{T-1}`.
In the pheromone store each entry carries a *signal value* that evolves
in time by an exponential decay factor.  By treating the decayed pheromone
signal as an additional coordinate that is appended to every path point,
the lead‑lag transform and signature calculations naturally inherit the
entropy‑optimising decay dynamics of the pheromone model.  The hybrid
algorithm therefore:

1. Retrieves the current decay factor for a given surface (or any
   identifier) at each timestamp.
2. Augments the original geometric path with the decayed pheromone value,
   yielding a higher‑dimensional path.
3. Applies the lead‑lag transform, optional B‑spline smoothing and finally
   computes a low‑order signature on the augmented data.

The result is a *pheromone‑weighted signature* that reflects both the
geometric evolution of the system and the information‑theoretic decay of
the pheromone field.

The implementation below provides three public helper functions that
demonstrate the hybrid operation.
"""

import math
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Pheromone infrastructure (derived from Parent A)
# ----------------------------------------------------------------------


class PheromoneEntry:
    """A single pheromone record with exponential decay."""

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
        """Seconds elapsed since the last decay update."""
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative factor for the current decay step."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        """Apply exponential decay to the stored signal."""
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """In‑memory singleton‑like container for pheromone entries."""

    _entries: dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> list[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def decay_surface(cls, surface_key: str) -> None:
        """Apply decay to *all* entries belonging to the given surface."""
        for entry in cls.get_by_surface(surface_key):
            entry.apply_decay()

    @classmethod
    def current_signal(cls, surface_key: str) -> float:
        """Return the sum of decayed signal values for a surface."""
        cls.decay_surface(surface_key)
        return sum(e.signal_value for e in cls.get_by_surface(surface_key))


# ----------------------------------------------------------------------
# Path‑signature utilities (derived from Parent B)
# ----------------------------------------------------------------------


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag embedding that doubles the temporal resolution.
    Input: (T, d) array.
    Output: (2*T‑1, 2*d) array.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Simple B‑spline basis evaluator (k = 1,2,3 supported).
    Returns a (len(x), len(grid)+k-2) matrix B where B[i,j] = B_j^k(x_i).
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    N = len(x)
    M = len(t) - 1
    B = np.zeros((N, M), dtype=np.float64)

    for i in range(N):
        xi = x[i]
        for j in range(M):
            if t[j] <= xi <= t[j + 1]:
                if k == 1:
                    B[i, j] = 1.0
                elif k == 2:
                    B[i, j] = (xi - t[j]) / (t[j + 1] - t[j])
                    B[i, j + 1] = (t[j + 2] - xi) / (t[j + 2] - t[j + 1])
                elif k == 3:
                    # Cubic B‑spline using Cox‑de Boor recursion (depth 2)
                    # First level (k=2)
                    b1 = (xi - t[j]) / (t[j + 1] - t[j]) if t[j + 1] != t[j] else 0.0
                    b2 = (t[j + 2] - xi) / (t[j + 2] - t[j + 1]) if t[j + 2] != t[j + 1] else 0.0
                    # Second level (k=3)
                    B[i, j] = ((t[j + 2] - t[j]) * b1 ** 2) / ((t[j + 2] - t[j]) * (t[j + 1] - t[j]))
                    B[i, j + 1] = ((xi - t[j]) * (t[j + 1] - xi) ** 2) / ((t[j + 1] - t[j]) * (t[j + 2] - t[j + 1]))
                    B[i, j + 2] = ((xi - t[j]) ** 2 * (t[j + 2] - xi)) / ((t[j + 2] - t[j + 1]) * (t[j + 2] - t[j]))
                break
    return B


def signature_level(path: np.ndarray, level: int) -> np.ndarray:
    """
    Low‑order signature.
    level 1 → endpoint increment.
    level 2 → iterated integral (running.T @ increments).
    """
    path = np.asarray(path, dtype=float)
    if level == 1:
        return path[-1] - path[0]
    if level == 2:
        increments = np.diff(path, axis=0)          # (T‑1, d)
        running = path[:-1] - path[0]               # (T‑1, d)
        return running.T @ increments               # (d, d)
    raise ValueError("Invalid signature level; only 1 or 2 supported.")


# ----------------------------------------------------------------------
# Hybrid operations (the new fused logic)
# ----------------------------------------------------------------------


def augment_path_with_pheromone(
    geometric_path: np.ndarray,
    surface_key: str,
    timestamps: np.ndarray | None = None,
) -> np.ndarray:
    """
    Append a decayed pheromone signal as an extra coordinate to each
    point of ``geometric_path``.

    Parameters
    ----------
    geometric_path : (T, d) array
        Original spatial path.
    surface_key : str
        Identifier of the pheromone field that influences the path.
    timestamps : (T,) array, optional
        If supplied, decay is evaluated at the *absolute* time of each
        point (seconds since epoch).  If omitted, the current global decay
        is used for every point.

    Returns
    -------
    augmented_path : (T, d+1) array
        The original coordinates with an additional column containing the
        decayed pheromone value at each time step.
    """
    geometric_path = np.asarray(geometric_path, dtype=float)
    T = geometric_path.shape[0]

    # Compute a decay factor for each timestamp (or reuse the current sum)
    if timestamps is None:
        # Use the current summed signal for the whole surface
        pheromone_val = PheromoneStore.current_signal(surface_key)
        extra = np.full((T, 1), pheromone_val, dtype=float)
    else:
        timestamps = np.asarray(timestamps, dtype=float)
        extra = np.empty((T, 1), dtype=float)
        # Temporarily store original last_decay to avoid mutating the store
        # while iterating over timestamps.
        for idx, ts in enumerate(timestamps):
            # Simulate decay for a given age without altering the entry.
            age = max(0.0, ts - datetime.now(timezone.utc).timestamp())
            factor = 0.5 ** (age / max(1, _effective_half_life(surface_key)))
            # Sum of initial values multiplied by the simulated factor.
            total = sum(e.signal_value / e.decay_factor() * factor for e in PheromoneStore.get_by_surface(surface_key))
            extra[idx, 0] = total

    return np.hstack([geometric_path, extra])


def _effective_half_life(surface_key: str) -> float:
    """Utility: average half‑life of all entries for a surface (fallback 1)."""
    entries = PheromoneStore.get_by_surface(surface_key)
    if not entries:
        return 1.0
    return sum(e.half_life_seconds for e in entries) / len(entries)


def hybrid_lead_lag_signature(
    geometric_path: np.ndarray,
    surface_key: str,
    level: int = 2,
    timestamps: np.ndarray | None = None,
) -> np.ndarray:
    """
    Full hybrid pipeline:

    1. Augment the geometric path with a pheromone coordinate.
    2. Apply the lead‑lag transform (doubling resolution).
    3. Optionally smooth with a cubic B‑spline basis.
    4. Compute a low‑order signature on the resulting path.

    Returns the signature tensor (flattened) for convenience.
    """
    # 1. Augment
    aug_path = augment_path_with_pheromone(geometric_path, surface_key, timestamps)

    # 2. Lead‑lag
    ll_path = lead_lag_transform(aug_path)

    # 3. B‑spline smoothing (use a uniform grid over the time axis)
    T = ll_path.shape[0]
    grid = np.linspace(0, 1, num=T)
    B = bspline_basis(np.linspace(0, 1, num=T), grid, k=3)  # (T, T+1)
    # Project the path onto the spline basis (simple linear combination)
    smoothed = B @ ll_path  # (T, d)

    # 4. Signature
    sig = signature_level(smoothed, level)

    # Flatten for a 1‑D return (use ravel)
    return sig.ravel()


def update_pheromone(
    surface_key: str,
    signal_kind: str,
    added_value: float,
    half_life_seconds: int = 60,
) -> None:
    """
    Create a new pheromone entry or, if an entry of the same kind already
    exists on the surface, increase its signal value.
    Afterwards, decay all entries on the surface to keep the store up‑to‑date.
    """
    # Decay existing entries first (keeps the system consistent)
    PheromoneStore.decay_surface(surface_key)

    # Look for an existing entry of the same kind
    candidates = [
        e for e in PheromoneStore.get_by_surface(surface_key) if e.signal_kind == signal_kind
    ]
    if candidates:
        # Merge into the first candidate
        entry = candidates[0]
        entry.signal_value += added_value
    else:
        entry = PheromoneEntry(
            surface_key=surface_key,
            signal_kind=signal_kind,
            signal_value=added_value,
            half_life_seconds=half_life_seconds,
        )
        PheromoneStore.add(entry)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # 1. Initialise pheromone store with a couple of entries
    update_pheromone("terrain_A", "food", added_value=10.0, half_life_seconds=30)
    update_pheromone("terrain_A", "danger", added_value=5.0, half_life_seconds=45)

    # 2. Construct a simple 2‑D geometric path (e.g., a random walk)
    rng = np.random.default_rng(seed=42)
    steps = rng.normal(loc=0.0, scale=1.0, size=(20, 2))
    path = np.cumsum(steps, axis=0)  # (20, 2)

    # 3. Compute the hybrid signature
    sig = hybrid_lead_lag_signature(path, surface_key="terrain_A", level=2)

    # 4. Print a short summary
    print("Hybrid signature (flattened, level=2) shape:", sig.shape)
    print("First 8 components:", sig[:8])

    # Ensure that the program exits cleanly
    sys.exit(0)