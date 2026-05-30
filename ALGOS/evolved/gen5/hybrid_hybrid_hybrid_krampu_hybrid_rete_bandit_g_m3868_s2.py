# DARWIN HAMMER — match 3868, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_krampus_stick_hybrid_rbf_surrogate_m556_s0.py (gen4)
# parent_b: hybrid_rete_bandit_gate_hybrid_workshare_all_m43_s0.py (gen2)
# born: 2026-05-29T23:52:17Z

import math
import random
import sys
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
MAX_COMPONENT_TOKENS = 500
DEFAULT_EPSILON = 1.0          # kernel width
DEFAULT_ALPHA = 0.5            # blend factor
DEFAULT_TEMPERATURE = 1.0      # for softmax‑like scaling of regret

# ----------------------------------------------------------------------
# Pheromone infrastructure (from Parent A)
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(random.getrandbits(128))
        self.surface_key = surface_key          # e.g. work‑group name
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        self.created_at = datetime.utcnow()
        self.last_decay = self.created_at

    def age_seconds(self) -> float:
        return (datetime.utcnow() - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.utcnow()


class PheromoneStore:
    """Simple in‑memory singleton store."""
    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]

    @classmethod
    def all_entries(cls) -> List[PheromoneEntry]:
        return list(cls._entries.values())

    @classmethod
    def decay_all(cls) -> None:
        """Decay each entry exactly once per call."""
        for e in cls._entries.values():
            e.apply_decay()


# ----------------------------------------------------------------------
# Regret infrastructure (new – deeper integration)
# ----------------------------------------------------------------------
class RegretEntry:
    __slots__ = ("group", "cumulative_loss", "observations", "last_update")

    def __init__(self, group: str):
        self.group = group
        self.cumulative_loss = 0.0
        self.observations = 0
        self.last_update = datetime.utcnow()

    def update(self, loss: float) -> None:
        """Incorporate a new loss observation."""
        self.cumulative_loss += loss
        self.observations += 1
        self.last_update = datetime.utcnow()

    def avg_loss(self) -> float:
        return self.cumulative_loss / self.observations if self.observations else 0.0


class RegretStore:
    """Singleton store for per‑group regret statistics."""
    _entries: Dict[str, RegretEntry] = {}

    @classmethod
    def get(cls, group: str) -> RegretEntry:
        if group not in cls._entries:
            cls._entries[group] = RegretEntry(group)
        return cls._entries[group]

    @classmethod
    def update(cls, group: str, loss: float) -> None:
        cls.get(group).update(loss)

    @classmethod
    def avg_losses(cls, groups: List[str]) -> np.ndarray:
        return np.array([cls.get(g).avg_loss() for g in groups], dtype=float)


# ----------------------------------------------------------------------
# Gaussian radial‑basis function (core of Parent A)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = DEFAULT_EPSILON) -> float:
    """Gaussian RBF: exp( -r² / (2·ε²) )."""
    return math.exp(-(r ** 2) / (2.0 * epsilon ** 2))


# ----------------------------------------------------------------------
# Work‑group definitions (from Parent B)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")


# ----------------------------------------------------------------------
# Helper: aggregate pheromone signal per group
# ----------------------------------------------------------------------
def aggregated_signal(group: str) -> float:
    """Sum of decayed signal values for a given group."""
    entries = PheromoneStore.get_by_surface(group)
    return sum(e.signal_value for e in entries)


# ----------------------------------------------------------------------
# Build Gaussian similarity matrix from pheromone levels
# ----------------------------------------------------------------------
def pheromone_similarity_matrix(groups: List[str],
                                epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """
    Returns a row‑stochastic similarity matrix S where
        S_ij = gaussian(|agg_i - agg_j|, epsilon)
    and each row sums to 1.
    """
    agg = np.array([aggregated_signal(g) for g in groups], dtype=float)
    diff = np.abs(agg[:, None] - agg[None, :])
    raw = np.vectorize(lambda r: gaussian(r, epsilon))(diff)

    # Row‑normalize; guard against zero rows
    row_sums = raw.sum(axis=1, keepdims=True)
    # If a row sum is zero (unlikely), replace with uniform distribution
    zero_mask = (row_sums == 0)
    raw[zero_mask] = 1.0 / len(groups)
    row_sums[zero_mask] = 1.0
    return raw / row_sums


# ----------------------------------------------------------------------
# Regret‑derived base weight vector
# ----------------------------------------------------------------------
def regret_base_vector(groups: List[str],
                       temperature: float = DEFAULT_TEMPERATURE) -> np.ndarray:
    """
    Convert average losses into a probability‑like vector.
    Lower loss → higher weight.  Uses a softmax over the negative losses.
    """
    losses = RegretStore.avg_losses(groups)          # shape (n,)
    # Prevent overflow/underflow
    scaled = -losses / max(temperature, 1e-8)
    # Shift for numerical stability
    shifted = scaled - np.max(scaled)
    exp_vals = np.exp(shifted)
    if exp_vals.sum() == 0:
        # Fallback to uniform if all losses are NaN/inf
        return np.full(len(groups), 1.0 / len(groups), dtype=float)
    return exp_vals / exp_vals.sum()


# ----------------------------------------------------------------------
# Hybrid weight computation (deep integration)
# ----------------------------------------------------------------------
def compute_hybrid_weights(groups: List[str],
                           epsilon: float = DEFAULT_EPSILON,
                           alpha: float = DEFAULT_ALPHA,
                           temperature: float = DEFAULT_TEMPERATURE) -> np.ndarray:
    """
    Hybrid weight computation:
      1. Derive a regret‑aware base vector v via softmax over average losses.
      2. Diffuse v through the pheromone similarity matrix S.
      3. Blend: w = (1‑α)·v + α·(S·v)
      4. Normalise to sum to 1.
    Parameters
    ----------
    groups : list of group names
    epsilon : kernel width for the Gaussian RBF
    alpha : mixing coefficient (0 → pure regret, 1 → pure pheromone diffusion)
    temperature : softmax temperature controlling regret sharpness
    """
    n = len(groups)
    if n == 0:
        return np.array([])

    # 1️⃣ Regret‑aware base vector
    v = regret_base_vector(groups, temperature=temperature)

    # 2️⃣ Pheromone diffusion matrix (row‑stochastic)
    S = pheromone_similarity_matrix(groups, epsilon=epsilon)

    # 3️⃣ Blend diffusion with base
    w = (1.0 - alpha) * v + alpha * (S @ v)

    # 4️⃣ Normalise (numerical safety)
    w_sum = w.sum()
    if w_sum > 0:
        w /= w_sum
    else:
        w = np.full(n, 1.0 / n, dtype=float)

    return w


# ----------------------------------------------------------------------
# Allocation routine (core of Parent B) now using hybrid weights
# ----------------------------------------------------------------------
def compute_allocation(*,
                       total_units: float,
                       year: int,
                       month: int,
                       day: int,
                       epsilon: float = DEFAULT_EPSILON,
                       alpha: float = DEFAULT_ALPHA,
                       temperature: float = DEFAULT_TEMPERATURE,
                       deterministic_target_pct: float = 90.0) -> dict[str, Any]:
    """
    Compute work‑unit allocation for the given date.
    The allocation weights are obtained from the hybrid
    regret‑plus‑pheromone mechanism.
    """
    # 1️⃣ Apply decay once per allocation call
    PheromoneStore.decay_all()

    # 2️⃣ Day of week (doomsday calendar) – kept for legacy compatibility
    day_of_week = (date(year, month, day).weekday() + 1) % 7

    # 3️⃣ Hybrid weight vector
    groups = list(GROUPS)
    weights = compute_hybrid_weights(groups,
                                     epsilon=epsilon,
                                     alpha=alpha,
                                     temperature=temperature)

    # 4️⃣ Allocate units proportionally
    raw_units = total_units * weights
    total_allocated = raw_units.sum()
    if total_allocated == 0:
        allocated = np.zeros_like(raw_units)
    else:
        allocated = raw_units / total_allocated * total_units

    # 5️⃣ Round for readability
    allocated = np.round(allocated.astype(float), 6)

    allocation = {
        "total_units": round(total_units, 6),
        "deterministic_target_pct": round(deterministic_target_pct, 6),
        "day_of_week": day_of_week,
        "lanes": [
            {"group": grp, "units": float(allocated[i])}
            for i, grp in enumerate(groups)
        ],
    }
    return allocation


def summarize_allocation(allocation: dict[str, Any]) -> None:
    """Pretty‑print the allocation dictionary."""
    print(f"Allocation Summary (Day {allocation['day_of_week']}):")
    print(f"  Total Units: {allocation['total_units']}")
    print(f"  Target %   : {allocation['deterministic_target_pct']}")
    for lane in allocation["lanes"]:
        print(f"  - {lane['group']}: {lane['units']} units")


# ----------------------------------------------------------------------
# Example usage / smoke test (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed some pheromone entries
    for grp in GROUPS:
        for _ in range(3):
            PheromoneStore.add(
                PheromoneEntry(
                    surface_key=grp,
                    signal_kind="reward",
                    signal_value=random.uniform(0.5, 5.0),
                    half_life_seconds=random.randint(300, 1800),
                )
            )
    # Seed some regret observations
    for grp in GROUPS:
        for _ in range(5):
            loss = random.uniform(0.0, 1.0)  # lower is better
            RegretStore.update(grp, loss)

    alloc = compute_allocation(
        total_units=1000.0,
        year=2026,
        month=5,
        day=29,
        epsilon=0.8,
        alpha=0.6,
        temperature=0.2,
    )
    summarize_allocation(alloc)