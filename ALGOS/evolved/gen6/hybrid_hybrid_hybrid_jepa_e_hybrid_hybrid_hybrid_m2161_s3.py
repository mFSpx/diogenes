# DARWIN HAMMER — match 2161, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_jepa_energy_h_hybrid_shannon_entro_m777_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m456_s0.py (gen5)
# born: 2026-05-29T23:41:12Z

import math
import random
from collections import Counter
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTier:
    """Immutable description of a model that can be loaded into the pool."""
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


@dataclass
class Morphology:
    """Physical constraints used by the hybrid algorithms."""
    length: int
    width: int
    height: int
    mass: int

    @property
    def volume(self) -> int:
        return self.length * self.width * self.height


# ----------------------------------------------------------------------
# Model pool with principled eviction
# ----------------------------------------------------------------------


class ModelPool:
    """
    Manages a set of loaded models under a RAM ceiling.
    Energy is a scalar that reflects the “health” of the pool:
    * positive energy → penalties (e.g. memory overflow, tier conflicts)
    * negative energy → rewards (e.g. successful loads, useful evictions)
    """

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb: int = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._energy: float = 0.0
        # keep an ordered list to implement LRU eviction
        self._access_order: List[str] = []

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _used(self) -> int:
        """Current RAM consumption in MB."""
        return sum(m.ram_mb for m in self.loaded.values())

    def _record_access(self, name: str) -> None:
        """Update LRU order."""
        if name in self._access_order:
            self._access_order.remove(name)
        self._access_order.append(name)

    def _evict_lru(self) -> None:
        """Evict the least‑recently used model."""
        if not self._access_order:
            return
        lru_name = self._access_order.pop(0)
        evicted = self.loaded.pop(lru_name, None)
        if evicted:
            # modest reward for freeing RAM
            self._energy -= 5e2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def free_energy(self) -> float:
        """Current energy value (higher is “worse”)."""
        return self._energy

    def add_model(self, model: ModelTier) -> None:
        """
        Insert a model without any eviction. Penalties are applied for:
        * Tier‑conflict: a T3 model cannot coexist with any T2 model.
        * RAM overflow: exceeding the ceiling.
        """
        # Tier conflict penalty
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            self._energy += 1e10

        # RAM overflow penalty
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            self._energy += 1e6

        self.loaded[model.name] = model
        self._record_access(model.name)

    def load(self, model: ModelTier) -> None:
        """
        Load a model, rewarding the action.
        If RAM would overflow, the method automatically evicts LRU models
        until the new model fits.
        """
        self._energy -= 1e4  # reward for a successful load
        self._ensure_fit(model.ram_mb)
        self.add_model(model)

    def load_with_eviction(self, model: ModelTier) -> None:
        """
        Load a model while explicitly evicting the least‑recently used
        models until enough RAM is freed.
        """
        self._energy -= 1e3  # smaller reward because eviction is required
        self._ensure_fit(model.ram_mb)
        self.add_model(model)

    # ------------------------------------------------------------------
    # Internal eviction logic
    # ------------------------------------------------------------------

    def _ensure_fit(self, required_ram: int) -> None:
        """Evict LRU models until `required_ram` can be accommodated."""
        while self.loaded and required_ram + self._used() > self.ram_ceiling_mb:
            self._evict_lru()


# ----------------------------------------------------------------------
# Information‑theoretic utilities
# ----------------------------------------------------------------------


def shannon_entropy(
    observations: Iterable[Union[Hashable, float]],
    *,
    is_distribution: bool = False,
) -> float:
    """
    Compute Shannon entropy (base‑2) either from raw observations
    (counts are inferred) or from an explicit probability distribution.

    Parameters
    ----------
    observations:
        Iterable of hashable items or probabilities.
    is_distribution:
        If True, `observations` is interpreted as a probability mass
        function that already sums to 1. Otherwise, frequencies are
        derived from the raw data.

    Returns
    -------
    float
        Entropy in bits.
    """
    xs = list(observations)
    if not xs:
        return 0.0

    if is_distribution:
        probs = [float(p) for p in xs]
        if any(p < 0 for p in probs):
            raise ValueError("probabilities must be non‑negative")
        total = sum(probs)
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            # Normalise silently – a common source of bugs in the original code
            probs = [p / total for p in probs]
    else:
        # Build a frequency table and normalise to a distribution
        counter = Counter(xs)
        total = sum(counter.values())
        probs = [cnt / total for cnt in counter.values()]

    # Entropy formula, ignoring zero‑probability events
    return -sum(p * math.log2(p) for p in probs if p > 0.0)


def fractional_power_binding(value: float, exponent: float) -> float:
    """
    Apply a fractional power (exponent between 0 and 1) to a positive value.
    This mimics the “fractional power binding” described in the original
    hybrid narrative, providing a smoother scaling than a raw linear term.
    """
    if value < 0:
        raise ValueError("value must be non‑negative for fractional power binding")
    return value ** exponent


# ----------------------------------------------------------------------
# Geometry / vector utilities
# ----------------------------------------------------------------------


def euclidean(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_dhash(values: List[float]) -> int:
    """Difference hash – 1 bit per adjacent pair."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits


def compute_phash(values: List[float]) -> int:
    """Perceptual hash – thresholded against the mean of the first 64 values."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def cluster_by_phash(hashes: Dict[str, int], max_distance: int = 4) -> List[List[str]]:
    """
    Simple agglomerative clustering based on Hamming distance.
    """
    clusters: List[List[str]] = []
    for key, h in hashes.items():
        placed = False
        for cluster in clusters:
            if hamming_distance(h, hashes[cluster[0]]) <= max_distance:
                cluster.append(key)
                placed = True
                break
        if not placed:
            clusters.append([key])
    return clusters


# ----------------------------------------------------------------------
# Hybrid algorithmic components
# ----------------------------------------------------------------------


def hybrid_model_pool(
    pool: ModelPool,
    morphology: Morphology,
    decision_hygiene_features: List[float],
    *,
    entropy_exponent: float = 0.75,
) -> ModelPool:
    """
    Adjust the pool's energy based on the (fractionally bound) entropy of the
    decision‑hygiene feature vector and ensure RAM constraints respect the
    physical limits expressed by `morphology`.

    The fractional exponent deepens the integration by tempering the raw
    entropy, which otherwise would dominate the energy term.
    """
    raw_entropy = shannon_entropy(decision_hygiene_features)
    bounded_entropy = fractional_power_binding(raw_entropy, entropy_exponent)
    pool._energy += bounded_entropy  # higher entropy → higher penalty

    # Enforce a hard RAM ceiling derived from the morphology's volume.
    # If the current usage exceeds this bound, evict LRU models until it fits.
    max_ram_from_morph = morphology.volume * 10  # 10 MB per voxel as a heuristic
    if pool._used() > max_ram_from_morph:
        # Evict until we are under the morphological limit
        while pool._used() > max_ram_from_morph and pool.loaded:
            pool._evict_lru()
    return pool


def hybrid_decision_hygiene(pool: ModelPool, decision_hygiene_features: List[float]) -> float:
    """
    Produce a scalar “hygiene score”.  The score is the entropy normalised by
    the absolute value of the pool's current energy, yielding a dimensionless
    quantity that remains bounded even when the pool accumulates large penalties.
    """
    entropy = shannon_entropy(decision_hygiene_features)
    energy = abs(pool.free_energy()) + 1e-9  # avoid division by zero
    return entropy / energy


def hybrid_create_bipolar_vectors(
    morphology: Morphology,
    sphericity_index: float,
    *,
    seed: int | None = None,
) -> List[List[float]]:
    """
    Generate a set of bipolar (Gaussian‑distributed) vectors whose count is
    proportional to the product of the morphology volume and a sphericity index.
    A deterministic seed can be supplied for reproducibility – useful for
    unit testing and for tighter coupling with JEPA‑style predictive models.
    """
    if seed is not None:
        random.seed(seed)

    count = max(1, int(morphology.volume * sphericity_index))
    vectors: List[List[float]] = []
    for _ in range(count):
        vector = [random.gauss(0.0, 1.0) for _ in range(morphology.height)]
        # Convert to bipolar representation (+1 / -1) while preserving magnitude
        bipolar = [1.0 if v >= 0 else -1.0 for v in vector]
        vectors.append(bipolar)
    return vectors


# ----------------------------------------------------------------------
# Example driver (kept minimal for import safety)
# ----------------------------------------------------------------------


def _demo() -> None:
    pool = ModelPool(ram_ceiling_mb=8000)
    morph = Morphology(length=12, width=8, height=6, mass=4)

    # Populate pool with a few dummy models
    for i in range(5):
        tier = "T2" if i % 2 == 0 else "T1"
        pool.load(ModelTier(name=f"model_{i}", ram_mb=1500, tier=tier))

    features = [0.45, 0.33, 0.22, 0.55]
    hybrid_model_pool(pool, morph, features)
    print("Hygiene score:", hybrid_decision_hygiene(pool, features))

    bipolar = hybrid_create_bipolar_vectors(morph, sphericity_index=0.68, seed=42)
    print("First bipolar vector (sample):", bipolar[0] if bipolar else None)


if __name__ == "__main__":
    _demo()