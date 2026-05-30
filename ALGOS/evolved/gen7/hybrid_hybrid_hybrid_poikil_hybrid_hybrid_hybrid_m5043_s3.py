# DARWIN HAMMER — match 5043, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_poikilotherm__hybrid_regret_engine_m1595_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m2134_s1.py (gen6)
# born: 2026-05-29T23:59:28Z

"""Hybrid Pheromone‑Regret‑Hoeffding‑Gini‑Doomsday Hyperdimensional Engine

Parents:
    * hybrid_hybrid_poikilotherm__hybrid_regret_engine_m1595_s2.py
    * hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m2134_s1.py

Mathematical bridge:
    Both parents compute a Gini coefficient on a collection of values.
    The first parent applies that Gini to modulate pheromone‑based
    regret‑weighted probabilities, while the second mixes the same Gini
    with a Doomsday weekday and a Hoeffding concentration bound before
    encoding the result into a bipolar hypervector.  The fused algorithm
    therefore uses a single Gini value as the shared statistic, combines
    it with a Doomsday index and a Hoeffding bound to obtain a scalar
    “hybrid score”, and finally maps that score into a hyperdimensional
    representation that can be bundled with other scores.
"""

import math
import random
import sys
from pathlib import Path
from datetime import date
from typing import List, Dict, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Core statistical primitives (shared across parents)
# ----------------------------------------------------------------------
def gini_coefficient(values: List[float]) -> float:
    """Standard Gini coefficient for a non‑negative distribution."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cum = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cum / (n * sum(xs))


def doomsday(year: int, month: int, day: int) -> int:
    """Weekday index (0=Monday … 6=Sunday) using the Doomsday algorithm."""
    return (date(year, month, day).weekday() + 1) % 7


def hoeffding_bound(mean: float, value_range: float, n: int, delta: float = 0.05) -> float:
    """
    Hoeffding bound ε such that P(|X̄−μ|>ε) ≤ δ.
    Returns the half‑width ε.
    """
    if n <= 0:
        raise ValueError("sample size must be positive")
    return math.sqrt((value_range ** 2 * math.log(2 / delta)) / (2 * n))


# ----------------------------------------------------------------------
# Hyperdimensional primitives (from parent B)
# ----------------------------------------------------------------------
Vector = List[int]  # bipolar (+1 / -1)


def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    """Deterministic bipolar vector derived from a string."""
    seed = int.from_bytes(
        hash(symbol.encode("utf-8"))[:8] if isinstance(hash(symbol), bytes) else
        hash(symbol).to_bytes(8, "big", signed=False), "big", signed=False
    )
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Hadamard product (binding) of two bipolar vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: Iterable[Vector]) -> Vector:
    """Element‑wise majority vote (bipolar bundling)."""
    vecs = list(vectors)
    if not vecs:
        raise ValueError("no vectors to bundle")
    return [1 if sum(comp) > 0 else -1 for comp in zip(*vecs)]


# ----------------------------------------------------------------------
# Pheromone system (from parent A) with regret weighting
# ----------------------------------------------------------------------
class HybridPheromoneSystem:
    """Manages pheromone signals and applies decay."""
    def __init__(self):
        self.pheromones: Dict[str, Dict] = {}
        self.current_time: float = 0.0

    def _decay(self, value: float, half_life: float, elapsed: float) -> float:
        """Exponential decay based on half‑life."""
        if half_life <= 0:
            return value
        decay_factor = 0.5 ** (elapsed / half_life)
        return value * decay_factor

    def update_signal(
        self,
        surface_key: str,
        signal_value: float,
        half_life_seconds: float,
        elapsed_seconds: float = 0.0,
    ) -> None:
        """
        Insert or decay‑update a pheromone signal.
        """
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {
                "signal_value": signal_value,
                "half_life_seconds": half_life_seconds,
                "last_update": self.current_time,
            }
        else:
            rec = self.pheromones[surface_key]
            elapsed = (self.current_time - rec["last_update"]) + elapsed_seconds
            rec["signal_value"] = self._decay(
                rec["signal_value"], rec["half_life_seconds"], elapsed
            ) + signal_value
            rec["last_update"] = self.current_time

    def get_all_signals(self) -> List[float]:
        """Current (decayed) signal values for every surface."""
        values = []
        for rec in self.pheromones.values():
            elapsed = self.current_time - rec["last_update"]
            values.append(self._decay(rec["signal_value"], rec["half_life_seconds"], elapsed))
        return values


# ----------------------------------------------------------------------
# Hybrid metric combining Gini, Doomsday, Hoeffding & regret
# ----------------------------------------------------------------------
def compute_hybrid_score(
    pheromone_system: HybridPheromoneSystem,
    date_tuple: Tuple[int, int, int],
    regret_samples: List[float],
    delta: float = 0.05,
) -> float:
    """
    1. Gini of current pheromone signal distribution.
    2. Doomsday weekday index (0‑6) → normalized w = 1 - weekday/6.
    3. Hoeffding bound on regret samples (range assumed 0‑1).
    4. Combine: score = Gini * w * (1 - ε) where ε is the Hoeffding bound.
    """
    # 1. Gini
    signals = pheromone_system.get_all_signals()
    g = gini_coefficient(signals)

    # 2. Doomsday weight
    y, m, d = date_tuple
    weekday = doomsday(y, m, d)  # 0‑6
    weekday_weight = 1.0 - (weekday / 6.0)  # higher on early weekdays

    # 3. Hoeffding bound on regret (assume values in [0,1])
    n = len(regret_samples)
    if n == 0:
        hoeff = 0.0
    else:
        mean_regret = sum(regret_samples) / n
        hoeff = hoeffding_bound(mean_regret, 1.0, n, delta)

    # 4. Fusion
    score = g * weekday_weight * (1.0 - hoeff)
    # Clip to [0,1] for downstream encoding
    return max(0.0, min(1.0, score))


# ----------------------------------------------------------------------
# Hyperdimensional encoding of the hybrid score
# ----------------------------------------------------------------------
def encode_score_hypervector(
    score: float,
    dim: int = 10000,
    morphological_seed: int = 42,
) -> Vector:
    """
    Map a scalar score ∈ [0,1] to a bipolar hypervector.
    Steps:
        * Symbol vector from a deterministic string representation of the score.
        * Morphological vector (random but fixed) acts as a “shape” token.
        * Bind both vectors → final representation.
    """
    # Symbol based on discretised score (10 bins)
    bin_idx = int(score * 10)
    symbol = f"score_bin_{bin_idx}"
    sv = symbol_vector(symbol, dim)

    # Fixed morphological token
    mv = random_vector(dim, morphological_seed)

    return bind(sv, mv)


def bundle_scores_hypervectors(scores: List[float], dim: int = 10000) -> Vector:
    """Encode each score and return a bundled hypervector."""
    hvectors = (encode_score_hypervector(s, dim) for s in scores)
    return bundle(hvectors)


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise pheromone system and populate with synthetic data
    ph_sys = HybridPheromoneSystem()
    for i in range(5):
        ph_sys.update_signal(
            surface_key=f"node_{i}",
            signal_value=random.uniform(0.1, 1.0),
            half_life_seconds=300.0,
        )
    # Advance internal clock to simulate decay
    ph_sys.current_time += 120.0

    # Synthetic regret samples (e.g., losses from recent decisions)
    regrets = [random.random() for _ in range(20)]

    # Compute hybrid score for a given date
    today = date.today()
    score = compute_hybrid_score(
        pheromone_system=ph_sys,
        date_tuple=(today.year, today.month, today.day),
        regret_samples=regrets,
    )
    print(f"Hybrid score: {score:.4f}")

    # Encode the score into a hypervector and display its first 20 components
    hv = encode_score_hypervector(score, dim=2000)
    print("Hypervector (first 20 bits):", hv[:20])

    # Bundle a few scores together
    more_scores = [score, score * 0.8, min(1.0, score + 0.1)]
    bundled = bundle_scores_hypervectors(more_scores, dim=2000)
    print("Bundled hypervector (first 20 bits):", bundled[:20])