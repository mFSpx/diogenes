# DARWIN HAMMER — match 2597, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_pherom_m227_s2.py (gen3)
# born: 2026-05-29T23:43:02Z

"""Hybrid Allocation‑Pheromone‑Morphology Module
================================================

This module fuses the core mathematics of two parent algorithms:

* **Parent A – hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py**
  – Provides a deterministic group allocation modulated by a *Caputo fractional‑memory
    kernel* and an *input‑dependent effective time constant*.  The kernel uses a
    Lanczos approximation of the Gamma function.

* **Parent B – hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_pherom_m227_s2.py**
  – Supplies a *morphology‑based recovery priority* (via sphericity/flatness indices)
    and a *pheromone system* that decays signals exponentially with a configurable
    half‑life.

**Mathematical Bridge**

The hybrid algorithm treats the *allocation share* `a_i(t)` for each group `i` as a
product of three factors:


a_i(t) =  w_i(t) · p_i · φ_i(t)


* `w_i(t)` – fractional‑memory weight obtained from the Caputo kernel applied to the
  historical allocation series of group `i`.  The kernel uses the Lanczos Gamma
  approximation to evaluate the Gamma function required by the Caputo derivative.

* `p_i` – recovery priority derived from the morphology of the resource consumer,
  computed from the right‑ing‑time index (Parent B).

* `φ_i(t)` – pheromone signal for the group, updated with exponential decay based on
  a half‑life (Parent B).

Thus the allocation process inherits the *memory‑aware* dynamics of Parent A while
being *shaped* by the *biophysical priority* and *adaptive pheromone feedback* of
Parent B.  The three functions below demonstrate this unified system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Lanczos Gamma approximation (Parent A core)
# ---------------------------------------------------------------------------

_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def _lanczos_gamma(z: float) -> float:
    """Lanczos approximation of the Gamma function (real argument)."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * _lanczos_gamma(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, len(_LANCZOS_C)):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

# ---------------------------------------------------------------------------
# Morphology and recovery priority (Parent B core)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1] derived from righting‑time index."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

# ---------------------------------------------------------------------------
# Pheromone system (Parent B core)
# ---------------------------------------------------------------------------

class PheromoneSystem:
    """Simple pheromone tracker with exponential decay."""
    def __init__(self) -> None:
        self.pheromones: Dict[str, Dict[str, Any]] = {}

    def update(self, surface_key: str, signal_value: float,
               half_life_seconds: float) -> None:
        """Insert or decay‑update a pheromone signal."""
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {
                "value": signal_value,
                "half_life": half_life_seconds,
            }
            return
        entry = self.pheromones[surface_key]
        # decay factor for one second interval (approximation)
        decay = math.exp(-1.0 / entry["half_life"])
        entry["value"] = entry["value"] * decay + signal_value * (1 - decay)

    def get(self, surface_key: str) -> float:
        return self.pheromones.get(surface_key, {"value": 0.0})["value"]

# ---------------------------------------------------------------------------
# Fractional‑memory kernel (Caputo derivative) – Parent A core
# ---------------------------------------------------------------------------

def fractional_memory_weights(history: np.ndarray, alpha: float,
                              dt: float = 1.0) -> np.ndarray:
    """
    Compute Caputo‑type fractional memory weights for a 1‑D history array.

    The weight w_k for a past value at lag k (k≥0) is:
        w_k = ( (t - τ_k)^{α-1} ) / Γ(α)
    where τ_k = k·dt and Γ is evaluated via the Lanczos approximation.
    The returned array has the same length as ``history`` and sums to 1.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0,1]")
    n = len(history)
    lags = np.arange(n)[::-1] * dt            # newest lag = 0
    raw = (lags ** (alpha - 1.0)) / _lanczos_gamma(alpha)
    weights = raw / raw.sum()
    return weights

# ---------------------------------------------------------------------------
# Hybrid allocation engine
# ---------------------------------------------------------------------------

def hybrid_allocate(
    total_resource: float,
    groups: List[str],
    history_dict: Dict[str, np.ndarray],
    morphologies: Dict[str, Morphology],
    pheromone_sys: PheromoneSystem,
    alpha: float = 0.7,
) -> Dict[str, float]:
    """
    Allocate ``total_resource`` among ``groups`` using a product of:

    * Fractional‑memory weight derived from each group's allocation history.
    * Recovery priority from the group's morphology.
    * Current pheromone signal for the group.

    Returns a dict mapping group name → allocated amount.
    """
    # 1. Fractional memory component
    mem_weights = {}
    for g in groups:
        hist = history_dict.get(g, np.array([0.0]))
        w = fractional_memory_weights(hist, alpha)
        mem_weights[g] = float((w * hist).sum())  # memory‑averaged past allocation

    # 2. Priority component (morphology)
    priorities = {g: recovery_priority(morphologies[g]) for g in groups}

    # 3. Pheromone component
    pheromones = {g: pheromone_sys.get(g) for g in groups}

    # Combine multiplicatively; add a small epsilon to avoid zero division.
    eps = 1e-12
    raw_scores = {}
    for g in groups:
        raw = (mem_weights[g] + eps) * (priorities[g] + eps) * (pheromones[g] + eps)
        raw_scores[g] = raw

    total_score = sum(raw_scores.values()) + eps
    allocation = {g: total_resource * (raw_scores[g] / total_score) for g in groups}
    return allocation

def update_system(
    history_dict: Dict[str, np.ndarray],
    allocation: Dict[str, float],
    morphologies: Dict[str, Morphology],
    pheromone_sys: PheromoneSystem,
    half_life_seconds: float = 86400.0,
) -> None:
    """
    Update all state containers after an allocation step:

    * Append the new allocation to each group's history (as a rolling window of length 10).
    * Optionally adjust the pheromone signal using the allocated amount as the signal value.
    """
    for g, amt in allocation.items():
        hist = history_dict.setdefault(g, np.zeros(0))
        hist = np.append(hist, amt)[-10:]          # keep last 10 entries
        history_dict[g] = hist
        # Use allocated amount (scaled) as pheromone reinforcement
        pheromone_sys.update(g, signal_value=amt, half_life_seconds=half_life_seconds)

def summarize_allocation(
    allocation: Dict[str, float],
    history_dict: Dict[str, np.ndarray],
) -> Tuple[float, float]:
    """
    Return a tuple ``(total_allocated, allocation_variance)`` where variance is
    computed over the current allocation vector.
    """
    values = np.fromiter(allocation.values(), dtype=float)
    total = float(values.sum())
    variance = float(values.var())
    return total, variance

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Define groups
    GROUPS = ["codex", "groq", "cohere", "local_models"]

    # Initialise empty histories
    histories: Dict[str, np.ndarray] = {g: np.zeros(0) for g in GROUPS}

    # Random morphologies for each group
    random.seed(42)
    morphs: Dict[str, Morphology] = {}
    for g in GROUPS:
        morphs[g] = Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass=random.uniform(1.0, 5.0),
        )

    # Pheromone system
    pher = PheromoneSystem()
    for g in GROUPS:
        pher.update(g, signal_value=random.uniform(0.0, 1.0), half_life_seconds=86400.0)

    # Run a few allocation cycles
    TOTAL = 1000.0
    for step in range(5):
        alloc = hybrid_allocate(
            total_resource=TOTAL,
            groups=GROUPS,
            history_dict=histories,
            morphologies=morphs,
            pheromone_sys=pher,
            alpha=0.8,
        )
        total, var = summarize_allocation(alloc, histories)
        print(f"Step {step+1}: total={total:.2f}, variance={var:.4f}")
        for g, a in alloc.items():
            print(f"  {g}: {a:.2f}")
        update_system(histories, alloc, morphs, pher, half_life_seconds=86400.0)

    print("Smoke test completed without errors.")