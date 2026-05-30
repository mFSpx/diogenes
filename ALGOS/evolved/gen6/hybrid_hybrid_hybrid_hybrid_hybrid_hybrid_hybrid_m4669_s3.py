# DARWIN HAMMER — match 4669, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2484_s3.py (gen5)
# born: 2026-05-29T23:57:19Z

"""Hybrid Allocation‑Regret Engine

This module fuses the *weekday sinusoidal weight vector* from the allocator
(parent A) with the *Fisher‑information‑driven regret weighting* from the
diffusion‑regret engine (parent B).  The mathematical bridge is the element‑wise
product of the stochastic weekday vector **wₖ** and the Fisher score
**Fₖ(θ)** for each logical group/action *k*.  The resulting hybrid weight

    Hₖ = wₖ · Fₖ(θ)

is used both to rank actions and to guide VRAM‑aware GPU selection, thus
producing a single unified resource‑allocation system.

The implementation provides three core functions:
* ``weekday_fisher_weight_vector`` – builds the hybrid weight vector.
* ``vram_fisher_regret_selection`` – picks GPUs that satisfy VRAM constraints
  and pairs them with actions using the hybrid weights.
* ``hybrid_allocate`` – distributes a total resource budget across actions
  proportionally to their hybrid weights.

All code is pure Python 3 with only the standard library and NumPy.
"""

import datetime as dt
import hashlib
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants (mirroring the parents)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1
DEFAULT_BUDGET_MB = 4096
DEFAULT_RESERVE_MB = 768

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Normalised sinusoidal weight vector for *groups* based on weekday ``dow``.
    The vector is row‑stochastic (sums to 1).
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def vram_aware_gpu_selection(
    gpus: List[Dict[str, Any]], budget_mb: int, reserve_mb: int
) -> List[Dict[str, Any]]:
    """
    Return the subset of ``gpus`` that have enough free VRAM to satisfy
    ``budget_mb + reserve_mb``.
    """
    needed = budget_mb + reserve_mb
    return [gpu for gpu in gpus if gpu.get("memory.free", 0) >= needed]


# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on a seed and a token."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian kernel evaluated at ``theta``."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam.
    ``theta`` is the parameter, ``center`` the mean, ``width`` the std‑dev.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


@dataclass(frozen=True)
class MathAction:
    """
    Logical action / model group used by the hybrid allocator.
    """
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Hybrid core – mathematical bridge
# ----------------------------------------------------------------------
def weekday_fisher_weight_vector(
    actions: Sequence[MathAction],
    date: dt.date,
    fisher_center: float = 0.0,
    fisher_width: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Build the hybrid weight vector ``H`` where

        Hₖ = wₖ · Fₖ

    * ``wₖ`` – weekday sinusoidal weight (parent A).
    * ``Fₖ`` – Fisher information score derived from the action identifier
      (parent B).

    The returned vector is normalised to sum to 1.
    """
    # 1️⃣ Weekday component
    dow = (date.weekday() + 1) % 7          # 0 = Sunday … 6 = Saturday
    w_vec = weekday_weight_vector([a.id for a in actions], dow)

    # 2️⃣ Fisher component – map each action to an angle via a deterministic hash
    angles = np.empty(len(actions), dtype=np.float64)
    for i, act in enumerate(actions):
        h = _hash(seed, act.id)
        # Map 64‑bit integer uniformly to [0, 2π)
        angles[i] = (h % MAX64) / MAX64 * 2.0 * math.pi

    f_vec = np.array(
        [fisher_score(theta, fisher_center, fisher_width) for theta in angles],
        dtype=np.float64,
    )

    # 3️⃣ Hybrid product and normalisation
    hybrid_raw = w_vec * f_vec
    if hybrid_raw.sum() == 0:
        # Fallback to uniform distribution if everything collapses
        hybrid_raw = np.ones_like(hybrid_raw) / len(hybrid_raw)
    else:
        hybrid_raw = hybrid_raw / hybrid_raw.sum()
    return hybrid_raw


def vram_fisher_regret_selection(
    gpus: List[Dict[str, Any]],
    actions: Sequence[MathAction],
    hybrid_weights: np.ndarray,
    budget_mb: int = DEFAULT_BUDGET_MB,
    reserve_mb: int = DEFAULT_RESERVE_MB,
) -> List[Tuple[MathAction, Dict[str, Any]]]:
    """
    Pair actions with GPUs respecting VRAM constraints while respecting the
    hybrid ordering.  The most heavily weighted actions are assigned first to
    the most capable GPUs (greedy allocation).

    Returns a list of ``(action, gpu)`` tuples.
    """
    eligible_gpus = vram_aware_gpu_selection(gpus, budget_mb, reserve_mb)
    if not eligible_gpus:
        raise RuntimeError("No GPU satisfies the VRAM requirements")

    # Sort GPUs descending by free memory for deterministic greedy pairing
    eligible_gpus.sort(key=lambda g: g.get("memory.free", 0), reverse=True)

    # Sort actions by descending hybrid weight
    sorted_idx = np.argsort(-hybrid_weights)
    pairs: List[Tuple[MathAction, Dict[str, Any]]] = []

    gpu_cycle = iter(eligible_gpus)
    for idx in sorted_idx:
        try:
            gpu = next(gpu_cycle)
        except StopIteration:
            # Restart cycle if we exhausted the list – allows multiple actions per GPU
            gpu_cycle = iter(eligible_gpus)
            gpu = next(gpu_cycle)
        pairs.append((actions[idx], gpu))

    return pairs


def hybrid_allocate(
    total_units: float,
    date: dt.date,
    actions: Sequence[MathAction],
    gpus: List[Dict[str, Any]],
    budget_mb: int = DEFAULT_BUDGET_MB,
    reserve_mb: int = DEFAULT_RESERVE_MB,
    fisher_center: float = 0.0,
    fisher_width: float = 1.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Allocate ``total_units`` of a generic resource (e.g. token budget) across
    ``actions`` while simultaneously selecting VRAM‑compatible GPUs.
    The allocation proportion for each action is exactly its hybrid weight.

    Returns a mapping ``action_id -> { 'units': ..., 'gpu': ... }``.
    """
    # 1️⃣ Compute hybrid weights
    hybrid_weights = weekday_fisher_weight_vector(
        actions,
        date,
        fisher_center=fisher_center,
        fisher_width=fisher_width,
    )

    # 2️⃣ Pair actions with GPUs
    pairs = vram_fisher_regret_selection(
        gpus,
        actions,
        hybrid_weights,
        budget_mb=budget_mb,
        reserve_mb=reserve_mb,
    )

    # 3️⃣ Distribute units proportionally
    allocation: Dict[str, Dict[str, Any]] = {}
    for (action, gpu), weight in zip(pairs, hybrid_weights):
        allocated = _pct(total_units * weight)
        allocation[action.id] = {
            "units": allocated,
            "gpu_id": gpu.get("id", "unknown"),
            "gpu_free_mb": gpu.get("memory.free", 0),
            "weight": _pct(weight),
        }

    return allocation


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal synthetic data to exercise the hybrid engine
    today = dt.date.today()

    # Define four dummy actions
    actions = [
        MathAction(id="codex", expected_value=0.9, cost=0.1),
        MathAction(id="groq", expected_value=0.7, cost=0.05),
        MathAction(id="cohere", expected_value=0.6, cost=0.2),
        MathAction(id="local_models", expected_value=0.8, cost=0.15),
    ]

    # Fake GPU inventory
    gpus = [
        {"id": "gpu-A", "memory.free": 8192},
        {"id": "gpu-B", "memory.free": 6144},
        {"id": "gpu-C", "memory.free": 4096},
    ]

    # Run the allocator
    result = hybrid_allocate(
        total_units=1000.0,
        date=today,
        actions=actions,
        gpus=gpus,
        budget_mb=2048,
        reserve_mb=256,
        fisher_center=0.0,
        fisher_width=1.0,
    )

    # Print a tidy summary
    print("Hybrid Allocation Result")
    print("-" * 30)
    for aid, info in result.items():
        print(
            f"Action {aid!r}: {info['units']} units "
            f"(weight {info['weight']}) -> GPU {info['gpu_id']} "
            f"(free {info['gpu_free_mb']} MB)"
        )
    sys.exit(0)