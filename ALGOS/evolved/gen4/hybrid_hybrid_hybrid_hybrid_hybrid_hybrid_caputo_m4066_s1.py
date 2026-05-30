# DARWIN HAMMER — match 4066, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_ternar_m374_s2.py (gen3)
# parent_b: hybrid_hybrid_caputo_fracti_hybrid_percyphon_hyb_m352_s0.py (gen3)
# born: 2026-05-29T23:53:22Z

"""Hybrid Fractional Pheromone System
Parents:
- hybrid_hybrid_hybrid_pherom_hybrid_hybrid_ternar_m374_s2.py
- hybrid_hybrid_caputo_fracti_hybrid_percyphon_hyb_m352_s0.py

Mathematical Bridge:
The pheromone decay in the original HybridPheromoneSystem follows an exponential
half‑life model.  Here we replace the pure exponential kernel with a
Caputo‑fractional power‑law decay.  The fractional order α is derived from the
morphology of a procedurally generated entity (length, width, height, mass) via
its sphericity index.  The Caputo derivative provides a history‑dependent decay
rate, while the pheromone signal still modulates decision scores together with
a similarity metric (e.g., SSIM).  Thus the two topologies are fused:
  · Morphology → α (fractional order) → fractional_decay kernel → effective
    half‑life for pheromone update.
  · Pheromone signal + similarity → decision score.
"""

import math
import random
import sys
import pathlib
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np

# ---------- Fractional Calculus Utilities (Parent B) ----------
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


def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of Γ(z) for z > 0."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    else:
        return math.sqrt(2 * math.pi) * (z + _LANCZOS_G + 0.5) ** (z + 0.5) * \
               math.exp(-(z + _LANCZOS_G + 0.5)) * np.polyval(_LANCZOS_C, z)


def caputo_derivative(values: np.ndarray, alpha: float, dt: float) -> np.ndarray:
    """
    Discrete Caputo fractional derivative of order ``alpha`` for a 1‑D array.
    ``values`` must be ordered in time, ``dt`` is the uniform time step.
    """
    n = len(values)
    coeff = np.array([((t + 1) ** (1 - alpha) - t ** (1 - alpha)) for t in range(n)])
    coeff = coeff / gamma_lanczos(1 - alpha)
    derivative = np.zeros_like(values, dtype=float)
    for i in range(1, n):
        derivative[i] = np.sum(values[:i][::-1] * coeff[1:i+1]) / dt
    return derivative


def fractional_decay(alpha: float):
    """Return a power‑law decay kernel λ(t) = t^{-α} (up to a constant)."""
    def kernel(t: float) -> float:
        return t ** (-alpha) if t > 0 else 1.0
    return kernel


# ---------- Procedural Morphology (Parent B) ----------
def generate_morphology() -> Dict[str, float]:
    """Randomly generate length, width, height and mass for an entity."""
    length = random.uniform(0.5, 5.0)
    width = random.uniform(0.5, 5.0)
    height = random.uniform(0.5, 5.0)
    mass = random.uniform(1.0, 20.0)
    return {"length": length, "width": width, "height": height, "mass": mass}


def sphericity_index(morph: Dict[str, float]) -> float:
    """Compute a simple sphericity index from the morphology."""
    l, w, h = morph["length"], morph["width"], morph["height"]
    volume = l * w * h
    surface = 2 * (l * w + w * h + h * l)
    if surface == 0:
        return 0.0
    sph = (math.pi ** (1.0 / 3.0) * (6 * volume) ** (2.0 / 3.0)) / surface
    return max(0.0, min(1.0, sph))  # clamp to [0,1]


# ---------- Hybrid Pheromone System (Parent A) ----------
class HybridPheromoneSystem:
    def __init__(self):
        self.pheromones: Dict[str, Dict] = {}
        self.total_records = 0

    def _current_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def calculate_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
        morphology: Dict[str, float],
        alpha_base: float = 0.5,
    ) -> float:
        """
        Update (or create) a pheromone entry using a fractional decay kernel.
        The fractional order α is obtained from the entity's sphericity.
        """
        now = self._current_utc()
        sph = sphericity_index(morphology)
        alpha = max(0.01, min(0.99, alpha_base * sph))  # ensure 0<α<1
        decay_kernel = fractional_decay(alpha)

        if surface_key not in self.pheromones:
            effective_half_life = half_life_seconds
        else:
            prev = self.pheromones[surface_key]
            elapsed = (now - prev["created_time"]).total_seconds()
            # Apply power‑law correction to the exponential half‑life
            correction = decay_kernel(elapsed)
            effective_half_life = half_life_seconds * (1.0 + correction)

        # Store the updated pheromone
        self.pheromones[surface_key] = {
            "signal_kind": signal_kind,
            "signal_value": signal_value,
            "half_life_seconds": effective_half_life,
            "created_time": now,
            "alpha": alpha,
        }
        return signal_value

    def get_pheromone(self, surface_key: str) -> Tuple[float, float]:
        """Return (signal_value, remaining_fraction) for a given key."""
        if surface_key not in self.pheromones:
            return 0.0, 0.0
        entry = self.pheromones[surface_key]
        now = self._current_utc()
        elapsed = (now - entry["created_time"]).total_seconds()
        half_life = entry["half_life_seconds"]
        # Exponential decay with the effective half‑life
        remaining = 0.5 ** (elapsed / half_life) if half_life > 0 else 0.0
        return entry["signal_value"], remaining


# ---------- Hybrid Operations ----------
def similarity_score(payload_a: np.ndarray, payload_b: np.ndarray) -> float:
    """
    Very lightweight similarity metric (placeholder for SSIM).
    Returns a value in [0,1].
    """
    if payload_a.shape != payload_b.shape:
        return 0.0
    a = payload_a.astype(float).flatten()
    b = payload_b.astype(float).flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine = np.dot(a, b) / (norm_a * norm_b)
    return max(0.0, min(1.0, cosine))


def decision_score(
    system: HybridPheromoneSystem,
    surface_key: str,
    similarity: float,
) -> float:
    """
    Combine pheromone signal with payload similarity to produce a decision metric.
    """
    signal, remaining = system.get_pheromone(surface_key)
    # The remaining fraction modulates confidence; similarity modulates relevance.
    return signal * remaining * similarity


def fractional_pheromone_history(
    system: HybridPheromoneSystem,
    surface_key: str,
    times: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Retrieve the time‑series of pheromone signal values for a key and compute its
    Caputo fractional derivative, returning the derivative array.
    """
    values = np.array([
        system.pheromones.get(surface_key, {"signal_value": 0.0})["signal_value"]
        for _ in times
    ], dtype=float)
    return caputo_derivative(values, alpha=0.5, dt=dt)


# ---------- Smoke Test ----------
if __name__ == "__main__":
    # Initialise system
    system = HybridPheromoneSystem()

    # Generate a random morphology and compute α‑modulated pheromone
    morph = generate_morphology()
    key = "node_42"
    system.calculate_pheromone_signal(
        surface_key=key,
        signal_kind="access",
        signal_value=1.0,
        half_life_seconds=30.0,
        morphology=morph,
        alpha_base=0.6,
    )

    # Simulate two payloads and compute similarity
    payload1 = np.random.rand(64, 64)
    payload2 = payload1 + np.random.normal(0, 0.05, size=payload1.shape)
    sim = similarity_score(payload1, payload2)

    # Decision metric
    score = decision_score(system, key, sim)
    print(f"Decision score for {key}: {score:.4f}")

    # Fractional derivative history (dummy time points)
    times = np.arange(0, 10, 1)
    derivative = fractional_pheromone_history(system, key, times)
    print("Caputo fractional derivative of pheromone series:", derivative)