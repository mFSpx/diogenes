# DARWIN HAMMER — match 480, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_rlct_grokking_hybrid_pheromone_inf_m208_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_endpoi_epistemic_certainty_m59_s0.py (gen3)
# born: 2026-05-29T23:29:27Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np

EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


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
    """Maps righting‑time index to a confidence weight in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


class PheromoneRLCTSystem:
    def __init__(self):
        self.pheromone_signals: Dict[str, float] = {}

    @staticmethod
    def estimate_rlct_from_losses(train_losses_per_n: List[float],
                                  n_values: List[float]) -> float:
        losses = np.asarray(train_losses_per_n, dtype=np.float64)
        ns = np.asarray(n_values, dtype=np.float64)

        if np.any(ns <= np.e):
            raise ValueError("all n_values must be > e for log(log(n)) to be positive")
        if losses.shape != ns.shape:
            raise ValueError("train_losses_per_n and n_values must have the same length")

        y = np.log(np.maximum(losses, 1e-300))
        x = np.log(np.log(ns))

        x_c = x - x.mean()
        y_c = y - y.mean()
        var_x = (x_c ** 2).sum()
        if var_x < 1e-15:
            raise ValueError("n_values have no variance in log(log(n)) space")
        return float((x_c * y_c).sum() / var_x)

    @staticmethod
    def sodium_current(V: float, m: float, h: float,
                       g_Na: float = 120.0, E_Na: float = 50.0) -> float:
        return g_Na * (m ** 3) * h * (V - E_Na)

    @staticmethod
    def potassium_current(V: float, n: float,
                          g_K: float = 36.0, E_K: float = -77.0) -> float:
        return g_K * (n ** 4) * (V - E_K)

    @staticmethod
    def optimize_energy(V: float, m: float, h: float, n: float,
                        g_Na: float = 120.0, E_Na: float = 50.0,
                        g_K: float = 36.0, E_K: float = -77.0) -> float:
        return (PheromoneRLCTSystem.sodium_current(V, m, h, g_Na, E_Na) +
                PheromoneRLCTSystem.potassium_current(V, n, g_K, E_K))

    def decay_pheromones(self, half_life: float = 3600.0, dt: float = 1.0) -> None:
        decay_factor = 0.5 ** (dt / half_life)
        for loc in list(self.pheromone_signals.keys()):
            self.pheromone_signals[loc] *= decay_factor
            if self.pheromone_signals[loc] < 1e-12:
                del self.pheromone_signals[loc]

    def deposit_pheromone(self, location: str, amount: float) -> None:
        self.pheromone_signals[location] = self.pheromone_signals.get(location, 0.0) + amount

    def expected_entropy(self) -> float:
        total = sum(self.pheromone_signals.values())
        if total == 0:
            return 0.0
        probs = np.array(list(self.pheromone_signals.values())) / total
        return -float(np.sum(probs * np.log(np.maximum(probs, 1e-300))))


def compute_certainty_factor(rlct: float) -> float:
    if rlct == 0:
        return 0.0
    entropy_like = -math.log(abs(rlct) + 1e-12)  
    phi = 1.0 / (1.0 + math.exp(entropy_like - 5.0)) # adjusted to better saturate
    return max(0.0, min(1.0, phi))


def hybrid_score(V: float, m: float, h: float, n: float,
                 train_losses: List[float], n_vals: List[float],
                 morphology: Morphology,
                 pheromone_system: PheromoneRLCTSystem,
                 location: str) -> float:
    energy = pheromone_system.optimize_energy(V, m, h, n)

    rlct = pheromone_system.estimate_rlct_from_losses(train_losses, n_vals)
    phi = compute_certainty_factor(rlct)

    priority = recovery_priority(morphology)

    entropy = pheromone_system.expected_entropy()
    pheromone_weight = math.exp(-entropy)  

    score = ( -energy * 0.4  
              + phi * 0.25 # slightly reduced weight
              + priority * 0.2 
              + pheromone_weight * 0.15)

    return score


def adaptive_pheromone_update(pheromone_system: PheromoneRLCTSystem,
                              location: str,
                              score: float,
                              alpha: float = 0.1) -> None:
    current_pheromone = pheromone_system.pheromone_signals.get(location, 0.0)
    updated_pheromone = current_pheromone + alpha * (score - 0.5)
    pheromone_system.deposit_pheromone(location, updated_pheromone - current_pheromone)


def main():
    # Example usage
    pheromone_system = PheromoneRLCTSystem()
    morphology = Morphology(1.0, 1.0, 1.0, 1.0)

    train_losses = [0.1, 0.2, 0.3]
    n_vals = [10, 20, 30]

    V, m, h, n = 1.0, 1.0, 1.0, 1.0
    location = "example_location"

    score = hybrid_score(V, m, h, n, train_losses, n_vals, morphology, pheromone_system, location)
    adaptive_pheromone_update(pheromone_system, location, score)


if __name__ == "__main__":
    main()