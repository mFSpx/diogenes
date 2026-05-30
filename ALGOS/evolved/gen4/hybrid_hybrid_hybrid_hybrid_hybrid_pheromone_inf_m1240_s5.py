# DARWIN HAMMER — match 1240, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s0.py (gen3)
# parent_b: hybrid_pheromone_infotaxis_m3_s3.py (gen1)
# born: 2026-05-29T23:34:45Z

import numpy as np
import math
import random
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GROUPS: tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1
DEFAULT_BUDGET_MB = 4096
DEFAULT_RESERVE_MB = 768

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where 0 = Sunday … 6 = Saturday."""
    import datetime
    return (datetime.date(year, month, day).weekday() + 1) % 7

def weekday_weight_vector(groups: list[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    Sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow / 7)
    amplitude = 0.9  # Adjusted amplitude to ensure row-stochastic vector
    weights = 0.5 + 0.5 * np.sin(base_angles + phase)  # Normalized to [0, 1]
    return weights / np.sum(weights)  # Normalize to ensure row-stochastic

class PheromoneSystem:
    def __init__(self):
        self.pheromones = {}

    def calculate_pheromone_signal(self, surface_key, signal_kind, signal_value, half_life_seconds):
        import datetime
        current_time = datetime.datetime.now(datetime.timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds, 'created_time': current_time}
        else:
            previous_signal_value = self.pheromones[surface_key]['signal_value']
            previous_half_life_seconds = self.pheromones[surface_key]['half_life_seconds']
            previous_created_time = self.pheromones[surface_key]['created_time']
            elapsed_time = (current_time - previous_created_time).total_seconds()
            decayed_signal_value = previous_signal_value * math.pow(0.5, elapsed_time / previous_half_life_seconds)
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds, 'created_time': current_time}
            return decayed_signal_value
        return signal_value

    def calculate_entropy(self, probabilities, eps=1e-12):
        total = sum(probabilities)
        if total <= 0:
            raise ValueError('positive probability mass required')
        return -sum((p/total) * math.log(max(p/total, eps)) for p in probabilities if p > 0)

    def expected_entropy(self, p_hit, hit_state, miss_state):
        if not 0 <= p_hit <= 1:
            raise ValueError('p_hit must be in [0,1]')
        return p_hit * self.calculate_entropy(hit_state) + (1.0 - p_hit) * self.calculate_entropy(miss_state)

def calculate_hybrid_entropy(pheromone_system: PheromoneSystem, groups: list[str], dow: int):
    weight_vector = weekday_weight_vector(groups, dow)
    signal_values = [pheromone_system.pheromones.get(group, {'signal_value': 0})['signal_value'] for group in groups]
    probabilities = [weight * signal_value for weight, signal_value in zip(weight_vector, signal_values)]
    probabilities = [p / sum(probabilities) for p in probabilities]  # Normalize probabilities
    return pheromone_system.calculate_entropy(probabilities)

def calculate_hybrid_expected_entropy(pheromone_system: PheromoneSystem, groups: list[str], dow: int, p_hit, hit_state, miss_state):
    weight_vector = weekday_weight_vector(groups, dow)
    signal_values = [pheromone_system.pheromones.get(group, {'signal_value': 0})['signal_value'] for group in groups]
    probabilities = [weight * signal_value for weight, signal_value in zip(weight_vector, signal_values)]
    probabilities = [p / sum(probabilities) for p in probabilities]  # Normalize probabilities
    return pheromone_system.expected_entropy(p_hit, hit_state, miss_state)

if __name__ == "__main__":
    pheromone_system = PheromoneSystem()
    groups = list(GROUPS)
    dow = doomsday(2024, 9, 16)
    pheromone_system.calculate_pheromone_signal(groups[0], 'kind', 1.0, 3600)
    print(calculate_hybrid_entropy(pheromone_system, groups, dow))
    print(calculate_hybrid_expected_entropy(pheromone_system, groups, dow, 0.5, [0.2, 0.3, 0.5], [0.1, 0.2, 0.7]))