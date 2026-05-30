# DARWIN HAMMER — match 5150, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m522_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_ssim_h_hybrid_hybrid_hybrid_m2479_s1.py (gen6)
# born: 2026-05-30T00:00:04Z

"""
Module fusion_hybrid: A hybrid algorithm merging regret-weighted strategy 
(Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m522_s2.py) 
with geometric-multivector similarity and contextual multi-armed bandit 
(Parent B: hybrid_hybrid_hybrid_ssim_h_hybrid_hybrid_hybrid_m2479_s1.py).

The mathematical bridge is formed by using the regret-weighted strategy 
as the basis for the bandit's expected rewards, and integrating the 
geometric-multivector similarity as an inflow term to the virtual-store 
dynamics of the bandit. This allows for joint exploitation of geometric-
algebraic similarity and online bandit learning in a single, mathematically 
consistent loop.
"""

import os
import sys
import math
import random
import pathlib
from typing import List, Tuple, Dict, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# VRAM-aware learning-rate utilities
# ----------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2] if pathlib.Path(__file__).exists() else pathlib.Path.cwd()
DEFAULT_BUDGET_MB = int(os.environ.get("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.environ.get("LUCIDOTA_VRAM_RESERVE_MB", "768"))

def _now_iso_z() -> str:
    """Current UTC timestamp in ISO-8601 with trailing “Z”."""
    return sys.modules['datetime'].datetime.now(sys.modules['datetime'].timezone.utc).isoformat().replace("+00:00", "Z")

def _mock_free_vram_mb() -> int:
    """Return a pseudo-random free-VRAM estimate (for CPU-only testing)."""
    total = 8192
    used = random.randint(0, total - DEFAULT_RESERVE_MB)
    return max(total - used - DEFAULT_RESERVE_MB, 0)

def budgeted_lr(base_lr: float,
                free_mb: int,
                budget_mb: int = DEFAULT_BUDGET_MB,
                reserve_mb: int = DEFAULT_RESERVE_MB) -> float:
    """
    Scale ``base_lr`` according to available VRAM.

    If the free memory exceeds the usable budget (budget – reserve) the full
    learning-rate is returned; otherwise a linear decay down to 10 % is applied.
    """
    usable = max(budget_mb - reserve_mb, 1)
    if free_mb >= usable:
        return base_lr
    scale = 0.1 + 0.9 * (free_mb / usable)
    return base_lr * scale

# ----------------------------------------------------------------------
# Regret-weighted strategy
# ----------------------------------------------------------------------
def compute_regret_weighted_strategy(regrets: List[float]) -> List[float]:
    """
    Convert a list of non-negative regrets into a probability distribution.

    The classic regret-matching rule is used:
        p_i ∝ max(regret_i, 0)
    The probabilities sum to one; if all regrets are zero a uniform distribution
    is returned.
    """
    positive = [max(r, 0.0) for r in regrets]
    total = sum(positive)
    if total == 0.0:
        n = len(regrets)
        return [1.0 / n] * n
    return [p / total for p in positive]

# ----------------------------------------------------------------------
# Multivector and geometric similarity
# ----------------------------------------------------------------------
class Multivector:
    """Simple Euclidean Clifford algebra up to grade 2."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        # Remove near‑zero components for cleanliness
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }
        self.n = int(n)  # dimension of the underlying vector space

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, value in other.components.items():
            result[blade] = result.get(blade, 0.0) + value
        return Multivector(result, self.n)

def stats_to_multivector(sequence: List[float]) -> Multivector:
    """Build a grade-0/1/2 multivector from the first two statistical moments of a sequence."""
    mean = np.mean(sequence)
    variance = np.var(sequence)
    components = {frozenset(): mean, frozenset({0}): variance}
    return Multivector(components, 1)

def geometric_ssim(multivector1: Multivector, multivector2: Multivector) -> float:
    """Return the scalar part of the geometric product of the two multivectors – a similarity score."""
    return multivector1.scalar_part() * multivector2.scalar_part()

# ----------------------------------------------------------------------
# Contextual multi-armed bandit
# ----------------------------------------------------------------------
class Bandit:
    def __init__(self, alpha: float, beta: float, store: float):
        self.alpha = alpha
        self.beta = beta
        self.store = store

    def update(self, similarity: float):
        """Update the store value based on the similarity score."""
        self.store += self.alpha * similarity - self.beta * self.store

    def get_expected_reward(self, similarity: float) -> float:
        """Return the expected reward for an action."""
        return similarity + self.store

# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_algorithm(regrets: List[float], sequences: List[List[float]], alpha: float, beta: float, store: float) -> List[float]:
    """
    Jointly exploit geometric-algebraic similarity and online bandit learning.

    Args:
    regrets (List[float]): A list of non-negative regrets.
    sequences (List[List[float]]): A list of sequences, each representing an action and the current context.
    alpha (float): The inflow coefficient for the bandit's store dynamics.
    beta (float): The outflow coefficient for the bandit's store dynamics.
    store (float): The initial store value for the bandit.

    Returns:
    List[float]: A list of propensities for each action, derived from a soft-max over the expected rewards.
    """
    multivectors = [stats_to_multivector(sequence) for sequence in sequences]
    similarities = [geometric_ssim(multivector1, multivector2) for multivector1, multivector2 in zip(multivectors, multivectors[1:] + [multivectors[0]])]
    strategy = compute_regret_weighted_strategy(regrets)
    bandit = Bandit(alpha, beta, store)
    expected_rewards = []
    for i, similarity in enumerate(similarities):
        bandit.update(similarity)
        expected_reward = bandit.get_expected_reward(similarity)
        expected_rewards.append(expected_reward)
    propensities = np.exp(expected_rewards) / np.sum(np.exp(expected_rewards))
    return propensities

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    regrets = [1.0, 2.0, 3.0]
    sequences = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    alpha = 0.1
    beta = 0.1
    store = 0.0
    propensities = hybrid_algorithm(regrets, sequences, alpha, beta, store)
    print(propensities)