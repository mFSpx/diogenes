# DARWIN HAMMER — match 2727, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_sheaf_cohomol_m179_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_fisher_m31_s4.py (gen4)
# born: 2026-05-29T23:43:45Z

"""
Hybrid allocation and multivector consistency module.

This module fuses two parent algorithms:
- **Parent A** (`hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s3.py`): 
  computes a weekday-dependent weight vector and allocates a total resource 
  into deterministic and residual parts across a set of groups.
- **Parent B** (`hybrid_hybrid_hybrid_geomet_hybrid_hybrid_fisher_m31_s4.py`): 
  defines a cellular multivector on a graph, with linear restriction maps between 
  node blades and edge blades, and provides tools to test section consistency.

**Mathematical bridge**  
Both parents operate on linear objects.  The weight vector from Parent A is a 
row-stochastic vector  w∈ℝ^|G| that linearly maps a scalar residual r to a 
vector of allocations r·w.  In a multivector, a *section* assigns a vector to each node; 
the collection of allocations is therefore a multivector section over a graph whose 
nodes are the groups.  By equipping the graph with identity restriction maps,
the coboundary operator reduces to differences of neighbouring allocations.  
Thus we can test the “coherence’’ of the allocation by measuring the norm of the
coboundary, directly linking the two parent topologies.

The functions below implement this hybrid logic.
"""
import math
import random
import sys
import pathlib
import numpy as np
from typing import Tuple, List, Dict

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1

# ----------------------------------------------------------------------
# Utility helpers (from Parent A)
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    return (dt.date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Normalised weigh
    """
    # compute weights based on day of week
    weights = np.zeros(len(groups))
    for i, group in enumerate(groups):
        weights[i] = 1 / len(groups)  # uniform weights
    # adjust weights based on day of week
    weights *= np.sin(dow * math.pi / 7) + 1  # add 1 to shift to [1, 2]
    weights /= np.sum(weights)
    return weights


# ----------------------------------------------------------------------
# Utility helpers (from Parent B)
# ----------------------------------------------------------------------
class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components: Dict[frozenset[int], float], n: int):
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0) + coef
        return Multivector(result, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({blade: coef * scalar for blade, coef in self.components.items()}, self.n)


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> Tuple[frozenset[int], int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble-sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j:j + 2]
                n -= 2
                sign *= 1
                continue
            j += 1
        i += 1
    return lst, sign


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_allocation(groups: Sequence[str], dow: int, residual: float) -> Multivector:
    """Compute allocation as a multivector section."""
    weights = weekday_weight_vector(groups, dow)
    allocation = Multivector({frozenset([i]): weights[i] * residual for i in range(len(groups))}, len(groups))
    return allocation


def hybrid_coboundary(allocation: Multivector) -> float:
    """Compute coboundary norm of a multivector section."""
    # since we use identity restriction maps, the coboundary reduces to differences
    # between neighbouring allocations
    coboundary = np.zeros(len(allocation.components))
    for i in range(len(allocation.components)):
        j = (i + 1) % len(allocation.components)
        coboundary[i] = np.linalg.norm(allocation.components[frozenset({i})] - allocation.components[frozenset({j})])
    return np.linalg.norm(coboundary)


def hybrid_tests():
    """Run some smoke tests on the hybrid functions."""
    groups = ("codex", "groq", "cohere", "local_models")
    dow = doomsday(2024, 3, 15)  # Friday
    residual = 1.0
    allocation = hybrid_allocation(groups, dow, residual)
    print(allocation.components)
    print(hybrid_coboundary(allocation))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    hybrid_tests()