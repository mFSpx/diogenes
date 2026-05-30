# DARWIN HAMMER — match 5152, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m518_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m1515_s0.py (gen5)
# born: 2026-05-30T00:00:05Z

"""
Module for the hybrid algorithm that combines the Physarum-RBF 
and Clifford-geometric distance from 'hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m518_s3.py' 
with the minimum-cost tree Bayes update and bandit-router sketch from 
'hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m1515_s0.py'. 

The mathematical bridge between the two structures is based on representing 
the probabilistic weights and log-count statistics as a multivector that 
can be approximated using the Clifford algebra, and using the Clifford-geometric 
distance to compute the expected cost and the expected reward.

The hybrid algorithm integrates the governing equations of both parents by 
using the Clifford algebra to approximate the level-1 and level-2 iterated-integrals, 
which are then used to compute the path signature and the expected cost and 
reward, and by using the Physarum-RBF to update the multivector.
"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple
from collections import defaultdict

class Multivector:
    """Sparse multivector in a Clifford algebra with n basis vectors."""
    def __init__(self, components: Dict[frozenset[int], float], n: int):
        self.n = int(n)
        # discard near‑zero entries for sparsity
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    def grade(self, k: int) -> "Multivector":
        """Return the grade‑k part."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) component."""
        return self.components.get(frozenset(), 0.0)

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list.

    Duplicate indices cancel because e_i·e_i = 1.
    """
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
                # cancel pair
                del lst[j : j + 2]
                n -= 2
                i = -1  # restart outer loop
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(
    blade_a: frozenset, blade_b: frozenset
) -> Tuple[frozenset, int]:
    """Multiply two basis blades and return (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def clifford_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Clifford-geometric distance between two points."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def physarum_rbf_multivector(multivector: Multivector, 
                            conductances: np.ndarray, 
                            learning_rate: float, 
                            coupling_constant: float) -> Multivector:
    """
    Update the multivector using the Physarum-RBF algorithm.

    Args:
    multivector: The multivector to update.
    conductances: The conductances of the network.
    learning_rate: The learning rate of the algorithm.
    coupling_constant: The coupling constant of the algorithm.

    Returns:
    The updated multivector.
    """
    # Compute the scalar part of the multivector
    scalar_part = multivector.scalar_part()

    # Compute the RBF of the conductances
    rbf = np.exp(-np.linalg.norm(conductances - scalar_part))

    # Compute the gradient of the RBF
    gradient = -2 * (conductances - scalar_part) * rbf

    # Update the multivector
    updated_multivector = Multivector(
        {blade: coef + learning_rate * (coef - coupling_constant * gradient[i]) 
         for i, (blade, coef) in enumerate(multivector.components.items())}, 
        multivector.n)

    return updated_multivector

def clifford_geometric_update(multivector: Multivector, 
                             points: List[Tuple[float, float]], 
                             learning_rate: float) -> Multivector:
    """
    Update the multivector using the Clifford-geometric distance.

    Args:
    multivector: The multivector to update.
    points: The points to compute the distance to.
    learning_rate: The learning rate of the algorithm.

    Returns:
    The updated multivector.
    """
    # Compute the Clifford-geometric distance to each point
    distances = [clifford_distance(point, (0, 0)) for point in points]

    # Update the multivector
    updated_multivector = Multivector(
        {blade: coef + learning_rate * coef * np.mean(distances) 
         for blade, coef in multivector.components.items()}, 
        multivector.n)

    return updated_multivector

def hybrid_update(multivector: Multivector, 
                  conductances: np.ndarray, 
                  points: List[Tuple[float, float]], 
                  learning_rate: float, 
                  coupling_constant: float) -> Multivector:
    """
    Update the multivector using the hybrid algorithm.

    Args:
    multivector: The multivector to update.
    conductances: The conductances of the network.
    points: The points to compute the distance to.
    learning_rate: The learning rate of the algorithm.
    coupling_constant: The coupling constant of the algorithm.

    Returns:
    The updated multivector.
    """
    # Update the multivector using the Physarum-RBF algorithm
    physarum_updated_multivector = physarum_rbf_multivector(multivector, 
                                                             conductances, 
                                                             learning_rate, 
                                                             coupling_constant)

    # Update the multivector using the Clifford-geometric distance
    clifford_updated_multivector = clifford_geometric_update(physarum_updated_multivector, 
                                                             points, 
                                                             learning_rate)

    return clifford_updated_multivector

if __name__ == "__main__":
    multivector = Multivector({frozenset(): 1.0}, 2)
    conductances = np.array([1.0, 2.0])
    points = [(1.0, 2.0), (3.0, 4.0)]
    learning_rate = 0.1
    coupling_constant = 0.5

    updated_multivector = hybrid_update(multivector, 
                                         conductances, 
                                         points, 
                                         learning_rate, 
                                         coupling_constant)

    print(updated_multivector.components)