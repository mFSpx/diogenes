# DARWIN HAMMER — match 96, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_geomet_koopman_operator_m59_s0.py (gen3)
# parent_b: hybrid_hybrid_pheromone_inf_hybrid_decision_hygi_m37_s3.py (gen2)
# born: 2026-05-29T23:28:10Z

"""
Hybrid module combining geometric algebra (from hybrid_hybrid_geometric_pro_decision_hygiene_m25_s1.py) 
and pheromone-based surface usage tracking with entropy-based action selection (from hybrid_pheromone_infotaxis_m3_s0.py and hybrid_decision_hygiene_shannon_entropy_m12_s0.py).
The mathematical bridge is established by applying the Koopman operator to the multivector representation of the geometric algebra, 
and then using the Shannon entropy calculation to analyze the distribution of decision hygiene scores, 
which are then used to inform the pheromone probabilities, ultimately guiding the selection of actions based on surface usage patterns and decision-making processes.
"""

import math
import random
import sys
import pathlib
import numpy as np

# Geometric algebra core
def _blade_sign(indices: list) -> tuple:
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


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> tuple:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components: dict, n: int):
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> 'Multivector':
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)


def koopman_operator(multivector: Multivector, X: np.ndarray, X_prime: np.ndarray) -> np.ndarray:
    """Apply the Koopman operator to the multivector representation."""
    # Convert the multivector to a numpy array
    multivector_array = np.array([coef for coef in multivector.components.values()])

    # Apply the Koopman operator
    koopman_matrix = np.dot(X_prime.T, X)
    result = np.dot(koopman_matrix, multivector_array)

    return result


def shannon_entropy(probabilities: np.ndarray) -> float:
    """Calculate the Shannon entropy of a probability distribution."""
    return -np.sum(probabilities * np.log2(probabilities))


def pheromone_update(probabilities: np.ndarray, entropy: float) -> np.ndarray:
    """Update the pheromone probabilities based on the Shannon entropy."""
    # Normalize the probabilities
    probabilities = probabilities / np.sum(probabilities)

    # Update the pheromone probabilities based on the entropy
    pheromone_probabilities = probabilities * np.exp(-entropy)

    return pheromone_probabilities


def hybrid_operation(X: np.ndarray, X_prime: np.ndarray, components: dict, n: int) -> np.ndarray:
    """Perform the hybrid operation."""
    # Create a multivector
    multivector = Multivector(components, n)

    # Apply the Koopman operator
    result = koopman_operator(multivector, X, X_prime)

    # Calculate the Shannon entropy of the decision hygiene scores
    decision_hygiene_scores = np.array([components[blade] for blade in multivector.components.keys()])
    entropy = shannon_entropy(np.exp(-decision_hygiene_scores))

    # Update the pheromone probabilities based on the entropy
    pheromone_probabilities = pheromone_update(np.array([1.0, 1.0]), entropy)

    return result, pheromone_probabilities


if __name__ == "__main__":
    # Smoke test
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_prime = np.array([[5.0, 6.0], [7.0, 8.0]])
    components = {frozenset([1]): 0.5, frozenset([2]): 0.5}
    n = 2

    result, pheromone_probabilities = hybrid_operation(X, X_prime, components, n)

    print(result)
    print(pheromone_probabilities)