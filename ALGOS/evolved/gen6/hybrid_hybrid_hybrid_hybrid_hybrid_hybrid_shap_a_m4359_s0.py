# DARWIN HAMMER — match 4359, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_sheaf_cohomol_m179_s2.py (gen3)
# parent_b: hybrid_hybrid_shap_attribut_dense_associative_me_m2066_s0.py (gen5)
# born: 2026-05-29T23:55:04Z

import numpy as np
import random
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
    Normalised weight vector for a given day of the week.
    """
    # Create a matrix where each row represents a group
    group_matrix = np.eye(len(groups))
    # Get the day of the week and calculate the weight vector
    weight_vector = np.zeros(len(groups))
    weight_vector[dow] = 1
    return group_matrix @ weight_vector


# ----------------------------------------------------------------------
# Utility helpers (from Parent B)
# ----------------------------------------------------------------------

def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Compute the Shapley kernel weight for a subset and feature count."""
    return math.factorial(subset_size) * math.factorial(feature_count - subset_size - 1) / math.factorial(feature_count)


def compute_dhash(values: List[float]) -> int:
    """Compute a digital hash for a list of values."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Compute the Hamming distance between two integers."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Hybrid Module
# ----------------------------------------------------------------------

def hybrid_energy_with_shap(xi: np.ndarray, M: np.ndarray, beta: float, feature_index: int, feature_count: int) -> float:
    """
    Compute the hybrid energy function with SHAP value consideration.
    """
    # Compute SHAP value for the given feature index
    shap_score = shap_value(feature_index, feature_count, lambda x: (M @ x).sum())
    # Apply SHAP value to the energy function
    return -beta**(-1) * np.log(
        np.sum(
            np.exp(
                beta * (M @ xi + shap_score)
            )
        )
    ) + 0.5 * (xi ** 2).sum()


def hybrid_retrieve_with_shap(M: np.ndarray, beta: float, query: np.ndarray, feature_index: int, feature_count: int) -> np.ndarray:
    """
    Retrieve a pattern with SHAP value consideration.
    """
    # Compute SHAP value for the given feature index
    shap_score = shap_value(feature_index, feature_count, lambda x: (M @ x).sum())
    # Retrieve the pattern with SHAP value consideration
    return np.exp(beta * (M @ query + shap_score)) / np.sum(np.exp(beta * (M @ query + shap_score)))


def shap_value(feature_index: int, feature_count: int, value_fn: Callable[[int], float]) -> float:
    """
    Compute the SHAP value for a given feature index and feature count.
    """
    total = 0.0
    for k in range(len(value_fn) + 1):
        for subset in combinations(value_fn, k):
            s = frozenset(subset)
            total += shapley_kernel_weight(k, feature_count) * (value_fn(s | {feature_index}) - value_fn(s))
    return total


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------

def hybrid_allocate_resources(groups: Sequence[str], dow: int, resources: float, feature_index: int, feature_count: int) -> Dict[str, float]:
    """
    Allocate resources based on the weekday and feature consideration.
    """
    # Get the weight vector for the given day of the week
    weight_vector = weekday_weight_vector(groups, dow)
    # Compute the SHAP value for the given feature index
    shap_score = shap_value(feature_index, feature_count, lambda x: (weight_vector @ x).sum())
    # Allocate the resources based on the SHAP value and weight vector
    allocations = {}
    for i, group in enumerate(groups):
        allocations[group] = weight_vector[i] * (resources - shap_score)
    return allocations


def hybrid_test_section_consistency(groups: Sequence[str], dow: int, feature_index: int, feature_count: int) -> float:
    """
    Test the section consistency based on the weekday and feature consideration.
    """
    # Get the weight vector for the given day of the week
    weight_vector = weekday_weight_vector(groups, dow)
    # Compute the SHAP value for the given feature index
    shap_score = shap_value(feature_index, feature_count, lambda x: (weight_vector @ x).sum())
    # Compute the coboundary operator
    coboundary = np.zeros(len(groups))
    for i in range(len(groups)):
        coboundary[i] = weight_vector[i] * (shap_score - (weight_vector[i + 1] if i < len(groups) - 1 else 0))
    # Return the norm of the coboundary operator
    return np.linalg.norm(coboundary)


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    groups = ("codex", "groq", "cohere", "local_models")
    dow = 0
    resources = 100.0
    feature_index = 0
    feature_count = 10
    print(hybrid_allocate_resources(groups, dow, resources, feature_index, feature_count))
    print(hybrid_test_section_consistency(groups, dow, feature_index, feature_count))