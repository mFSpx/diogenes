# DARWIN HAMMER — match 4698, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_endpoi_m247_s0.py (gen3)
# parent_b: shap_attribution.py (gen0)
# born: 2026-05-29T23:57:38Z

import math
import itertools
import random
from typing import Callable, List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
class Morphology:
    """Geometric description of an object used in the free‑energy and shape
    calculations."""

    def __init__(self, length: float, width: float, height: float, mass: float):
        if min(length, width, height, mass) <= 0:
            raise ValueError("All morphology parameters must be positive.")
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


# ----------------------------------------------------------------------
# Shape descriptors (unchanged, but type‑annotated)
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of the three dimensions to the longest side."""
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness as defined in the original hybrid paper."""
    return (length + width) / (2.0 * height)


# ----------------------------------------------------------------------
# Shapley utilities
# ----------------------------------------------------------------------
def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Exact Shapley kernel weight for a given subset size."""
    return (
        math.factorial(subset_size)
        * math.factorial(feature_count - subset_size - 1)
        / math.factorial(feature_count)
    )


def _enumerate_subsets(
    feature_indices: Tuple[int, ...],
) -> List[Tuple[int, ...]]:
    """Return all subsets of a tuple of feature indices."""
    subsets = []
    for r in range(len(feature_indices) + 1):
        subsets.extend(itertools.combinations(feature_indices, r))
    return subsets


def exact_shapley_value(
    value_fn: Callable[[Tuple[int, ...]], float],
    feature_index: int,
    feature_count: int,
) -> float:
    """
    Compute the exact Shapley value for a single feature.

    Parameters
    ----------
    value_fn
        Callable that receives a tuple of *present* feature indices and returns the
        model's scalar output for that coalition.
    feature_index
        Index of the feature whose contribution we are measuring.
    feature_count
        Total number of features in the problem.

    Returns
    -------
    float
        The Shapley value for the requested feature.
    """
    other_indices = tuple(i for i in range(feature_count) if i != feature_index)
    total = 0.0
    for subset in _enumerate_subsets(other_indices):
        subset_with_feature = tuple(sorted(subset + (feature_index,)))
        weight = shapley_kernel_weight(len(subset), feature_count)
        total += weight * (value_fn(subset_with_feature) - value_fn(subset))
    return total


def monte_carlo_shapley(
    value_fn: Callable[[Tuple[int, ...]], float],
    feature_count: int,
    n_samples: int = 10_000,
    rng: random.Random | None = None,
) -> np.ndarray:
    """
    Approximate Shapley values using Monte‑Carlo sampling of permutations.

    This is far more scalable than the exact exponential algorithm while still
    preserving the theoretical foundation of the Shapley value.

    Parameters
    ----------
    value_fn
        Callable that receives a tuple of present feature indices and returns the
        model's scalar output.
    feature_count
        Number of features.
    n_samples
        Number of random permutations to sample.
    rng
        Optional random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Approximate Shapley values for all features (shape ``(feature_count,)``).
    """
    if rng is None:
        rng = random.Random()
    shapley = np.zeros(feature_count, dtype=float)

    for _ in range(n_samples):
        perm = list(range(feature_count))
        rng.shuffle(perm)
        coalition = ()
        prev_val = value_fn(coalition)
        for idx in perm:
            coalition = tuple(sorted(coalition + (idx,)))
            new_val = value_fn(coalition)
            shapley[idx] += new_val - prev_val
            prev_val = new_val

    shapley /= n_samples
    return shapley


# ----------------------------------------------------------------------
# Variational free‑energy model
# ----------------------------------------------------------------------
def variational_free_energy(
    morphology: Morphology,
    feature_subset: Tuple[int, ...],
    feature_values: np.ndarray,
) -> float:
    """
    Compute a simple Gaussian variational free‑energy for a given subset of
    features.  Missing features are treated as having zero contribution.

    The formulation mirrors the original ``hybrid_variational_free_energy`` but
    now respects the coalition of features supplied by the Shapley machinery.

    Parameters
    ----------
    morphology
        Geometric description used to modulate the energy.
    feature_subset
        Indices of features that are *present* in the coalition.
    feature_values
        Full vector of feature values (unchanged across coalitions).

    Returns
    -------
    float
        Free‑energy estimate for the coalition.
    """
    if not feature_subset:
        # Empty coalition: baseline energy based solely on morphology
        return 0.0

    # Linear contribution of the selected features (mean of selected values)
    selected_mean = np.mean(feature_values[list(feature_subset)])
    return selected_mean * flatness_index(
        morphology.length, morphology.width, morphology.height
    )


# ----------------------------------------------------------------------
# Integrated hybrid functions
# ----------------------------------------------------------------------
def hybrid_endpoint_selection(
    morphology: Morphology,
    features: List[float],
    feature_names: List[str],
    use_exact: bool = False,
    n_mc_samples: int = 5_000,
) -> float:
    """
    Health score that blends variational free‑energy with Shapley‑based feature
    importance.  The score is the dot product of Shapley values and a shape
    scaling factor derived from sphericity.

    Parameters
    ----------
    morphology
        Morphology object.
    features
        List of raw feature values.
    feature_names
        Human‑readable names (only used for validation).
    use_exact
        If ``True`` compute exact Shapley values (exponential cost).  Otherwise
        use Monte‑Carlo approximation.
    n_mc_samples
        Number of permutations for the Monte‑Carlo estimator.

    Returns
    -------
    float
        Health score for the endpoint.
    """
    if len(features) != len(feature_names):
        raise ValueError("features and feature_names must have the same length.")
    feature_array = np.asarray(features, dtype=float)
    f_count = len(features)

    # Define the value function that the Shapley engine will query.
    def value_fn(coalition: Tuple[int, ...]) -> float:
        return variational_free_energy(morphology, coalition, feature_array)

    if use_exact:
        shapley_vals = np.array(
            [
                exact_shapley_value(value_fn, i, f_count)
                for i in range(f_count)
            ],
            dtype=float,
        )
    else:
        shapley_vals = monte_carlo_shapley(
            value_fn, f_count, n_samples=n_mc_samples
        )

    # Scale by a morphology‑dependent factor to embed geometric information.
    scale = sphericity_index(
        morphology.length, morphology.width, morphology.height
    )
    health_score = float(np.dot(shapley_vals, np.ones(f_count)) * scale)
    return health_score


def hybrid_feature_importance(
    morphology: Morphology,
    features: List[float],
    feature_names: List[str],
    use_exact: bool = False,
    n_mc_samples: int = 5_000,
) -> Dict[str, float]:
    """
    Return a dictionary mapping each feature name to its importance, defined as
    the Shapley value multiplied by a sphericity‑based scaling factor.

    Parameters
    ----------
    morphology
        Morphology object.
    features
        List of raw feature values.
    feature_names
        Names of the features.
    use_exact
        Whether to compute exact Shapley values.
    n_mc_samples
        Number of Monte‑Carlo permutations when ``use_exact`` is ``False``.

    Returns
    -------
    dict
        Mapping ``feature_name -> importance``.
    """
    if len(features) != len(feature_names):
        raise ValueError("features and feature_names must have the same length.")
    feature_array = np.asarray(features, dtype=float)
    f_count = len(features)

    def value_fn(coalition: Tuple[int, ...]) -> float:
        return variational_free_energy(morphology, coalition, feature_array)

    if use_exact:
        shapley_vals = np.array(
            [
                exact_shapley_value(value_fn, i, f_count)
                for i in range(f_count)
            ],
            dtype=float,
        )
    else:
        shapley_vals = monte_carlo_shapley(
            value_fn, f_count, n_samples=n_mc_samples
        )

    scale = sphericity_index(
        morphology.length, morphology.width, morphology.height
    )
    importance = {
        name: float(shapley_vals[i] * scale) for i, name in enumerate(feature_names)
    }
    return importance


def hybrid_variational_free_energy(
    morphology: Morphology,
    features: List[float],
) -> float:
    """
    Baseline variational free‑energy (no Shapley integration).  Kept for backward
    compatibility and for cases where a pure energy estimate is desired.
    """
    return np.mean(features) * flatness_index(
        morphology.length, morphology.width, morphology.height
    )


# ----------------------------------------------------------------------
# Simple demonstration (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example morphology and feature set
    morph = Morphology(length=1.2, width=0.8, height=1.5, mass=3.4)
    feats = [0.5, 1.2, -0.3, 2.1]
    names = ["size", "speed", "temperature", "pressure"]

    # Compute health score with Monte‑Carlo Shapley (fast)
    health_mc = hybrid_endpoint_selection(
        morph, feats, names, use_exact=False, n_mc_samples=10_000
    )
    print(f"Health score (MC Shapley): {health_mc:.4f}")

    # Compute health score with exact Shapley (only feasible for ≤5 features)
    health_exact = hybrid_endpoint_selection(
        morph, feats, names, use_exact=True
    )
    print(f"Health score (Exact Shapley): {health_exact:.4f}")

    # Feature importance dictionary
    importance = hybrid_feature_importance(
        morph, feats, names, use_exact=False, n_mc_samples=10_000
    )
    for fn, imp in importance.items():
        print(f"Importance of {fn}: {imp:.4f}")

    # Baseline free‑energy for reference
    print(
        "Baseline variational free energy:",
        hybrid_variational_free_energy(morph, feats),
    )