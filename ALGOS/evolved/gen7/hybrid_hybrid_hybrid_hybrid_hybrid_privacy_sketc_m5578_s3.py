# DARWIN HAMMER — match 5578, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_indy_l_m1306_s1.py (gen6)
# parent_b: hybrid_privacy_sketches_m15_s2.py (gen1)
# born: 2026-05-30T00:03:02Z

import math
import random
import hashlib
import numpy as np
from typing import List, Tuple


# ----------------------------------------------------------------------
# Gaussian / Fisher utilities (Parent A)
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """
    Gaussian beam intensity G(θ) with centre `center` and standard‑deviation `width`.

    Parameters
    ----------
    theta : float
        Angle (or generic coordinate) at which the intensity is evaluated.
    center : float
        Mean of the Gaussian.
    width : float
        Standard deviation; must be positive.

    Returns
    -------
    float
        The intensity value G(θ).
    """
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_information(width: float) -> float:
    """
    Fisher information for the mean of a Gaussian N(μ, σ²) with respect to μ.
    For a Gaussian the Fisher information is I = 1 / σ².

    Parameters
    ----------
    width : float
        Standard deviation σ of the Gaussian.

    Returns
    -------
    float
        Fisher information I = 1 / σ².
    """
    if width <= 0:
        raise ValueError("width must be positive")
    return 1.0 / (width ** 2)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Alternative Fisher‑score implementation that mirrors the original code
    ( (∂G/∂θ)² / G ) but is numerically stable.

    Parameters
    ----------
    theta, center, width : float
        Parameters of the Gaussian beam.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Fisher‑score value.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width ** 2))
    return (derivative ** 2) / intensity


# ----------------------------------------------------------------------
# Count‑Min Sketch utilities (Parent B)
# ----------------------------------------------------------------------


def _cms_hash(item: str, depth: int, width: int) -> List[int]:
    """
    Produce a deterministic list of column indices for the given item,
    one per hash row, using SHA‑256.

    Parameters
    ----------
    item : str
        The element to be hashed.
    depth, width : int
        Dimensions of the sketch.

    Returns
    -------
    List[int]
        Column indices for each row.
    """
    return [
        int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
        for d in range(depth)
    ]


def count_min_sketch(items: List[str], width: int = 64, depth: int = 4) -> np.ndarray:
    """
    Build a Count‑Min Sketch matrix.

    Parameters
    ----------
    items : List[str]
        Stream of items to be inserted.
    width, depth : int
        Sketch dimensions.

    Returns
    -------
    np.ndarray
        Integer matrix of shape (depth, width).
    """
    cms = np.zeros((depth, width), dtype=np.int64)
    for item in items:
        cols = _cms_hash(item, depth, width)
        for d, c in enumerate(cols):
            cms[d, c] += 1
    return cms


def _linear_count_estimate(cms: np.ndarray) -> float:
    """
    Linear‑counting estimator for distinct elements based on the fraction
    of empty cells. This estimator works well when the sketch is not too
    saturated (i.e., occupancy < ~0.9).

    Returns
    -------
    float
        Estimated number of distinct items.
    """
    depth, width = cms.shape
    total_cells = depth * width
    empty_cells = np.count_nonzero(cms == 0)
    if empty_cells == 0:
        # Sketch is fully saturated; fall back to a conservative upper bound.
        return total_cells * 0.99
    fraction_empty = empty_cells / total_cells
    # Linear counting formula: D ≈ -m * ln(f), where m = total_cells.
    return -total_cells * math.log(fraction_empty)


def _estimate_cardinality_from_cms(cms: np.ndarray, fisher_info: float) -> int:
    """
    Combine a linear‑counting estimate with Fisher information.
    The Fisher information acts as a confidence scaling factor:
    higher information (i.e., narrower beam) yields a tighter estimate.

    Parameters
    ----------
    cms : np.ndarray
        Count‑Min Sketch matrix.
    fisher_info : float
        Fisher information derived from the Gaussian beam.

    Returns
    -------
    int
        Cardinality estimate (at least 1).
    """
    # Base estimate from the sketch.
    base_est = _linear_count_estimate(cms)

    # Scale by Fisher information. The scaling is designed so that
    # I = 1 (width = 1) leaves the estimate unchanged, while larger I
    # (narrower beams) reduces variance by shrinking the estimate modestly.
    scaled_est = base_est / (1.0 + math.log1p(fisher_info))

    # Ensure a positive integer result.
    return max(1, int(round(scaled_est)))


def hybrid_cardinality_estimate(
    items: List[str],
    width: int = 64,
    depth: int = 4,
    beam_center: float = 0.0,
    beam_width: float = 1.0,
) -> int:
    """
    Public API: estimate the number of distinct items while incorporating
    Gaussian‑beam Fisher information.

    Parameters
    ----------
    items : List[str]
        Stream of identifiers.
    width, depth : int
        Sketch dimensions.
    beam_center, beam_width : float
        Parameters of the Gaussian beam that model the temporal or spatial
        distribution of the data. Only `beam_width` influences the Fisher
        information; `beam_center` is kept for API compatibility.

    Returns
    -------
    int
        Estimated cardinality.
    """
    cms = count_min_sketch(items, width, depth)
    fisher_info = fisher_information(beam_width)
    return _estimate_cardinality_from_cms(cms, fisher_info)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def hybrid_reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """
    Ratio‑based reconstruction risk, clipped to the interval [0, 1].

    Parameters
    ----------
    unique_quasi_identifiers : int
        Number of distinct quasi‑identifier combinations observed.
    total_records : int
        Total number of records in the dataset.

    Returns
    -------
    float
        Risk score where 0 means no risk and 1 means maximal risk.
    """
    if total_records <= 0:
        return 0.0
    risk = 1.0 - unique_quasi_identifiers / total_records
    return float(min(max(risk, 0.0), 1.0))


def hybrid_privacy_aggregation(cms: np.ndarray, laplace_scale: float) -> np.ndarray:
    """
    Apply Laplace noise to each cell of a Count‑Min Sketch, preserving
    differential privacy.

    Parameters
    ----------
    cms : np.ndarray
        Original sketch.
    laplace_scale : float
        Scale parameter (b) of the Laplace distribution; larger values give
        stronger privacy at the cost of accuracy.

    Returns
    -------
    np.ndarray
        Noised sketch (float dtype).
    """
    if laplace_scale < 0:
        raise ValueError("laplace_scale must be non‑negative")
    noise = np.random.laplace(loc=0.0, scale=laplace_scale, size=cms.shape)
    return cms.astype(float) + noise


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Example data
    items = [f"item{i}" for i in range(1, 101)]

    # Estimate cardinality with a realistic beam width
    cardinality = hybrid_cardinality_estimate(
        items,
        width=128,
        depth=5,
        beam_center=0.0,
        beam_width=0.5,  # narrower beam → higher Fisher information
    )
    print(f"Cardinality estimate: {cardinality}")

    # Compute a reconstruction risk score assuming 1000 total records
    risk = hybrid_reconstruction_risk_score(cardinality, total_records=1000)
    print(f"Reconstruction risk score: {risk:.4f}")

    # Demonstrate privacy‑preserving aggregation
    cms = count_min_sketch(items, width=128, depth=5)
    noisy_cms = hybrid_privacy_aggregation(cms, laplace_scale=1.0)
    print(f"Noisy CMS sample (first row, first 5 cells): {noisy_cms[0, :5]}")