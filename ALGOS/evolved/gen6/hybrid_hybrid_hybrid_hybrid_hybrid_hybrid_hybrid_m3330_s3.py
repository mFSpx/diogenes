# DARWIN HAMMER — match 3330, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2354_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m497_s0.py (gen5)
# born: 2026-05-29T23:49:25Z

import numpy as np
import math
import hashlib
from typing import List, Sequence, Tuple

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def _hash_index(depth: int, item: str, width: int) -> int:
    """Deterministic hash used by count_min_sketch."""
    h = hashlib.sha256(f"{depth}:{item}".encode()).hexdigest()
    return int(h, 16) % width


def count_min_sketch(items: Sequence[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """
    Simple Count‑Min Sketch implementation.

    Parameters
    ----------
    items : sequence of hashable objects
        Elements to be inserted.
    width : int, optional
        Number of counters per hash function.
    depth : int, optional
        Number of independent hash functions.

    Returns
    -------
    table : list of lists
        Sketch matrix of shape (depth, width).
    """
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = _hash_index(d, str(item), width)
            table[d][idx] += 1
    return table


# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """
    Gaussian intensity I(θ) of a beam centred at *center* with *width*.

    Parameters
    ----------
    theta, center, width : float
        Beam parameters. ``width`` must be positive.

    Returns
    -------
    float
        Intensity value.
    """
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_information_matrix(
    thetas: Sequence[float],
    center: float = 0.5,
    width: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute the Fisher information matrix for a set of angles under the
    Gaussian‑beam likelihood model.

    The likelihood for a single observation θ is
        p(θ|μ) = N(θ; μ=center, σ=width)

    The Fisher information for a single parameter is
        I(μ) = E[(∂/∂μ log p)^2] = 1 / width²

    For independent observations the matrix is diagonal with that value.
    To keep the implementation generic we compute the empirical version
    using the observed intensities.

    Parameters
    ----------
    thetas : sequence of float
        Observed angles.
    center, width : float
        Parameters of the underlying Gaussian model.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        (n, n) Fisher information matrix, where n = len(thetas).
    """
    thetas = np.asarray(thetas, dtype=np.float64)
    if thetas.ndim != 1:
        raise ValueError("thetas must be a 1‑D sequence")
    n = thetas.size
    intensities = np.maximum(
        np.vectorize(gaussian_beam)(thetas, center, width), eps
    )
    # Derivative of log‑likelihood w.r.t. center for each observation
    deriv = -(thetas - center) / (width * width)
    # Empirical Fisher: (∂ℓ/∂μ) (∂ℓ/∂μ)ᵀ averaged over observations
    fisher_diag = (deriv ** 2) / intensities
    return np.diag(fisher_diag)


def compute_ssim(
    x: Sequence[float],
    y: Sequence[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-12,
) -> float:
    """
    Structural Similarity Index (SSIM) for 1‑D signals.

    The implementation follows the standard definition but adds a small
    epsilon to guard against zero variance.

    Parameters
    ----------
    x, y : sequences of float
        Input signals of equal length.
    dynamic_range : float, optional
        The difference between the maximum and minimum possible values.
    k1, k2 : float, optional
        Stabilisation constants.
    eps : float, optional
        Numerical floor for variance terms.

    Returns
    -------
    float
        SSIM value in [0, 1].
    """
    if len(x) != len(y):
        raise ValueError("inputs must have the same length")
    if not x:
        raise ValueError("inputs must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = max(((x_arr - mx) ** 2).mean(), eps)
    vy = max(((y_arr - my) ** 2).mean(), eps)
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)

    return float(numerator / denominator)


def morphology_vector(
    length: float, width: float, height: float, mass: float
) -> np.ndarray:
    """
    Simple 4‑dimensional morphology descriptor.

    Returns
    -------
    np.ndarray
        Shape (4,).
    """
    return np.array([length, width, height, mass], dtype=np.float64)


# ----------------------------------------------------------------------
# Fusion primitives
# ----------------------------------------------------------------------


def energy_function(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    fisher_weight: float = 0.5,
) -> float:
    """
    Energy that blends SSIM similarity with Fisher information.

    The energy is low when vectors are similar (high SSIM) and when the
    Fisher information (i.e. the sensitivity of the underlying Gaussian
    model) is small.  The `fisher_weight` balances the two contributions.

    Parameters
    ----------
    vec_a, vec_b : np.ndarray
        Morphology vectors of equal length.
    fisher_weight : float, optional
        Weight of the Fisher term in [0, 1].

    Returns
    -------
    float
        Non‑negative scalar energy.
    """
    if vec_a.shape != vec_b.shape:
        raise ValueError("vectors must have the same shape")

    # 1 – SSIM gives a distance‑like term
    ssim_dist = 1.0 - compute_ssim(vec_a.tolist(), vec_b.tolist())

    # Fisher information based on the angular representation of the vectors
    # (we treat each component as an angle for the purpose of the model)
    fisher_mat = fisher_information_matrix(vec_a, center=0.5, width=1.0)
    fisher_trace = np.trace(fisher_mat) / vec_a.size  # average per dimension

    # Blend the two terms
    energy = (1 - fisher_weight) * ssim_dist + fisher_weight * fisher_trace
    return float(max(energy, 0.0))


def hybrid_operation(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    energy_scale: float = 1.0,
) -> np.ndarray:
    """
    Produce a morphology vector that encodes the interaction between
    ``vec_a`` and ``vec_b`` via the energy landscape.

    The returned vector has the same dimensionality as the original
    morphology vectors (assumed to be 4‑D).  If the input vectors have a
    different size, they are first reduced to 4 dimensions by averaging
    groups of components.

    Parameters
    ----------
    vec_a, vec_b : np.ndarray
        Input vectors.
    energy_scale : float, optional
        Multiplicative factor applied to the computed energy before
        embedding it into the output morphology vector.

    Returns
    -------
    np.ndarray
        4‑D morphology vector.
    """
    if vec_a.ndim != 1 or vec_b.ndim != 1:
        raise ValueError("inputs must be 1‑D vectors")

    # Compute scalar energy
    energy = energy_function(vec_a, vec_b)

    # Reduce arbitrary‑length vectors to 4 dimensions by simple averaging.
    # This keeps the operation well‑defined for any input size.
    def _reduce_to_four(v: np.ndarray) -> np.ndarray:
        if v.size == 4:
            return v
        # Pad with the mean value if not divisible by 4
        pad_len = (-v.size) % 4
        if pad_len:
            v = np.concatenate([v, np.full(pad_len, v.mean())])
        return v.reshape(4, -1).mean(axis=1)

    a_red = _reduce_to_four(vec_a)
    b_red = _reduce_to_four(vec_b)

    # Embed the scaled energy uniformly across the four components.
    scaled_energy = energy * energy_scale
    result = morphology_vector(scaled_energy, scaled_energy, scaled_energy, scaled_energy)

    # Optionally blend with the reduced vectors to retain some original
    # structural information.
    blended = 0.5 * result + 0.5 * (a_red + b_red) / 2.0
    return blended.astype(np.float64)


def hybrid_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    fisher_center: float = 0.5,
    fisher_width: float = 1.0,
) -> float:
    """
    Similarity measure that multiplies an average Fisher information term
    with the SSIM between two vectors.

    Parameters
    ----------
    vec_a, vec_b : np.ndarray
        Input vectors.
    fisher_center, fisher_width : float, optional
        Parameters of the Gaussian model used for Fisher computation.

    Returns
    -------
    float
        Product of normalized Fisher information and SSIM.
    """
    if vec_a.shape != vec_b.shape:
        raise ValueError("vectors must have the same shape")

    # Average Fisher information per dimension (normalised to [0, 1])
    fisher_mat = fisher_information_matrix(vec_a, center=fisher_center, width=fisher_width)
    avg_fisher = np.trace(fisher_mat) / vec_a.size
    # Normalise by a plausible upper bound (1/width²) to keep the term ≤1
    fisher_norm = avg_fisher * (fisher_width ** 2)

    ssim_val = compute_ssim(vec_a.tolist(), vec_b.tolist())
    return float(fisher_norm * ssim_val)


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Example vectors of length 5 (arbitrary dimensionality)
    vector_a = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)
    vector_b = np.array([0.3, 0.4, 0.2, 0.6, 0.5], dtype=np.float64)

    print("Hybrid operation output:", hybrid_operation(vector_a, vector_b))
    print("Hybrid similarity:", hybrid_similarity(vector_a, vector_b))