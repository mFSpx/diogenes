# DARWIN HAMMER — match 736, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_path_signatur_hybrid_hybrid_model__m332_s1.py (gen3)
# parent_b: caputo_fractional.py (gen0)
# born: 2026-05-29T23:30:48Z

import numpy as np
from typing import Callable, Tuple


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform of a multivariate path.

    Parameters
    ----------
    path : (T, d) array
        Original time series.

    Returns
    -------
    out : (2*T-1, 2*d) array
        Interleaved lead‑lag representation.  The first ``d`` columns contain the
        “lead’’ channel, the second ``d`` columns the “lag’’ channel.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time × dimension)")

    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)

    for t in range(T - 1):
        # lead at time t, lag at time t
        out[2 * t] = np.concatenate([path[t], path[t]])
        # lead at time t+1, lag at time t
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])

    out[-1] = np.concatenate([path[-1], path[-1]])
    return out


def _augmented_knot_vector(grid: np.ndarray, k: int) -> np.ndarray:
    """
    Build the knot vector with ``k`` repeated end knots (open uniform B‑splines).
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError("grid must be one‑dimensional")
    t_start = np.full(k, grid[0])
    t_end = np.full(k, grid[-1])
    return np.concatenate([t_start, grid, t_end])


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Evaluate all B‑spline basis functions of order ``k`` (degree ``k‑1``)
    on the points ``x`` for a uniform knot vector defined by ``grid``.

    Parameters
    ----------
    x : (N,) array
        Evaluation points.
    grid : (M,) array
        Interior knots (usually the time stamps).
    k : int, default 3
        Order of the spline (k=1 → piecewise constant, k=4 → cubic).

    Returns
    -------
    B : (N, M+k-1) array
        Each column ``i`` contains the value of the i‑th basis function at all
        points in ``x``.
    """
    x = np.asarray(x, dtype=float)
    grid = np.asarray(grid, dtype=float)

    t = _augmented_knot_vector(grid, k)
    n_basis = len(grid) + k - 1
    N = len(x)

    # Initialise zeroth‑order (piecewise constant) basis functions
    B = np.zeros((N, n_basis), dtype=float)
    for i in range(n_basis):
        left = t[i]
        right = t[i + 1]
        # The last knot interval is closed on the right to capture the endpoint
        if i == n_basis - 1:
            B[:, i] = (x >= left) & (x <= right)
        else:
            B[:, i] = (x >= left) & (x < right)

    # Recursion for higher orders
    for order in range(2, k + 1):
        B_next = np.zeros((N, n_basis - (order - 1)), dtype=float)
        for i in range(n_basis - (order - 1)):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]

            term_l = np.zeros(N, dtype=float)
            term_r = np.zeros(N, dtype=float)

            if denom_l > 0:
                term_l = ((x - t[i]) / denom_l) * B[:, i]
            if denom_r > 0:
                term_r = ((t[i + order] - x) / denom_r) * B[:, i + 1]

            B_next[:, i] = term_l + term_r
        B = B_next

    return B


def gamma_lanczos(z: float) -> float:
    """
    Lanczos approximation of the Gamma function for positive real ``z``.
    """
    if z <= 0:
        raise ValueError("Lanczos approximation requires z > 0")
    _g = 7
    _c = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ])
    z_minus_one = z - 1
    x = _c[0]
    for i in range(1, len(_c)):
        x += _c[i] / (z_minus_one + i)
    t = z_minus_one + _g + 0.5
    return np.sqrt(2 * np.pi) * t ** (z_minus_one + 0.5) * np.exp(-t) * x


def caputo_fractional_derivative(
    values: np.ndarray,
    times: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Compute the Caputo fractional derivative of order ``alpha`` for a
    scalar time series ``values`` sampled at ``times`` using the
    Grünwald‑Letnikov (convolution) formulation with the trapezoidal rule.

    Parameters
    ----------
    values : (T,) array
        Function values f(t_i).
    times : (T,) array
        Strictly increasing time stamps.
    alpha : float, 0 < alpha < 1
        Fractional order.

    Returns
    -------
    dC : (T,) array
        Approximation of D^α f(t_i).
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0, 1) for the Caputo derivative")
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)

    if times.ndim != 1 or values.ndim != 1:
        raise ValueError("times and values must be one‑dimensional")
    if len(times) != len(values):
        raise ValueError("times and values must have the same length")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")

    dt = np.diff(times)
    # First‑order forward differences (approximation of f')
    f_prime = np.concatenate(([0.0], np.diff(values) / dt))

    # Build the lower‑triangular kernel matrix K_{ij} = (t_i - t_j)^{−α} / Γ(1-α)
    # for i >= j, zero otherwise.
    gamma_term = gamma_lanczos(1 - alpha)
    K = np.zeros((len(times), len(times)), dtype=float)
    for i in range(len(times)):
        K[i, : i + 1] = (times[i] - times[: i + 1]) ** (-alpha) / gamma_term

    # Convolution (matrix‑vector product) approximates the integral.
    dC = np.dot(K, f_prime)

    return dC


def _approximate_signature(
    lead_lag_path: np.ndarray,
    times: np.ndarray,
    spline_order: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate the path signature by projecting the lead‑lag path onto a
    B‑spline basis.  Returns the coefficient matrix and the basis itself.

    Parameters
    ----------
    lead_lag_path : (T, D) array
        Lead‑lag transformed path.
    times : (T,) array
        Time stamps associated with ``lead_lag_path``.
    spline_order : int, default 3
        Order of the B‑splines (cubic = 3).

    Returns
    -------
    coeffs : (M, D) array
        B‑spline coefficients (M = len(times) + spline_order - 1).
    basis : (T, M) array
        Evaluated basis functions.
    """
    basis = bspline_basis(times, times, k=spline_order)
    # Least‑squares solve B * c = path  →  c = (BᵀB)⁻¹ Bᵀ path
    BtB = basis.T @ basis
    BtY = basis.T @ lead_lag_path
    coeffs = np.linalg.solve(BtB, BtY)
    return coeffs, basis


def hybrid_step(
    path: np.ndarray,
    alpha: float,
    spline_order: int = 3,
) -> np.ndarray:
    """
    Perform a single hybrid update that fuses a B‑spline approximation of the
    path signature with a Caputo fractional derivative acting on the spline
    coefficients.

    Parameters
    ----------
    path : (T, d) array
        Raw input trajectory.
    alpha : float, 0 < alpha < 1
        Fractional order controlling the memory kernel.
    spline_order : int, default 3
        Order of the B‑splines used for the signature approximation.

    Returns
    -------
    updated : (T, 2*d) array
        The hybrid representation: each time step contains the fractional‑
        derivative of the corresponding spline coefficient vector, reshaped
        back to the lead‑lag dimension.
    """
    # 1. Lead‑lag encoding
    ll_path = lead_lag_transform(path)                     # (2T‑1, 2d)
    # Align time stamps with the original path length (the extra point
    # introduced by lead‑lag does not carry new information, so we drop it).
    times = np.arange(path.shape[0], dtype=float)

    # 2. Approximate the signature via B‑splines
    coeffs, basis = _approximate_signature(ll_path[: len(times)], times, spline_order)

    # 3. Apply the Caputo fractional derivative to each coefficient series
    dcoeffs = np.empty_like(coeffs)
    for dim in range(coeffs.shape[1]):
        dcoeffs[:, dim] = caputo_fractional_derivative(
            coeffs[:, dim], times, alpha
        )

    # 4. Reconstruct the hybrid signal from the differentiated coefficients
    updated = basis @ dcoeffs                                 # (T, 2d)

    return updated


def test_hybrid_step() -> None:
    np.random.seed(0)
    path = np.random.rand(12, 3)          # 12 time steps, 3‑dimensional
    alpha = 0.4
    out = hybrid_step(path, alpha, spline_order=3)
    print("Hybrid output shape:", out.shape)
    print(out)


if __name__ == "__main__":
    test_hybrid_step()