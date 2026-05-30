# DARWIN HAMMER — match 4157, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s1.py (gen3)
# born: 2026-05-29T23:54:00Z

import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ----------------------------------------------------------------------
# Parent A – Voronoi helpers
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def voronoi_assign(points: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """
    Assign each point to the index of the nearest site.

    Parameters
    ----------
    points : (N,2) array
        Points to be clustered.
    sites : (K,2) array
        Current Voronoi sites (centroids).

    Returns
    -------
    assignments : (N,) int array
        Index of the nearest site for each point.
    """
    # Efficient broadcasting distance computation
    diff = points[:, None, :] - sites[None, :, :]          # (N, K, 2)
    dists = np.linalg.norm(diff, axis=2)                  # (N, K)
    return np.argmin(dists, axis=1)                       # (N,)

def update_sites(points: np.ndarray, sites: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    """
    Update sites as centroids of assigned points.

    Parameters
    ----------
    points : (N,2) array
        Points to be clustered.
    sites : (K,2) array
        Current Voronoi sites (centroids).
    assignments : (N,) int array
        Index of the nearest site for each point.

    Returns
    -------
    new_sites : (K,2) array of updated centroids
    """
    K = sites.shape[0]
    new_sites = np.array([points[assignments == i].mean(axis=0) for i in range(K)])
    return new_sites

# ----------------------------------------------------------------------
# Parent B – TTT‑Linear helpers
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialise a random weight matrix W with shape (d_out, d_in)."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W: np.ndarray, X: np.ndarray) -> float:
    """
    Reconstruction loss ‖W Xᵀ − Xᵀ‖₂².

    Parameters
    ----------
    W : (d_out, d_in) matrix
    X : (N, d_in) data matrix

    Returns
    -------
    loss : float
    """
    diff = W @ X.T - X.T
    return float(np.sum(diff ** 2))

def ttt_grad(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Gradient of the reconstruction loss with respect to W.

    ∇_W L = 2 (W Xᵀ − Xᵀ) X
    """
    diff = W @ X.T - X.T                     # (d_out, N)
    return 2.0 * diff @ X                    # (d_out, d_in)

def ttt_step(W: np.ndarray, X: np.ndarray, eta: float) -> np.ndarray:
    """One gradient‑descent step on the reconstruction loss."""
    grad = ttt_grad(W, X)
    return W - eta * grad

# ----------------------------------------------------------------------
# Endpoint Circuit Breaker (Parent A)
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        """Reset failure count and close the breaker."""
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        """Increment failure count; open if threshold exceeded."""
        self.failures += 1
        self.last_event_at = now_z()
        if self.failures >= self.failure_threshold:
            self.open = True

    def is_open(self) -> bool:
        """Return True if the breaker is open."""
        return self.open

# ----------------------------------------------------------------------
# SSIM – similarity metric (simplified for 2‑D point clouds)
# ----------------------------------------------------------------------
def ssim_pointcloud(original: np.ndarray, transformed: np.ndarray,
                    dynamic_range: float = 1.0,
                    k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Compute a simplified SSIM between two point clouds.

    The formula is applied component‑wise and then averaged:
        C1 = (k1*R)^2 , C2 = (k2*R)^2
        μx, μy  – means
        σx², σy² – variances
        σxy     – covariance
        SSIM = ((2μxμy + C1)(2σxy + C2)) / ((μx²+μy² + C1)(σx²+σy² + C2))

    Parameters
    ----------
    original, transformed : (N,2) arrays
    dynamic_range : float
        Expected range of the data (default 1.0 for normalised coordinates).

    Returns
    -------
    ssim_value : float in [0,1]
    """
    if original.shape != transformed.shape:
        raise ValueError("point clouds must have the same shape")
    mu_x = original.mean(axis=0)
    mu_y = transformed.mean(axis=0)
    sigma_x2 = ((original - mu_x) ** 2).mean(axis=0)
    sigma_y2 = ((transformed - mu_y) ** 2).mean(axis=0)
    sigma_xy = ((original - mu_x) * (transformed - mu_y)).mean(axis=0)

    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = numerator / denominator
    return float(ssim_map.mean())

# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def hybrid_transform(points: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Apply the learnable linear map W to the point cloud.

    Returns transformed points of shape (N,2).
    """
    return (W @ points.T).T  # (N,2)

def hybrid_update(points: np.ndarray,
                  sites: np.ndarray,
                  W: np.ndarray,
                  eta: float,
                  breaker: EndpointCircuitBreaker) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    One hybrid iteration:
      1. Transform points with current W.
      2. Compute Voronoi assignment in transformed space.
      3. Update sites as centroids of assigned points (in original space).
      4. Perform a gradient‑descent step on W using the original points.
      5. Use the circuit breaker to decide whether to continue.

    Returns
    -------
    new_sites : (K,2) array of updated centroids
    new_W    : updated weight matrix
    loss     : reconstruction loss after the step
    """
    # 1. Transform
    transformed = hybrid_transform(points, W)

    # 2. Voronoi assignment (using transformed coordinates)
    assignments = voronoi_assign(transformed, sites)

    # 3. Update sites
    new_sites = update_sites(points, sites, assignments)

    # 4. Update W
    loss = ttt_loss(W, points)
    new_W = ttt_step(W, points, eta)

    # 5. Check circuit breaker
    if loss < ttt_loss(W, points):
        breaker.record_success()
    else:
        breaker.record_failure()

    return new_sites, new_W, loss

def hybrid_train(points: np.ndarray,
                 sites: np.ndarray,
                 W: np.ndarray,
                 eta: float = 0.01,
                 failure_threshold: int = 3,
                 max_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Train the hybrid model.

    Parameters
    ----------
    points : (N,2) array
        Points to be clustered.
    sites : (K,2) array
        Initial Voronoi sites (centroids).
    W : (2,2) matrix
        Initial weight matrix.
    eta : float
        Learning rate for gradient descent.
    failure_threshold : int
        Threshold for circuit breaker.
    max_iterations : int
        Maximum number of iterations.

    Returns
    -------
    sites : (K,2) array of final centroids
    W    : final weight matrix
    loss : final reconstruction loss
    """
    breaker = EndpointCircuitBreaker(failure_threshold)
    for _ in range(max_iterations):
        if breaker.is_open():
            break
        sites, W, loss = hybrid_update(points, sites, W, eta, breaker)
    return sites, W, loss