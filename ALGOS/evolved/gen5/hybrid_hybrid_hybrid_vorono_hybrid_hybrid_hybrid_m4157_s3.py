# DARWIN HAMMER — match 4157, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s1.py (gen3)
# born: 2026-05-29T23:54:00Z

"""Hybrid Voronoi‑TTT Fusion Module

Parents:
- hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s3 (Voronoi + EndpointCircuitBreaker)
- hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s1 (TTT‑Linear weight matrix, gradient descent, SSIM)

Mathematical Bridge:
Both parents manipulate a set of points in ℝ².  The TTT‑Linear component supplies a
learnable linear map **W** ∈ ℝ^{2×2} that is updated by gradient descent to minimise the
reconstruction loss  L(W)=‖W Xᵀ − Xᵀ‖₂².  The Voronoi component needs a distance metric;
by applying **W** to the raw points first, the Voronoi diagram is constructed in the
transformed space.  Consequently the optimisation of **W** directly influences the
Voronoi partition geometry.  The EndpointCircuitBreaker monitors the loss
improvement: if the loss fails to decrease for *failure_threshold* consecutive
updates the optimisation halts, preventing wasted computation.

The module therefore fuses:
1. Linear‑transform update (TTT‑Linear)
2. Voronoi assignment in the transformed space
3. Convergence control via EndpointCircuitBreaker
4. Quality assessment with a simplified SSIM between original and transformed
   point clouds.

The three core functions below demonstrate this hybrid operation.
"""

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

    # 3. Update sites as centroids of the *original* points belonging to each region
    new_sites = np.empty_like(sites)
    for i in range(sites.shape[0]):
        mask = assignments == i
        if np.any(mask):
            new_sites[i] = points[mask].mean(axis=0)
        else:
            # If a site loses all points, re‑initialise it randomly
            new_sites[i] = points[random.randint(0, points.shape[0] - 1)]

    # 4. Gradient descent on W
    loss_before = ttt_loss(W, points)
    new_W = ttt_step(W, points, eta)
    loss_after = ttt_loss(new_W, points)

    # 5. Circuit breaker logic (monitor loss decrease)
    if loss_after < loss_before:
        breaker.record_success()
    else:
        breaker.record_failure()

    return new_sites, new_W, loss_after

def hybrid_fuse(points: np.ndarray,
                n_sites: int = 4,
                eta: float = 1e-4,
                max_iter: int = 200,
                failure_threshold: int = 5) -> Dict[str, Any]:
    """
    Run the full hybrid optimisation loop.

    Parameters
    ----------
    points : (N,2) array of input coordinates.
    n_sites : number of Voronoi sites (clusters).
    eta : learning rate for the TTT‑Linear update.
    max_iter : maximum number of iterations.
    failure_threshold : breaker threshold.

    Returns
    -------
    result : dict with final sites, weight matrix, loss history and SSIM.
    """
    # Initialise sites by random sampling
    rng = np.random.default_rng(0)
    indices = rng.choice(points.shape[0], size=n_sites, replace=False)
    sites = points[indices].copy()

    # Initialise linear map
    W = init_ttt(d_in=2, d_out=2, scale=0.01, seed=1)

    breaker = EndpointCircuitBreaker(failure_threshold=failure_threshold)
    loss_history: List[float] = []

    for it in range(max_iter):
        sites, W, loss = hybrid_update(points, sites, W, eta, breaker)
        loss_history.append(loss)

        if breaker.is_open():
            print(f"[{now_z()}] Circuit breaker opened at iteration {it}. Stopping.")
            break

    # Final similarity assessment
    transformed = hybrid_transform(points, W)
    similarity = ssim_pointcloud(points, transformed)

    return {
        "sites": sites,
        "W": W,
        "loss_history": loss_history,
        "ssim": similarity,
        "iterations": len(loss_history),
        "breaker_open": breaker.is_open(),
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a synthetic 2‑D point cloud (e.g., mixture of Gaussians)
    rng = np.random.default_rng(42)
    n_points = 500
    # Two clusters
    cluster1 = rng.normal(loc=[0.2, 0.2], scale=0.05, size=(n_points // 2, 2))
    cluster2 = rng.normal(loc=[0.8, 0.8], scale=0.07, size=(n_points // 2, 2))
    points = np.vstack([cluster1, cluster2])

    result = hybrid_fuse(points,
                         n_sites=3,
                         eta=5e-5,
                         max_iter=300,
                         failure_threshold=7)

    print(f"Final loss: {result['loss_history'][-1]:.6f}")
    print(f"SSIM between original and transformed points: {result['ssim']:.4f}")
    print(f"Number of iterations performed: {result['iterations']}")
    print(f"Circuit breaker opened? {result['breaker_open']}")
    print("Final Voronoi sites (centroids):")
    for idx, site in enumerate(result["sites"]):
        print(f"  Site {idx}: {site}")