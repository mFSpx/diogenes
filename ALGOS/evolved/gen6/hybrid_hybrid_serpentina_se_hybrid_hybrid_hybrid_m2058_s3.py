# DARWIN HAMMER — match 2058, survivor 3
# gen: 6
# parent_a: hybrid_serpentina_self_righ_hybrid_hybrid_fisher_m29_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_semant_hybrid_hybrid_krampu_m787_s1.py (gen5)
# born: 2026-05-29T23:40:43Z

import math
import numpy as np
from dataclasses import dataclass
from random import random
from typing import List, Tuple


@dataclass(frozen=True)
class Morphology:
    """Geometric description of the self‑righting body."""
    length: float
    width: float
    height: float
    mass: float


def _check_positive(*values: float) -> None:
    for v in values:
        if v <= 0:
            raise ValueError("all geometric parameters must be positive")


def volume(m: Morphology) -> float:
    """Volume of the rectangular prism approximating the body."""
    _check_positive(m.length, m.width, m.height)
    return m.length * m.width * m.height


def surface_area(m: Morphology) -> float:
    """Surface area of the rectangular prism."""
    _check_positive(m.length, m.width, m.height)
    l, w, h = m.length, m.width, m.height
    return 2 * (l * w + w * h + h * l)


def sphericity_index(m: Morphology) -> float:
    """
    Classical sphericity: ratio of the surface area of a sphere
    with the same volume to the actual surface area.
    """
    v = volume(m)
    a = surface_area(m)
    sphere_surface = math.pi ** (1.0 / 3.0) * (6 * v) ** (2.0 / 3.0)
    return sphere_surface / a


def flatness_index(m: Morphology) -> float:
    """A simple flatness measure (larger → flatter)."""
    _check_positive(m.length, m.width, m.height)
    return (m.length + m.width) / (2.0 * m.height)


def gaussian_beam(theta: np.ndarray, center: float, width: float) -> np.ndarray:
    """Vectorised Gaussian beam."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return np.exp(-0.5 * z * z)


def fisher_score(
    theta: np.ndarray,
    center: float,
    width: float,
    eps: float = 1e-12,
    sphericity: float = 1.0,
) -> np.ndarray:
    """
    Fisher information for a Gaussian beam, scaled by the sphericity.
    Implements I(θ)= (∂_θ ln p(θ))² p(θ) with a small epsilon to avoid division by zero.
    """
    intensity = np.maximum(gaussian_beam(theta, center, width), eps)
    derivative = -(theta - center) / (width * width) * intensity
    return sphericity * (derivative ** 2) / intensity


def righting_time_index(
    m: Morphology,
    theta: np.ndarray,
    center: float = 0.0,
    width: float = 1.0,
    b: float = 1.0 / 3.0,
    k: float = 0.35,
    neck_lever: float = 1.0,
) -> np.ndarray:
    """
    Returns a time‑indexed right‑ing index.
    The index varies with the orientation angle θ.
    """
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m)
    sph = sphericity_index(m)
    fs = fisher_score(theta, center, width, sphericity=sph)
    base = (m.mass ** b) * np.exp(k * fi) / neck_lever
    return base * fs


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Scalar priority derived from a single‑point righting index."""
    theta = np.array([0.0])  # neutral orientation
    rti = righting_time_index(m, theta)[0]
    return max(0.0, min(1.0, rti / max_index))


def caputo_derivative(f: np.ndarray, alpha: float) -> np.ndarray:
    """
    Simple Caputo fractional derivative (uniform time step = 1).
    Uses the definition D^α f(t_i) = 1/Γ(1-α) Σ_{j=0}^{i-1} (f_i - f_j) / (i-j)^{α+1}.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    n = f.shape[0]
    coeff = 1.0 / math.gamma(1.0 - alpha)
    out = np.zeros_like(f, dtype=float)
    for i in range(1, n):
        diffs = f[i] - f[:i]
        weights = (i - np.arange(i)) ** (-(alpha + 1))
        out[i] = coeff * np.sum(diffs * weights)
    return out


def _morphology_corners(m: Morphology) -> np.ndarray:
    """Coordinates of the eight corners of the bounding box."""
    l, w, h = m.length / 2, m.width / 2, m.height / 2
    corners = np.array(
        [
            [-l, -w, -h],
            [-l, -w, h],
            [-l, w, -h],
            [-l, w, h],
            [l, -w, -h],
            [l, -w, h],
            [l, w, -h],
            [l, w, h],
        ]
    )
    return corners


def _edge_list(corners: np.ndarray) -> List[Tuple[int, int]]:
    """All edges of the rectangular prism (12 edges)."""
    return [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]


def _ollivier_ricci_curvature(m: Morphology) -> float:
    """
    Very coarse Ollivier‑Ricci curvature estimate:
    average over edges of (1 - W_1(μ_i, μ_j) / d_ij),
    where μ_i is the uniform distribution over the neighbours of node i.
    For a regular rectangular prism the neighbourhoods are identical,
    giving a constant curvature that depends only on geometry.
    """
    corners = _morphology_corners(m)
    edges = _edge_list(corners)
    curvatures = []
    for i, j in edges:
        d_ij = np.linalg.norm(corners[i] - corners[j])
        # neighbours of i (excluding j) and of j (excluding i)
        neigh_i = [corners[k] for k in range(8) if k != i and np.linalg.norm(corners[i] - corners[k]) < d_ij + 1e-9]
        neigh_j = [corners[k] for k in range(8) if k != j and np.linalg.norm(corners[j] - corners[k]) < d_ij + 1e-9]
        # uniform measures
        mu_i = np.mean(neigh_i, axis=0) if neigh_i else corners[i]
        mu_j = np.mean(neigh_j, axis=0) if neigh_j else corners[j]
        w1 = np.linalg.norm(mu_i - mu_j)  # 1‑Wasserstein distance for uniform measures on a single point
        curvatures.append(1.0 - w1 / d_ij)
    return float(np.mean(curvatures))


def hybrid_operation(
    m: Morphology,
    theta_center: float,
    theta_width: float,
    alpha: float = 0.5,
    neck_lever: float = 1.0,
    steps: int = 20,
) -> float:
    """
    Deeply integrated hybrid metric:
    1. Build a time series of right‑ing indices over a sweep of orientation angles.
    2. Apply a Caputo fractional derivative to capture memory effects.
    3. Modulate the result by the Ollivier‑Ricci curvature of the morphology graph.
    """
    # 1. orientation sweep
    thetas = np.linspace(theta_center - math.pi, theta_center + math.pi, steps)
    rti_series = righting_time_index(
        m, thetas, center=theta_center, width=theta_width, neck_lever=neck_lever
    )
    # 2. fractional derivative
    frac_deriv = caputo_derivative(rti_series, alpha)
    # 3. curvature weighting
    curvature = _ollivier_ricci_curvature(m)
    return float(np.mean(frac_deriv) * (1.0 + curvature))


if __name__ == "__main__":
    # Example usage
    m = Morphology(length=1.0, width=2.0, height=3.0, mass=10.0)
    theta_center = random() * math.pi
    theta_width = random() * math.pi * 0.5 + 0.1
    print(hybrid_operation(m, theta_center, theta_width))