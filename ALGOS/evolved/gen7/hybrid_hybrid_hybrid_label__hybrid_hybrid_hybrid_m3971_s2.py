# DARWIN HAMMER — match 3971, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_label_foundry_hybrid_hybrid_hybrid_m2219_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m2309_s0.py (gen4)
# born: 2026-05-29T23:52:51Z

"""Hybrid algorithm merging:

- **Parent A**: `hybrid_hybrid_label_foundry_hybrid_hybrid_hybrid_m2219_s0.py`
  Uses a multivector representation of Fisher information to update the state‑transition
  matrix **A** in a linear state‑space model and then propagates state and output via
  state‑space duality.

- **Parent B**: `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m2309_s0.py`
  Computes a scalar Fisher information score for a Gaussian‑beam intensity and feeds it
  into a geometric‑algebra (multivector) routine.

**Mathematical bridge** – Both parents treat *Fisher information* as the core scalar that
modulates a *multivector* object.  In this hybrid we (1) compute the scalar Fisher
information **I(θ,σ)** from a Gaussian model, (2) embed **I** into a simple multivector
`M = I*e0 + I*e12` (scalar + bivector), and (3) use the scalar part of **M** to adapt the
state‑transition matrix **A** of a discrete‑time linear system:


A_{k+1} = A_k + α·I·I_n
x_{k+1} = A_{k+1} x_k + B u_k
y_k     = C x_k


The geometric‑algebra side also provides deterministic Voronoi assignment utilities
(distance, nearest, assign) that can be used for contextual bandit‑like updates or
spatial clustering of measurement points.

The module below implements this fused workflow with three public functions that
demonstrate the hybrid operation.
"""

import math
import random
import sys
import pathlib
import numpy as np
from typing import List, Tuple, Dict, Callable, Sequence

# ----------------------------------------------------------------------
# Basic geometric utilities (from Parent B)
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the nearest seed (ties broken by lower index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Deterministic Voronoi assignment of points to the nearest seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Minimal geometric‑algebra (multivector) implementation
# ----------------------------------------------------------------------
class Multivector:
    """
    Very small GA engine supporting scalar (grade‑0) and bivector (grade‑2) blades
    in 2‑D Euclidean space: e0 (scalar), e12 (bivector).  Internally a dict maps
    blade identifiers to coefficients.
    """
    def __init__(self, scalar: float = 0.0, bivector: float = 0.0):
        self.blades: Dict[str, float] = {'e0': scalar, 'e12': bivector}

    @property
    def scalar(self) -> float:
        return self.blades.get('e0', 0.0)

    @property
    def bivector(self) -> float:
        return self.blades.get('e12', 0.0)

    def __add__(self, other: 'Multivector') -> 'Multivector':
        return Multivector(
            scalar=self.scalar + other.scalar,
            bivector=self.bivector + other.bivector
        )

    def __mul__(self, other: 'Multivector') -> 'Multivector':
        """
        Geometric product for the restricted basis:
        (a0 + a12 e12)(b0 + b12 e12) = a0 b0 + a0 b12 e12 + a12 b0 e12 + a12 b12 (e12)^2
        In 2‑D Euclidean GA, (e12)^2 = -1.
        """
        a0, a12 = self.scalar, self.bivector
        b0, b12 = other.scalar, other.bivector
        scalar_part = a0 * b0 - a12 * b12
        biv_part = a0 * b12 + a12 * b0
        return Multivector(scalar=scalar_part, bivector=biv_part)

    def __repr__(self) -> str:
        return f"Multivector(scalar={self.scalar:.4g}, bivector={self.bivector:.4g})"

# ----------------------------------------------------------------------
# Fisher information for a 1‑D Gaussian (Parent B core)
# ----------------------------------------------------------------------
def fisher_information(theta: float, sigma: float) -> float:
    """
    Fisher information for estimating the mean `theta` of a Gaussian with known
    standard deviation `sigma`.  I(θ) = 1 / sigma².
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return 1.0 / (sigma ** 2)

# ----------------------------------------------------------------------
# Bridge: embed Fisher info into a multivector (Parent A side)
# ----------------------------------------------------------------------
def fisher_to_multivector(I: float) -> Multivector:
    """
    Map scalar Fisher information `I` to a multivector.
    The scalar part carries the raw information; the bivector part mirrors it
    to illustrate geometric coupling.
    """
    return Multivector(scalar=I, bivector=I)

# ----------------------------------------------------------------------
# State‑space update using the multivector (Parent A side)
# ----------------------------------------------------------------------
def update_state_transition(A: np.ndarray, mv: Multivector, alpha: float = 0.1) -> np.ndarray:
    """
    Adapt the state‑transition matrix `A` using the scalar part of the multivector.
    A_new = A + α·I·I_n, where I = mv.scalar and I_n is the identity of matching size.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    I = mv.scalar
    return A + alpha * I * np.eye(A.shape[0])

def state_space_step(x: np.ndarray,
                    A: np.ndarray,
                    B: np.ndarray,
                    u: np.ndarray,
                    C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one discrete‑time step of the linear system:
        x_{k+1} = A x_k + B u_k
        y_k     = C x_k
    Returns (x_next, y).
    """
    x_next = A @ x + B @ u
    y = C @ x
    return x_next, y

# ----------------------------------------------------------------------
# High‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_step(state: np.ndarray,
                A: np.ndarray,
                B: np.ndarray,
                C: np.ndarray,
                u: np.ndarray,
                theta: float,
                sigma: float,
                alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute one hybrid iteration:
      1. Compute Fisher information I(θ,σ).
      2. Convert I to a multivector M.
      3. Update the state‑transition matrix A ← A + α·I·I_n.
      4. Propagate the state and produce the output.
    Returns (x_next, y, A_new).
    """
    I = fisher_information(theta, sigma)
    mv = fisher_to_multivector(I)
    A_new = update_state_transition(A, mv, alpha=alpha)
    x_next, y = state_space_step(state, A_new, B, u, C)
    return x_next, y, A_new

# ----------------------------------------------------------------------
# Example utility: cluster measurement points and adjust control input
# ----------------------------------------------------------------------
def cluster_and_control(points: List[Point],
                        seeds: List[Point],
                        base_u: np.ndarray,
                        scale: float = 0.5) -> np.ndarray:
    """
    Assign points to Voronoi cells, count points per cell, and modulate the control
    vector `base_u` proportionally to the occupancy of the first cell.
    This demonstrates using the geometric utilities alongside the hybrid core.
    """
    regions = assign(points, seeds)
    # Simple heuristic: more points in region 0 → larger control magnitude
    weight = 1.0 + scale * len(regions.get(0, []))
    return base_u * weight

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # System dimensions
    n = 3  # state dimension
    m = 2  # input dimension
    p = 2  # output dimension

    # Random but reproducible seed
    np.random.seed(0)

    # Initialise linear system matrices
    A = np.eye(n) * 0.9
    B = np.random.randn(n, m)
    C = np.random.randn(p, n)

    # Initial state
    x = np.random.randn(n, 1)

    # Control input (constant)
    u = np.array([[0.1], [0.2]])

    # Measurement parameters (theta, sigma) that evolve over time
    theta = 0.0
    sigma = 0.5

    # Run a few hybrid steps
    for step in range(5):
        theta += 0.1  # pretend the mean drifts
        sigma = max(0.1, sigma * (0.95 + 0.1 * random.random()))  # slowly vary sigma

        x, y, A = hybrid_step(state=x,
                              A=A,
                              B=B,
                              C=C,
                              u=u,
                              theta=theta,
                              sigma=sigma,
                              alpha=0.05)

        # Demonstrate auxiliary geometric utility
        points = [(random.random(), random.random()) for _ in range(20)]
        seeds = [(0.25, 0.25), (0.75, 0.75)]
        u = cluster_and_control(points, seeds, base_u=u, scale=0.2)

        print(f"Step {step+1}:")
        print(f"  theta={theta:.3f}, sigma={sigma:.3f}, Fisher I={fisher_information(theta, sigma):.4f}")
        print(f"  state x = {x.ravel()}")
        print(f"  output y = {y.ravel()}")
        print(f"  updated A diagonal = {np.diag(A)}\n")