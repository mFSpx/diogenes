# DARWIN HAMMER — match 2875, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_vorono_gliner_zero_shot_ext_m489_s0.py (gen4)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_caputo_fracti_m1185_s3.py (gen3)
# born: 2026-05-29T23:46:32Z

"""Hybrid Voronoi‑GLiNER / Fisher‑Caputo Algorithm
================================================

Parent A: *hybrid_hybrid_hybrid_vorono_gliner_zero_shot_ext_m489_s0.py* – provides
geometric descriptors from a Voronoi point set and a similarity measure that
relies on Euclidean geometry.

Parent B: *hybrid_hybrid_fisher_locali_hybrid_caputo_fracti_m1185_s3.py* – supplies
the Fisher information score derived from a Gaussian beam, the structural‑
similarity index (SSIM) and a Caputo fractional‑order derivative based on the
Lanczos approximation of the Gamma function.

Mathematical Bridge
-------------------
The bridge is built on the observation that the Fisher score needs a *center*
and a *width* parameter for its Gaussian beam.  The Voronoi descriptor returns a
centroid (a 2‑D point) and a scalar dispersion (standard deviation).  By projecting
the centroid onto a 1‑D axis (e.g. its Euclidean norm) we obtain a scalar *center*,
while the dispersion directly serves as the *width*.  The GLiNER label set is
converted to a numeric probe vector by hashing each label and normalising; this
probe plays the role of the angle ``theta`` in the Fisher score.

The resulting Fisher score is then passed through a Caputo fractional derivative,
producing a temporally‑aware similarity that can be combined with a graph‑theoretic
tree‑cost where edge weights fuse Euclidean lengths with the fractional similarity.
The three public functions below illustrate this unified pipeline.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
Point = tuple[float, float]

def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def gliner_voronoi_descriptor(voronoi_points: np.ndarray) -> np.ndarray:
    """
    Compute geometric descriptors from a Voronoi partition.

    Returns an array ``[centroid_x, centroid_y, std_dev]``.
    """
    centroid = np.mean(voronoi_points, axis=0)                     # (2,)
    distances = np.linalg.norm(voronoi_points - centroid, axis=1)
    std_dev = np.std(distances)
    return np.array([centroid[0], centroid[1], std_dev])

# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.cov(x, y, ddof=0)[0, 1]
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))

# Lanczos Gamma approximation (Parent B)
_LANCZOS_G = 7
_LANCZOS_C = np.array([
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

def gamma_lanczos(z: float) -> float:
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

def caputo_derivative(f, alpha: float, t: float) -> float:
    """Caputo fractional derivative of order ``alpha`` for function ``f`` at time ``t``."""
    if not (0 < alpha < 1):
        raise ValueError('alpha must be in (0,1)')
    dt = 0.01
    tau = np.arange(0, t, dt)
    f_tau = f(tau)
    integral = np.trapz(f_tau / (t - tau) ** alpha, tau)
    return integral / gamma_lanczos(1 - alpha)

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def _label_probe_vector(labels: list[str]) -> np.ndarray:
    """
    Convert a list of GLiNER labels into a deterministic numeric probe vector.
    Each label is hashed, reduced modulo 1e6 and normalised.
    """
    if not labels:
        raise ValueError('label list cannot be empty')
    hashes = np.array([abs(hash(lbl)) % 1_000_000 for lbl in labels], dtype=float)
    return hashes / np.max(hashes)

def fractional_fisher_similarity(voronoi_points: np.ndarray,
                                 gliner_labels: list[str],
                                 alpha: float,
                                 t: float) -> float:
    """
    Compute a similarity score that fuses Voronoi geometry, Fisher information,
    and a Caputo fractional derivative.

    Steps
    -----
    1. Obtain the Voronoi descriptor ``[cx, cy, sigma]``.
    2. Collapse the 2‑D centroid to a scalar ``center`` via its Euclidean norm.
    3. Use ``sigma`` as the Gaussian ``width``.
    4. Build a probe ``theta`` from the GLiNER labels (mean of the probe vector).
    5. Evaluate the Fisher score as a function of time ``t``: ``f(τ) = fisher_score(theta, center, sigma) * τ``.
    6. Apply the Caputo derivative of order ``alpha`` to ``f`` at ``t``.
    7. Normalise the result to ``[0,1]``.
    """
    # 1‑3: geometric parameters
    desc = gliner_voronoi_descriptor(voronoi_points)          # [cx, cy, sigma]
    center = math.hypot(desc[0], desc[1])                     # scalar centre
    sigma = desc[2] if desc[2] > 1e-8 else 1e-8               # avoid zero width

    # 4: label‑derived angle
    probe = _label_probe_vector(gliner_labels)
    theta = float(np.mean(probe))

    # 5: time‑dependent function (linear scaling with τ to keep derivative non‑trivial)
    def f(tau: np.ndarray) -> np.ndarray:
        base = fisher_score(theta, center, sigma)
        return base * tau

    # 6: fractional derivative
    deriv = caputo_derivative(f, alpha, t)

    # 7: normalisation (heuristic)
    max_possible = fisher_score(1.0, center, sigma) * t
    return min(max(deriv / (max_possible + 1e-12), 0.0), 1.0)

def hybrid_tree_cost(voronoi_points: np.ndarray,
                     gliner_labels: list[str],
                     nodes: list[int],
                     edges: list[tuple[int, int]],
                     root: int,
                     alpha: float,
                     t: float) -> float:
    """
    Compute a fractional minimum‑cost tree where each edge weight blends Euclidean
    distance with the fractional Fisher similarity derived from the Voronoi‑GLiNER
    bridge.

    The total cost is the sum over edges of:
        w(e) = length(coord_u, coord_v) * (1 - S_frac)
    where ``S_frac`` is the fractional similarity from ``fractional_fisher_similarity``.
    """
    # Build a mapping from node id to a synthetic coordinate.
    # For demonstration we embed nodes uniformly on a circle.
    n = len(nodes)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    coords = {node: (math.cos(a), math.sin(a)) for node, a in zip(nodes, angles)}

    # Global similarity factor (same for all edges in this simple example)
    S_frac = fractional_fisher_similarity(voronoi_points, gliner_labels, alpha, t)

    total = 0.0
    for u, v in edges:
        pu = coords[u]
        pv = coords[v]
        euclid = euclidean_distance(pu, pv)
        total += euclid * (1.0 - S_frac)

    # Add a small root‑bias term proportional to distance from root to all others
    root_coord = coords[root]
    bias = sum(euclidean_distance(root_coord, coords[n]) for n in nodes) * 0.01
    return total + bias

def hybrid_ssim_fisher(voronoi_points: np.ndarray,
                       gliner_labels: list[str],
                       signal: np.ndarray,
                       reference: np.ndarray,
                       alpha: float,
                       t: float) -> float:
    """
    Combine structural similarity (SSIM) with the fractional Fisher similarity.
    The final metric is a weighted geometric mean:
        M = (SSIM ^ w1) * (S_frac ^ w2)
    with weights that depend on the Voronoi dispersion.
    """
    # Base SSIM
    base_ssim = ssim(signal, reference)

    # Fractional Fisher similarity
    S_frac = fractional_fisher_similarity(voronoi_points, gliner_labels, alpha, t)

    # Weighting from dispersion
    sigma = gliner_voronoi_descriptor(voronoi_points)[2]
    w1 = max(0.1, 1.0 - sigma)   # larger dispersion reduces emphasis on SSIM
    w2 = 1.0 - w1

    # Geometric mean in log‑space for numerical stability
    log_M = w1 * math.log(max(base_ssim, 1e-12)) + w2 * math.log(max(S_frac, 1e-12))
    return math.exp(log_M)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic Voronoi points (random cloud)
    rng = np.random.default_rng(42)
    voronoi_pts = rng.random((30, 2)) * 10.0

    # Dummy GLiNER labels
    labels = ["PERSON", "LOCATION", "ORGANIZATION", "DATE"]

    # Graph definition
    nodes = list(range(5))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    root_node = 0

    # Signals for SSIM
    sig = rng.integers(0, 256, size=100).astype(float)
    ref = rng.integers(0, 256, size=100).astype(float)

    # Parameters for fractional calculus
    alpha = 0.6
    t = 5.0

    # Run hybrid functions
    sim = fractional_fisher_similarity(voronoi_pts, labels, alpha, t)
    cost = hybrid_tree_cost(voronoi_pts, labels, nodes, edges, root_node, alpha, t)
    metric = hybrid_ssim_fisher(voronoi_pts, labels, sig, ref, alpha, t)

    print(f"Fractional Fisher similarity: {sim:.4f}")
    print(f"Hybrid tree cost: {cost:.4f}")
    print(f"Hybrid SSIM‑Fisher metric: {metric:.4f}")