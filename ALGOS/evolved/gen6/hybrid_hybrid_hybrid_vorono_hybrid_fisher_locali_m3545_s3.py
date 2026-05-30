# DARWIN HAMMER — match 3545, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m1697_s1.py (gen5)
# parent_b: hybrid_fisher_localization_hybrid_ternary_route_m40_s3.py (gen2)
# born: 2026-05-29T23:50:38Z

"""Hybrid Voronoi‑Fisher‑SSIM Algorithm
Parents:
- hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m1697_s1.py (Voronoi partition, sphericity, entropy)
- hybrid_fisher_localization_hybrid_ternary_route_m40_s3.py (Gaussian beam, Fisher information, SSIM)

Mathematical bridge:
Each Voronoi cell is treated as a “directional sample”.  The polar angle 𝜃 of the cell
centroid (relative to a user‑defined centre) is fed to the Gaussian‑beam Fisher score
F(𝜃).  The textual label attached to the cell is compared with a reference label by SSIM,
S.  The geometric quality of the cell is expressed by its sphericity σ and by the global
Shannon entropy H of the cell‑size distribution.  The final hybrid quality metric for a
cell i is

    Q_i = F(𝜃_i) · S_i · σ_i · (1 + H)

Thus the geometric topology (Voronoi) is modulated by the statistical similarity
(Fisher × SSIM) and by decision‑hygiene quantities (sphericity, entropy)."""

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Sequence

import numpy as np

Point = Tuple[float, float]
GridPoint = Tuple[int, int]

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ.

    F(θ) = (∂I/∂θ)² / I  where I = Gaussian beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: Sequence[float], y: Sequence[float],
         dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index between two 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


def text_to_signal(text: str) -> np.ndarray:
    """Convert a Unicode string to a numeric signal (code‑point float array)."""
    return np.array([float(ord(ch)) for ch in text])


# ----------------------------------------------------------------------
# Parent‑A building blocks (light‑weight Voronoi implementation)
# ----------------------------------------------------------------------
def generate_grid(bounds: Tuple[float, float, float, float],
                  resolution: Tuple[int, int]) -> np.ndarray:
    """Create a regular grid of points inside *bounds*.

    bounds = (xmin, xmax, ymin, ymax)
    resolution = (nx, ny) number of points along each axis.
    Returns an array of shape (nx*ny, 2).
    """
    xmin, xmax, ymin, ymax = bounds
    nx, ny = resolution
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xv, yv = np.meshgrid(xs, ys, indexing='xy')
    return np.column_stack([xv.ravel(), yv.ravel()])


def assign_voronoi_cells(grid: np.ndarray, seeds: np.ndarray) -> Dict[int, List[int]]:
    """Assign each grid point to the nearest seed index.

    Returns a dict mapping seed index → list of grid‑point indices.
    """
    # Compute squared Euclidean distance matrix (grid × seeds)
    diff = grid[:, None, :] - seeds[None, :, :]          # shape (Ngrid, Nseed, 2)
    d2 = np.einsum('ijk,ijk->ij', diff, diff)           # shape (Ngrid, Nseed)
    nearest = np.argmin(d2, axis=1)                     # shape (Ngrid,)
    cells: Dict[int, List[int]] = {i: [] for i in range(seeds.shape[0])}
    for idx, seed_idx in enumerate(nearest):
        cells[seed_idx].append(idx)
    return cells


def region_properties(grid: np.ndarray, indices: List[int],
                     grid_shape: Tuple[int, int]) -> Dict[str, float]:
    """Compute area, perimeter (approx), and sphericity for a region.

    *area*   = number of grid points.
    *perimeter* ≈ count of points that have a neighbour belonging to another region.
    *sphericity* = 4π·area / perimeter² (≈1 for a perfect circle).
    """
    area = float(len(indices))
    if area == 0:
        return {"area": 0.0, "perimeter": 0.0, "sphericity": 0.0}

    # reshape linear indices to 2‑D coordinates
    nx, ny = grid_shape
    coords = np.unravel_index(indices, (nx, ny))
    occupied = set(zip(coords[0], coords[1]))

    perimeter = 0
    for x, y in occupied:
        # 4‑neighbourhood
        neighbours = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        if any((nxn, nyn) not in occupied for nxn, nyn in neighbours
               if 0 <= nxn < nx and 0 <= nyn < ny):
            perimeter += 1

    perimeter = float(perimeter) if perimeter > 0 else 1.0
    sphericity = (4 * math.pi * area) / (perimeter * perimeter)
    return {"area": area, "perimeter": perimeter, "sphericity": sphericity}


def shannon_entropy(sizes: Sequence[float]) -> float:
    """Weight‑scaled Shannon entropy of a discrete size distribution."""
    total = sum(sizes)
    if total == 0:
        return 0.0
    probs = np.array(sizes) / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_cell_quality(cell_idx: int,
                        cell_points: List[int],
                        grid: np.ndarray,
                        grid_shape: Tuple[int, int],
                        center: Point,
                        width: float,
                        packet_text: str,
                        reference_text: str,
                        global_entropy: float) -> float:
    """Compute the hybrid quality Q_i for a single Voronoi cell.

    Q_i = Fisher(θ_i) × SSIM(text_i, reference) × σ_i × (1 + H)

    where
        θ_i = angle of the cell centroid relative to *center*,
        σ_i = sphericity of the cell,
        H   = global Shannon entropy of all cell areas.
    """
    # centroid of the cell (in continuous coordinates)
    pts = grid[cell_points]                     # shape (N,2)
    centroid = pts.mean(axis=0)

    # polar angle θ_i (range [0, 2π))
    dx = centroid[0] - center[0]
    dy = centroid[1] - center[1]
    theta = math.atan2(dy, dx) % (2 * math.pi)

    # Fisher information for this angle
    f = fisher_score(theta, center=0.0, width=width)   # centre fixed at 0 for simplicity

    # SSIM between cell's label and reference label
    s = ssim(text_to_signal(packet_text), text_to_signal(reference_text))

    # geometric quality
    props = region_properties(grid, cell_points, grid_shape)
    sigma = props["sphericity"]

    return f * s * sigma * (1.0 + global_entropy)


def best_hybrid_cell(cells: Dict[int, List[int]],
                     grid: np.ndarray,
                     grid_shape: Tuple[int, int],
                     center: Point,
                     width: float,
                     packet_texts: Dict[int, str],
                     reference_text: str) -> Tuple[int, float]:
    """Select the cell index that maximises the hybrid quality metric.

    Returns (cell_index, quality_score).
    """
    # compute global entropy once
    sizes = [len(idx_list) for idx_list in cells.values()]
    H = shannon_entropy(sizes)

    best_idx = -1
    best_score = -math.inf
    for idx, pts in cells.items():
        txt = packet_texts.get(idx, "")
        q = hybrid_cell_quality(idx, pts, grid, grid_shape,
                                center=center, width=width,
                                packet_text=txt,
                                reference_text=reference_text,
                                global_entropy=H)
        if q > best_score:
            best_score = q
            best_idx = idx
    return best_idx, best_score


def hybrid_metric_over_grid(seeds: np.ndarray,
                            bounds: Tuple[float, float, float, float],
                            resolution: Tuple[int, int],
                            center: Point,
                            width: float,
                            packet_texts: Dict[int, str],
                            reference_text: str) -> Dict[int, float]:
    """Compute hybrid quality for every Voronoi cell and return a mapping."""
    grid = generate_grid(bounds, resolution)                     # (N,2)
    cells = assign_voronoi_cells(grid, seeds)                    # dict seed→indices
    sizes = [len(v) for v in cells.values()]
    H = shannon_entropy(sizes)

    results: Dict[int, float] = {}
    for idx, pts in cells.items():
        txt = packet_texts.get(idx, "")
        q = hybrid_cell_quality(idx, pts, grid,
                                grid_shape=resolution,
                                center=center,
                                width=width,
                                packet_text=txt,
                                reference_text=reference_text,
                                global_entropy=H)
        results[idx] = q
    return results


# ----------------------------------------------------------------------
# Simple circuit‑breaker (from Parent‑A) used to guard the main routine
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """A minimal failure‑count circuit breaker."""
    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        return not self.open


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # configuration
    NUM_SEEDS = 7
    BOUNDS = (0.0, 10.0, 0.0, 10.0)          # xmin, xmax, ymin, ymax
    RESOLUTION = (200, 200)                # grid points per axis
    CENTER_POINT = (5.0, 5.0)               # geometric centre for angle measurement
    WIDTH = 1.0                            # Gaussian beam width
    REFERENCE_TEXT = "reference"

    # random seeds inside the bounds
    rng = np.random.default_rng(42)
    seeds = rng.uniform(low=[BOUNDS[0], BOUNDS[2]],
                        high=[BOUNDS[1], BOUNDS[3]],
                        size=(NUM_SEEDS, 2))

    # assign a random short text to each seed
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    packet_texts = {
        i: "".join(rng.choice(list(alphabet), size=5))
        for i in range(NUM_SEEDS)
    }

    breaker = EndpointCircuitBreaker(failure_threshold=2)

    try:
        # compute hybrid qualities for all cells
        qualities = hybrid_metric_over_grid(seeds, BOUNDS, RESOLUTION,
                                            CENTER_POINT, WIDTH,
                                            packet_texts, REFERENCE_TEXT)

        # find best cell
        best_idx, best_score = best_hybrid_cell(
            assign_voronoi_cells(generate_grid(BOUNDS, RESOLUTION), seeds),
            generate_grid(BOUNDS, RESOLUTION),
            RESOLUTION,
            CENTER_POINT,
            WIDTH,
            packet_texts,
            REFERENCE_TEXT)

        print(f"Best cell index: {best_idx} with hybrid quality {best_score:.6f}")
        print("All cell qualities:")
        for idx, q in qualities.items():
            print(f"  Cell {idx}: {q:.6f}")

        breaker.record_success()
    except Exception as e:
        breaker.record_failure()
        print(f"Error during execution: {e}", file=sys.stderr)
        sys.exit(1)