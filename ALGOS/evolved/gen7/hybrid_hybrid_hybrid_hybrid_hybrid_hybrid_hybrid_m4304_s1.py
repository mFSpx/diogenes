# DARWIN HAMMER — match 4304, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m2361_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fracti_m506_s0.py (gen4)
# born: 2026-05-29T23:54:56Z

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Iterable, Set, Any

import numpy as np

# ----------------------------------------------------------------------
# Core geometric utilities
# ----------------------------------------------------------------------
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]

def distance(p1: Point2D, p2: Point2D) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest(point: Point2D, seeds: List[Point2D]) -> int:
    """Index of the seed closest to *point* (ties broken by lowest index)."""
    dists = [distance(point, s) for s in seeds]
    return int(min(range(len(dists)), key=lambda i: (dists[i], i)))

def hybrid_assign(points: List[Point2D], seeds: List[Point2D]) -> List[int]:
    """Assign each point to the nearest seed, returning a list of cell indices."""
    return [nearest(p, seeds) for p in points]

# ----------------------------------------------------------------------
# Morphology
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    sphericity: float   # 0 (flat) → 1 (perfect sphere)
    flatness: float     # 0 (spherical) → 1 (flat)

def _bbox_3d(points: List[Point2D]) -> Tuple[float, float, float]:
    """Simple extrusion of a 2‑D cloud to a 3‑D bounding box (z=0)."""
    xs, ys = zip(*points)
    length = max(xs) - min(xs)
    width = max(ys) - min(ys)
    height = 0.0  # planar cloud; can be enriched later
    return length, width, height

def region_morphology(region_points: List[Point2D]) -> Morphology:
    """Compute bounding‑box based morphology and derived shape indices."""
    length, width, height = _bbox_3d(region_points)
    # Avoid division by zero
    vol = max(length * width * height, 1e-9)
    # Sphericity ≈ (π^(1/3) * (6·vol)^(2/3)) / surface_area ; we approximate surface_area
    surface = 2 * (length * width + width * height + height * length)
    sphericity = (math.pi ** (1/3) * (6 * vol) ** (2/3)) / max(surface, 1e-9)
    sphericity = max(0.0, min(1.0, sphericity))
    flatness = 1.0 - sphericity
    return Morphology(length, width, height, sphericity, flatness)

# ----------------------------------------------------------------------
# MinHash signature
# ----------------------------------------------------------------------
def _hash_token(token: str, seed: int) -> int:
    h = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")

def minhash_signature(tokens: Iterable[str], num_perm: int = 128) -> List[int]:
    """Return a MinHash sketch of *tokens* as a list of length *num_perm*."""
    signature = [2**64 - 1] * num_perm
    for token in tokens:
        for i in range(num_perm):
            hv = _hash_token(token, i)
            if hv < signature[i]:
                signature[i] = hv
    return signature

# ----------------------------------------------------------------------
# Certainty flag
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CertaintyFlag:
    confidence_bps: int          # basis points (0‑10000)
    timestamp: datetime
    comment: str

def _confidence_from_similarity(sim: float) -> int:
    """Map similarity ∈[0,1] to basis‑points integer."""
    return int(max(0, min(1, sim)) * 10_000)

def region_certainty(tokens: Iterable[str]) -> CertaintyFlag:
    """Generate a CertaintyFlag from token similarity (self‑Jaccard proxy)."""
    # Self‑similarity of a set with itself is 1.0; we simulate a noisy estimate.
    sim = 1.0 - random.random() * 0.05  # up to 5 % degradation
    confidence = _confidence_from_similarity(sim)
    return CertaintyFlag(
        confidence_bps=confidence,
        timestamp=datetime.now(timezone.utc),
        comment=f"Estimated similarity {sim:.3f}"
    )

# ----------------------------------------------------------------------
# Hyper‑vector utilities
# ----------------------------------------------------------------------
def random_hv(d: int = 10000, kind: str = "complex", seed: Any = None) -> np.ndarray:
    """Generate a random hypervector of dimension *d*."""
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)               # unit‑magnitude complex numbers
    elif kind == "bipolar":
        return rng.choice([-1, 1], size=d)
    elif kind == "real":
        return rng.standard_normal(d)
    else:
        raise ValueError(f"Unsupported kind={kind}")

def fractional_binding(v: np.ndarray, phase: np.ndarray, alpha: float) -> np.ndarray:
    """
    Fractional power binding:
        b_i = v_i * (exp(j·phase_i))**alpha
    where *phase* is in radians (0‑2π) derived from MinHash.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0,1]")
    phase_complex = np.exp(1j * phase)               # unit‑circle
    bound = v * (phase_complex ** alpha)
    return bound

# ----------------------------------------------------------------------
# Differential‑privacy utilities
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Probability that a record can be re‑identified."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def dp_laplace(value: float, epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """Laplace mechanism: value + Laplace(0, sensitivity/epsilon)."""
    scale = sensitivity / max(epsilon, 1e-9)
    noise = np.random.laplace(0.0, scale)
    return value + noise

# ----------------------------------------------------------------------
# Unified hybrid operation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class HybridResult:
    cell_id: int
    hypervector: np.ndarray
    certainty: CertaintyFlag
    privacy_risk: float
    morphology: Morphology

def hybrid_cell_representation(
    cell_id: int,
    region_points: List[Point2D],
    token_map: Dict[Point2D, List[str]],
    hv_dim: int = 8192,
    num_minhash: int = 128,
    epsilon: float = 0.5
) -> HybridResult:
    """
    Produce a privacy‑aware, certainty‑aware hyper‑vector for a Voronoi cell.

    Steps
    -----
    1. Morphology → α = sp
    """
    morphology = region_morphology(region_points)
    alpha = morphology.sphericity

    point = region_points[0]  # arbitrary point in the region
    tokens = token_map.get(point, [])
    minhash = minhash_signature(tokens, num_perm=num_minhash)
    phase = np.array([2 * np.pi * (h / num_minhash) for h in minhash])

    hv = random_hv(d=hv_dim)
    bound_hv = fractional_binding(hv, phase, alpha)

    # Laplace mechanism to perturb the volume statistic
    volume = morphology.length * morphology.width * morphology.height
    noisy_volume = dp_laplace(volume, epsilon=epsilon, sensitivity=1.0)

    # Certainty flag
    certainty = region_certainty(tokens)

    # Reconstruction risk score
    quasi_identifiers = len(token_map)
    total_records = len(region_points)
    privacy_risk = reconstruction_risk_score(quasi_identifiers, total_records)

    return HybridResult(
        cell_id=cell_id,
        hypervector=bound_hv,
        certainty=certainty,
        privacy_risk=privacy_risk,
        morphology=morphology
    )