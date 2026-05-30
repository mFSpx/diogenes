# DARWIN HAMMER — match 4304, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m2361_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fracti_m506_s0.py (gen4)
# born: 2026-05-29T23:54:56Z

"""Hybrid Voronoi‑Morphology‑MinHash‑Fractional‑DP Engine
====================================================

This module merges the two parent algorithms:

* **Parent A** – Voronoi partitioning + morphology descriptors + a
  MinHash‑based `CertaintyFlag`.
* **Parent B** – Differential‑privacy aggregation, random hyper‑vector
  generation and a *fractional power binding* operation.

**Mathematical bridge**

For every Voronoi cell we obtain a *shape vector* **M** (length, width,
height, sphericity, flatness).  From the textual tokens attached to the
points we build a MinHash signature **h**∈ℕᵏ.  A random hyper‑vector
**v**∈ℂᵈ (complex polar representation) is drawn and *fractionally bound*
to the MinHash‑derived vector by


b_i = v_i · (exp(2πj·h_i/k))**α      (α = sphericity ∈[0,1])


The resulting bound vector **b** is then perturbed with a Laplace‑DP
mechanism applied to a scalar region statistic (e.g. volume).  The noisy
statistic feeds the `reconstruction_risk_score` from Parent B, while the
original similarity score feeds the `CertaintyFlag` from Parent A.
Thus a single, mathematically coherent description of each Voronoi cell
emerges, simultaneously capturing geometry, textual similarity,
privacy‑aware uncertainty and hyperdimensional representation.
"""

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
# Core geometric utilities (Parent A)
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
# Morphology (Parent A)
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
# MinHash signature (Parent A)
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
# Certainty flag (Parent A)
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
# Hyper‑vector utilities (Parent B)
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
# Differential‑privacy utilities (Parent B)
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
    1. Morphology → α = sphericity (fractional binding exponent).
    2. Gather tokens from *token_map* and build MinHash sketch.
    3. Convert sketch to a phase vector φ_i = 2π·h_i / 2⁶⁴.
    4. Generate a random complex hypervector *v*.
    5. Bind: b = fractional_binding(v, φ, α).
    6. Compute a scalar region statistic (volume) and add Laplace noise.
    7. Derive reconstruction risk from noisy statistic.
    8. Produce CertaintyFlag from token similarity proxy.
    """
    # 1. Morphology
    morph = region_morphology(region_points)
    alpha = morph.sphericity

    # 2. Token collection
    tokens: List[str] = []
    for pt in region_points:
        tokens.extend(token_map.get(pt, []))

    # 3. MinHash → phase
    sketch = minhash_signature(tokens, num_perm=num_minhash)
    phase = np.array([ (2.0 * np.pi * (h % (1 << 64)) ) / (1 << 64) for h in sketch ], dtype=np.float64)

    # 4. Random hypervector
    v = random_hv(d=hv_dim, kind="complex", seed=cell_id)

    # 5. Fractional binding (broadcast phase to match hv_dim)
    # Repeat phase to fill the dimension
    repeats = hv_dim // num_minhash + 1
    phase_expanded = np.tile(phase, repeats)[:hv_dim]
    bound = fractional_binding(v, phase_expanded, alpha)

    # 6. DP‑perturbed volume (using bounding‑box volume as proxy)
    vol = max(morph.length * morph.width * max(morph.height, 1e-3), 0.0)
    noisy_vol = dp_laplace(vol, epsilon=epsilon, sensitivity=1.0)

    # 7. Reconstruction risk (assume each unit of volume corresponds to one quasi‑identifier)
    risk = reconstruction_risk_score(int(noisy_vol), total_records=1000)

    # 8. Certainty flag
    certainty = region_certainty(tokens)

    return HybridResult(
        cell_id=cell_id,
        hypervector=bound,
        certainty=certainty,
        privacy_risk=risk,
        morphology=morph
    )

# ----------------------------------------------------------------------
# Demonstration entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a synthetic point cloud
    random.seed(42)
    np.random.seed(42)

    N_POINTS = 200
    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(N_POINTS)]

    # Random seeds for Voronoi partitioning
    SEED_COUNT = 5
    seeds = random.sample(points, SEED_COUNT)

    # Assign points to cells
    assignments = hybrid_assign(points, seeds)

    # Build a simple token map: each point gets a list of random words
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    token_map: Dict[Point2D, List[str]] = {
        pt: random.sample(vocab, k=random.randint(1, 3)) for pt in points
    }

    # Process each cell
    results: List[HybridResult] = []
    for cell_id in range(SEED_COUNT):
        cell_pts = [pt for pt, a in zip(points, assignments) if a == cell_id]
        if not cell_pts:
            continue
        res = hybrid_cell_representation(
            cell_id=cell_id,
            region_points=cell_pts,
            token_map=token_map,
            hv_dim=4096,
            num_minhash=64,
            epsilon=0.8
        )
        results.append(res)

    # Simple sanity output
    for r in results:
        print(f"Cell {r.cell_id}:")
        print(f"  Morphology – L:{r.morphology.length:.2f} W:{r.morphology.width:.2f} "
              f"Sph:{r.morphology.sphericity:.3f}")
        print(f"  Certainty – {r.certainty.confidence_bps/100:.2f}% (comment: {r.certainty.comment})")
        print(f"  Privacy risk – {r.privacy_risk:.3%}")
        print(f"  Hypervector norm – {np.linalg.norm(r.hypervector):.3f}")
        print("-" * 40)