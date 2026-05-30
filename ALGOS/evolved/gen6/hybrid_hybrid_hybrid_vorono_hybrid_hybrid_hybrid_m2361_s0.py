# DARWIN HAMMER — match 2361, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_liquid_m825_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m935_s3.py (gen5)
# born: 2026-05-29T23:42:00Z

"""Hybrid Voronoi‑Morphology‑MinHash Engine
========================================

This module fuses the two parent algorithms:

* **Parent A** – Voronoi partitioning combined with a Hybrid Liquid‑Time‑Constant
  MinHash signature generator and Jaccard‑like similarity estimator.
* **Parent B** – A small epistemic‑certainty framework centred on a
  `CertaintyFlag` dataclass together with geometric morphology descriptors
  (`sphericity_index`, `flatness_index`).

**Mathematical bridge**

Both parents operate on *metric‑space representations* of data:

* The Voronoi routine groups 2‑D points into regions using Euclidean distance.
* The morphology utilities treat a region as a 3‑D object whose extents
  (`length`, `width`, `height`) are derived from the point cloud’s bounding box.
* The MinHash signatures provide a similarity score `S ∈ [0,1]` for the textual
  tokens attached to each point.

We map the similarity `S` onto a confidence value (`confidence_bps`) that
parameterises a `CertaintyFlag`.  The confidence is further modulated by the
region’s shape (sphericity/flatness) producing a single, mathematically
coherent description of each Voronoi cell.

The public API consists of three core functions that demonstrate the hybrid
behaviour:

1. `hybrid_assign(points, seeds)` – Voronoi assignment.
2. `region_morphology(region_points)` – compute `Morphology` and shape indices.
3. `region_certainty(region_points, token_map)` – generate a `CertaintyFlag`
   from MinHash similarity and morphology.

All code is pure Python 3 and uses only the allowed standard‑library modules."""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Iterable, Set

# ----------------------------------------------------------------------
# Parent A – Voronoi utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the seed closest to *point* (ties broken by index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Voronoi partition: map each point to its nearest seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Parent A – MinHash utilities
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash for a (seed, token) pair."""
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')

def signature(tokens: List[str], k: int = 128) -> List[int]:
    """Hybrid Liquid‑Time‑Constant MinHash signature of a token list."""
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must be non‑empty')
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)

# ----------------------------------------------------------------------
# Parent B – Epistemic certainty and morphology
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # basis points, 0 .. 10 000
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")
        if not self.generated_at:
            object.__setattr__(
                self,
                "generated_at",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "confidence_bps": self.confidence_bps,
            "authority_class": self.authority_class,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "generated_at": self.generated_at,
        }

def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )

class Morphology:
    """Geometric descriptor derived from a point cloud."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of the three extents to the maximal extent."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (max - min) / max of the three extents."""
    mx = max(length, width, height)
    mn = min(length, width, height)
    if mx == 0:
        raise ValueError("max dimension cannot be zero")
    return (mx - mn) / mx

# ----------------------------------------------------------------------
# Hybrid layer – mathematical fusion
# ----------------------------------------------------------------------
def hybrid_assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Voronoi assignment (exposes Parent A's core operation)."""
    return assign(points, seeds)

def region_morphology(region_points: List[Point]) -> Morphology:
    """Derive a 3‑D morphology from a 2‑D point set.

    * length  – span in the x‑direction
    * width   – span in the y‑direction
    * height  – synthetic depth proportional to point density
    * mass    – number of points (treated as “mass”)
    """
    if not region_points:
        raise ValueError("region must contain at least one point")
    xs, ys = zip(*region_points)
    length = max(xs) - min(xs) or 1e-9   # avoid zero
    width  = max(ys) - min(ys) or 1e-9
    # Height is defined as sqrt(mass / area) to give denser clusters a larger depth.
    area = length * width
    mass = float(len(region_points))
    height = math.sqrt(mass / (area + 1e-12))
    return Morphology(length, width, height, mass)

def region_average_similarity(region_points: List[Point],
                              token_map: Dict[Point, List[str]],
                              k: int = 128) -> float:
    """Average pairwise MinHash similarity among all points in a region."""
    if len(region_points) < 2:
        return 1.0  # a single element is perfectly similar to itself
    sigs = [signature(token_map[p], k) for p in region_points]
    total = 0.0
    count = 0
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            total += similarity(sigs[i], sigs[j])
            count += 1
    return total / count if count else 1.0

def region_certainty(region_points: List[Point],
                     token_map: Dict[Point, List[str]]) -> CertaintyFlag:
    """Create a CertaintyFlag that fuses MinHash similarity with morphology."""
    # 1. similarity → raw confidence (basis points)
    sim = region_average_similarity(region_points, token_map)
    raw_confidence = int(sim * 10_000)  # scale to basis points

    # 2. morphology influences the epistemic label
    morph = region_morphology(region_points)
    sph = sphericity_index(morph.length, morph.width, morph.height)
    flat = flatness_index(morph.length, morph.width, morph.height)

    # Heuristic mapping to label
    if raw_confidence > 8000 and sph > 0.7:
        label = "FACT"
    elif raw_confidence > 6000 and sph > 0.5:
        label = "PROBABLE"
    elif raw_confidence > 4000:
        label = "POSSIBLE"
    elif flat > 0.6:
        label = "BULLSHIT"
    else:
        label = "SURE_MAYBE"

    rationale = (
        f"Similarity={sim:.3f}, Sphericity={sph:.3f}, Flatness={flat:.3f}, "
        f"Mass={morph.mass}"
    )
    return certainty(
        label,
        confidence_bps=raw_confidence,
        authority_class="HybridEngine",
        rationale=rationale,
        evidence_refs=[],
    )

def hybrid_process(points: List[Point],
                   seeds: List[Point],
                   token_map: Dict[Point, List[str]]) -> Dict[int, CertaintyFlag]:
    """Full pipeline: Voronoi partition → morphology → certainty per region."""
    regions = hybrid_assign(points, seeds)
    result: Dict[int, CertaintyFlag] = {}
    for idx, pts in regions.items():
        if not pts:
            # Empty cells inherit a neutral certainty
            result[idx] = certainty(
                "SURE_MAYBE",
                confidence_bps=0,
                authority_class="HybridEngine",
                rationale="empty region",
                evidence_refs=[],
            )
            continue
        result[idx] = region_certainty(pts, token_map)
    return result

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic random seed for reproducibility
    random.seed(42)

    # generate 200 random 2‑D points inside a 100×100 square
    points: List[Point] = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(200)]

    # pick 5 random seeds from the points
    seeds = random.sample(points, 5)

    # associate a tiny bag‑of‑words to each point (simulated tokens)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta"]
    token_map: Dict[Point, List[str]] = {}
    for p in points:
        # each point gets 3‑5 random tokens
        token_map[p] = random.sample(vocab, random.randint(3, 5))

    # run the hybrid engine
    certs = hybrid_process(points, seeds, token_map)

    # display results
    for region_id, flag in certs.items():
        print(f"Region {region_id}:")
        for k, v in flag.as_dict().items():
            print(f"  {k}: {v}")
        print()
    sys.exit(0)