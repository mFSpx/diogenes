# DARWIN HAMMER — match 4685, survivor 1
# gen: 6
# parent_a: hybrid_voronoi_partition_percyphon_m779_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m1980_s1.py (gen5)
# born: 2026-05-29T23:57:26Z

"""
Hybrid Voronoi‑Ternary Morphology Analyzer

Parents:
- hybrid_voronoi_partition_percyphon_m779_s2.py
- hybrid_hybrid_ternar_hybrid_hybrid_m1980_s1.py

Mathematical bridge:
The Voronoi partition (Parent A) supplies geometric descriptors of each
region (centroid, spread, point count).  These descriptors are injected
into the ternary‑vector generator of Parent B, producing a 12‑dimensional
ternary weight vector that modulates the dimensions of a physical
morphology (length, width, height).  The weighted morphology is then
evaluated with the classic sphericity index, completing a closed loop
that fuses spatial partitioning, deterministic hashing, and
shape‑analysis mathematics.
"""

import math
import random
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]

# ----------------------------------------------------------------------
# Parent A core (Voronoi + procedural slot)
# ----------------------------------------------------------------------
def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError("seed list empty")
    return min(range(len(seeds)), key=lambda i: (_distance(point, seeds[i]), i))

def assign_to_regions(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to the nearest Voronoi seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[_nearest(p, seeds)].append(p)
    return regions

def _uuid_from_sha256(seed: str) -> str:
    h = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

def _slot_name(base: str, idx: int) -> Tuple[str, str, str]:
    h = hashlib.sha256(f"{base}:{idx}".encode()).hexdigest()
    name = f"Villager-{int(h[:6], 16) % 5000:04d}"
    alias = f"Alias-{h[6:10]}"
    persona = ["ledger", "runner", "witness", "archivist", "carrier", "scout"][int(h[10:12], 16) % 6]
    return name, alias, persona

@dataclass(frozen=True)
class ProceduralSlot:
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

# ----------------------------------------------------------------------
# Parent B core (ternary vector & sphericity)
# ----------------------------------------------------------------------
TERNARY_DIMS = 12

def payload_hash(raw_command: str, normalized_intent: str, context: Dict[str, Any]) -> str:
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

def ternary_vector(raw_command: str, normalized_intent: str, context: Dict[str, Any]) -> np.ndarray:
    """Deterministic 12‑dimensional ternary vector (-1,0,1) derived from a hash."""
    h = payload_hash(raw_command, normalized_intent, context)
    val = int(h, 16)
    vec = np.empty(TERNARY_DIMS, dtype=int)
    for i in range(TERNARY_DIMS):
        vec[i] = (val % 3) - 1  # maps 0→-1, 1→0, 2→1
        val //= 3
    return vec

class Morphology:
    """Physical object described by three orthogonal dimensions and a mass."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

def sphericity_index(morph: Morphology) -> float:
    """Classic sphericity σ = (π^{1/3} (6V)^{2/3}) / A."""
    V = morph.length * morph.width * morph.height
    A = 2 * (morph.length * morph.width + morph.length * morph.height + morph.width * morph.height)
    if A == 0:
        return 0.0
    sigma = (math.pi ** (1.0 / 3.0) * (6 * V) ** (2.0 / 3.0)) / A
    return sigma

# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def generate_random_points(num: int, bounds: Tuple[float, float, float, float]) -> List[Point]:
    """Uniformly sample `num` points inside the rectangular bounds (xmin, xmax, ymin, ymax)."""
    xmin, xmax, ymin, ymax = bounds
    return [(random.uniform(xmin, xmax), random.uniform(ymin, ymax)) for _ in range(num)]

def region_centroid(points: List[Point]) -> Point:
    """Arithmetic mean of a list of points; returns (0,0) for empty list."""
    if not points:
        return (0.0, 0.0)
    xs, ys = zip(*points)
    return (float(np.mean(xs)), float(np.mean(ys)))

def region_spread(points: List[Point]) -> float:
    """Maximum Euclidean distance from centroid to any point in the region."""
    if not points:
        return 0.0
    cx, cy = region_centroid(points)
    dists = [_distance((cx, cy), p) for p in points]
    return max(dists)

def weighted_morphology(
    points: List[Point],
    raw_command: str,
    normalized_intent: str,
    context: Dict[str, Any],
) -> Morphology:
    """
    Build a Morphology where the raw geometric extents are modulated by a
    ternary vector derived from the command payload.
    """
    # Base dimensions from geometry
    spread = region_spread(points) or 1e-6  # avoid zero
    count = max(len(points), 1)
    base_len = spread * 1.2
    base_wid = spread * 0.9
    base_hei = spread * 0.7

    # Ternary weighting
    tvec = ternary_vector(raw_command, normalized_intent, context)
    # Use first three components as scaling factors (-1,0,1) → (0.8,1.0,1.2)
    scale_factors = 1.0 + 0.2 * tvec[:3].astype(float)

    length = base_len * scale_factors[0]
    width = base_wid * scale_factors[1]
    height = base_hei * scale_factors[2]
    mass = count * 0.5  # arbitrary linear mass model

    return Morphology(length, width, height, mass)

def generate_procedural_slots(
    regions: Dict[int, List[Point]],
    base_seed: str,
) -> List[ProceduralSlot]:
    """Create one ProceduralSlot per Voronoi region using deterministic naming."""
    slots: List[ProceduralSlot] = []
    for idx, pts in regions.items():
        name, alias, persona = _slot_name(base_seed, idx)
        uuid = _uuid_from_sha256(f"{base_seed}:{idx}")
        ternary_offset = len(pts) % TERNARY_DIMS
        slot = ProceduralSlot(
            slot_index=idx,
            name=name,
            alias=alias,
            persona=persona,
            uuid=uuid,
            ternary_offset=ternary_offset,
        )
        slots.append(slot)
    return slots

def hybrid_analyze(
    points: List[Point],
    seeds: List[Point],
    raw_command: str,
    normalized_intent: str,
    context: Dict[str, Any],
) -> Tuple[Dict[int, Morphology], Dict[int, float], List[ProceduralSlot]]:
    """
    Full hybrid pipeline:
    1. Voronoi assignment of `points` to `seeds`.
    2. For each region, build a weighted Morphology.
    3. Compute sphericity for each Morphology.
    4. Generate ProceduralSlot objects.
    Returns (region→Morphology, region→sphericity, slots).
    """
    regions = assign_to_regions(points, seeds)
    morphologies: Dict[int, Morphology] = {}
    sphericities: Dict[int, float] = {}

    for idx, pts in regions.items():
        morph = weighted_morphology(pts, raw_command, normalized_intent, context)
        morphologies[idx] = morph
        sphericities[idx] = sphericity_index(morph)

    slots = generate_procedural_slots(regions, raw_command)
    return morphologies, sphericities, slots

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration
    NUM_POINTS = 500
    NUM_SEEDS = 7
    BOUNDS = (0.0, 100.0, 0.0, 100.0)  # xmin, xmax, ymin, ymax
    RAW_CMD = "spawn_entities"
    INTENT = "generate"
    CONTEXT = {"session": "test", "user": "alice"}

    # Generate data
    random.seed(42)
    points = generate_random_points(NUM_POINTS, BOUNDS)
    seeds = generate_random_points(NUM_SEEDS, BOUNDS)

    # Run hybrid analysis
    morphs, sphers, slots = hybrid_analyze(points, seeds, RAW_CMD, INTENT, CONTEXT)

    # Simple verification output
    print(f"Processed {NUM_POINTS} points into {NUM_SEEDS} Voronoi regions.")
    for idx in sorted(morphs):
        m = morphs[idx]
        s = sphers[idx]
        slot = next(slt for slt in slots if slt.slot_index == idx)
        print(
            f"Region {idx:2d}: "
            f"len={m.length:.2f}, wid={m.width:.2f}, hei={m.height:.2f}, "
            f"sphericity={s:.4f}, slot={slot.name} ({slot.uuid})"
        )