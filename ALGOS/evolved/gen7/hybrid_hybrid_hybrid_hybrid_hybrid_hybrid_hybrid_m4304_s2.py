# DARWIN HAMMER — match 4304, survivor 2
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

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]

def distance(p1: Point2D, p2: Point2D) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest(point: Point2D, seeds: List[Point2D]) -> int:
    dists = [distance(point, s) for s in seeds]
    return int(min(range(len(dists)), key=lambda i: (dists[i], i)))

def hybrid_assign(points: List[Point2D], seeds: List[Point2D]) -> List[int]:
    return [nearest(p, seeds) for p in points]

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    sphericity: float
    flatness: float

def _bbox_3d(points: List[Point2D]) -> Tuple[float, float, float]:
    xs, ys = zip(*points)
    length = max(xs) - min(xs)
    width = max(ys) - min(ys)
    height = 0.0
    return length, width, height

def region_morphology(region_points: List[Point2D]) -> Morphology:
    length, width, height = _bbox_3d(region_points)
    vol = max(length * width * height, 1e-9)
    surface = 2 * (length * width + width * height + height * length)
    sphericity = (math.pi ** (1/3) * (6 * vol) ** (2/3)) / max(surface, 1e-9)
    sphericity = max(0.0, min(1.0, sphericity))
    flatness = 1.0 - sphericity
    return Morphology(length, width, height, sphericity, flatness)

def _hash_token(token: str, seed: int) -> int:
    h = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")

def minhash_signature(tokens: Iterable[str], num_perm: int = 128) -> List[int]:
    signature = [2**64 - 1] * num_perm
    for token in tokens:
        for i in range(num_perm):
            hv = _hash_token(token, i)
            if hv < signature[i]:
                signature[i] = hv
    return signature

@dataclass(frozen=True)
class CertaintyFlag:
    confidence_bps: int
    timestamp: datetime
    comment: str

def _confidence_from_similarity(sim: float) -> int:
    return int(max(0, min(1, sim)) * 10_000)

def region_certainty(tokens: Iterable[str]) -> CertaintyFlag:
    sim = 1.0 - random.random() * 0.05
    confidence = _confidence_from_similarity(sim)
    return CertaintyFlag(
        confidence_bps=confidence,
        timestamp=datetime.now(timezone.utc),
        comment=f"Estimated similarity {sim:.3f}"
    )

def random_hv(d: int = 10000, kind: str = "complex", seed: Any = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    elif kind == "bipolar":
        return rng.choice([-1, 1], size=d)
    elif kind == "real":
        return rng.standard_normal(d)
    else:
        raise ValueError(f"Unsupported kind={kind}")

def fractional_binding(v: np.ndarray, phase: np.ndarray, alpha: float) -> np.ndarray:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0,1]")
    phase_complex = np.exp(1j * phase)
    bound = v * (phase_complex ** alpha)
    return bound

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def dp_laplace(value: float, epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    scale = sensitivity / max(epsilon, 1e-9)
    noise = np.random.laplace(0.0, scale)
    return value + noise

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
    morphology = region_morphology(region_points)
    alpha = morphology.sphericity
    tokens = [t for p in region_points for t in token_map.get(p, [])]
    minhash_sig = minhash_signature(tokens, num_perm=num_minhash)
    phase = np.array([2 * math.pi * x / (2**64 - 1) for x in minhash_sig])
    hv = random_hv(d=hv_dim)
    bound_hv = fractional_binding(hv, phase, alpha)
    certainty = region_certainty(tokens)
    volume = max(morphology.length * morphology.width * morphology.height, 1e-9)
    noisy_volume = dp_laplace(volume, epsilon=epsilon)
    privacy_risk = reconstruction_risk_score(len(set(tokens)), len(region_points))
    return HybridResult(
        cell_id=cell_id,
        hypervector=bound_hv,
        certainty=certainty,
        privacy_risk=privacy_risk,
        morphology=morphology
    )