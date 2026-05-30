# DARWIN HAMMER — match 5346, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_ssim_h_hybrid_hybrid_hybrid_m2479_s3.py (gen6)
# parent_b: hybrid_liquid_time_constant_hybrid_hybrid_hdc_se_m174_s1.py (gen4)
# born: 2026-05-30T00:01:26Z

import math
import random
import sys
from pathlib import Path
from typing import Sequence, Dict, FrozenSet, Iterable, List
import numpy as np

class Multivector:
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }
        self.n = int(n)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, value in other.components.items():
            result[blade] = result.get(blade, 0.0) + value
        return Multivector(result, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        result: Dict[FrozenSet[int], float] = {}
        for b1, v1 in self.components.items():
            for b2, v2 in other.components.items():
                new_blade = frozenset(b1.union(b2))
                result[new_blade] = result.get(new_blade, 0.0) + v1 * v2
        return Multivector(result, self.n)


def stats_to_multivector(seq: Sequence[float]) -> Multivector:
    arr = np.asarray(seq, dtype=float)
    mean = float(np.mean(arr)) if arr.size else 0.0
    var = float(np.var(arr)) if arr.size else 0.0
    cov = 0.0
    comps: Dict[FrozenSet[int], float] = {
        frozenset(): mean,               
        frozenset({0}): var,             
        frozenset({0, 1}): cov,          
    }
    return Multivector(comps, 2)


Vector = np.ndarray  


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    data = np.fromiter(
        (1 if rng.getrandbits(1) else -1 for _ in range(dim)),
        dtype=np.int8,
        count=dim,
    )
    return data


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, byteorder="big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return a * b


def bundle(vectors: Iterable[Vector]) -> Vector:
    vecs = list(vectors)
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    stacked = np.stack(vecs, axis=0).astype(np.int32)
    summed = stacked.sum(axis=0)
    return np.where(summed >= 0, 1, -1).astype(np.int8)


ALPHA = 0.6
BETA = 0.4
GATE_CLIP_LO = 0.0
GATE_CLIP_HI = 1.0
DEFAULT_DIM = 10000


def ltc_gate(seq: Sequence[float]) -> float:
    arr = np.asarray(seq, dtype=float)
    if arr.size == 0:
        return 0.5
    mean = float(np.mean(arr))
    var = float(np.var(arr))
    z = ALPHA * mean + BETA * var
    gate = 1.0 / (1.0 + math.exp(-z))
    return max(GATE_CLIP_LO, min(GATE_CLIP_HI, gate))


def multivector_to_hypervector(mv: Multivector, dim: int = DEFAULT_DIM) -> Vector:
    hypervectors: List[Vector] = []
    for blade, value in mv.components.items():
        symbol = f"blade_{'_'.join(map(str, sorted(blade)))}"
        base_vec = symbol_vector(symbol, dim)
        scaled = base_vec if value >= 0 else -base_vec
        hypervectors.append(scaled * abs(value)) # Scale by magnitude
    return bundle(hypervectors)


def hybrid_bind(
    seq_x: Sequence[float],
    seq_y: Sequence[float],
    dim: int = DEFAULT_DIM,
) -> Vector:
    mv_x = stats_to_multivector(seq_x)
    mv_y = stats_to_multivector(seq_y)

    hv_x = multivector_to_hypervector(mv_x, dim)
    hv_y = multivector_to_hypervector(mv_y, dim)

    combined = list(seq_x) + list(seq_y)
    g = ltc_gate(combined)

    bound = bind(hv_x, hv_y).astype(np.int32)
    result = (bound * g).astype(np.int8) # Direct scaling by gate
    return result


def hybrid_similarity(seq_a: Sequence[float], seq_b: Sequence[float]) -> float:
    hv = hybrid_bind(seq_a, seq_b)
    return float(np.mean(hv == 1)) * 2.0 - 1.0


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    seq1 = rng.normal(loc=0.0, scale=1.0, size=128).tolist()
    seq2 = rng.normal(loc=0.5, scale=1.5, size=128).tolist()
    similarity = hybrid_similarity(seq1, seq2)
    print(similarity)