# DARWIN HAMMER — match 5346, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_ssim_h_hybrid_hybrid_hybrid_m2479_s3.py (gen6)
# parent_b: hybrid_liquid_time_constant_hybrid_hybrid_hdc_se_m174_s1.py (gen4)
# born: 2026-05-30T00:01:26Z

"""Hybrid Algorithm integrating Multivector statistics (Parent A) with
Liquid‑Time‑Constant gated Hyperdimensional Computing (Parent B).

Mathematical Bridge
-------------------
* Parent A supplies a *Multivector* whose scalar part equals the product of
  statistical descriptors (mean, variance, covariance) of a sequence.
* Parent B provides hyperdimensional primitives (random vectors, binding,
  bundling) and a Liquid‑Time‑Constant (LTC) style gating function that
  yields a time‑varying weight `g ∈ (0,1)` from the same statistics.

The fusion consists of:
1. Converting a `Multivector` into a bipolar hypervector by binding symbolic
   hypervectors of its blade identifiers and scaling by the blade’s scalar.
2. Computing an LTC‑style gate `g` from the input statistics.
3. Modulating the hyperdimensional *binding* of two such representations with
   `g`, i.e. `result = g * bind(H_x, H_y)`.

The resulting hypervector evolves with the data distribution (via the
Multivector) and adapts instantaneously through the LTC gate, unifying the
core topologies of both parents into a single system.
"""

import math
import random
import sys
from pathlib import Path
from typing import Sequence, Dict, FrozenSet, Iterable, List
import numpy as np

# ---------------------------------------------------------------------------
# Parent A – Multivector and statistical encoding
# ---------------------------------------------------------------------------

class Multivector:
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }
        self.n = int(n)

    def scalar_part(self) -> float:
        """Return the scalar (grade‑0) component."""
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, value in other.components.items():
            result[blade] = result.get(blade, 0.0) + value
        return Multivector(result, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product – here a simple outer product of scalar values."""
        result: Dict[FrozenSet[int], float] = {}
        for b1, v1 in self.components.items():
            for b2, v2 in other.components.items():
                new_blade = frozenset(b1.union(b2))
                result[new_blade] = result.get(new_blade, 0.0) + v1 * v2
        return Multivector(result, self.n)


def stats_to_multivector(seq: Sequence[float]) -> Multivector:
    """Encode mean, variance and a dummy covariance into a 2‑grade Multivector."""
    arr = np.asarray(seq, dtype=float)
    mean = float(np.mean(arr)) if arr.size else 0.0
    var = float(np.var(arr)) if arr.size else 0.0
    # Covariance placeholder (zero) – kept for structural compatibility
    cov = 0.0
    comps: Dict[FrozenSet[int], float] = {
        frozenset(): mean,               # grade‑0 (scalar)
        frozenset({0}): var,             # grade‑1
        frozenset({0, 1}): cov,          # grade‑2
    }
    return Multivector(comps, 2)

# ---------------------------------------------------------------------------
# Parent B – Hyperdimensional primitives and LTC‑style gating
# ---------------------------------------------------------------------------

Vector = np.ndarray  # bipolar hypervector of dtype int8


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    """Generate a random bipolar hypervector."""
    rng = random.Random(seed)
    data = np.fromiter(
        (1 if rng.getrandbits(1) else -1 for _ in range(dim)),
        dtype=np.int8,
        count=dim,
    )
    return data


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    """Deterministically map a string symbol to a bipolar hypervector."""
    # Use SHA‑256 to obtain a reproducible integer seed
    import hashlib

    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, byteorder="big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise multiplication (XOR‑like binding for bipolar vectors)."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return a * b


def bundle(vectors: Iterable[Vector]) -> Vector:
    """Majority‑vote bundling of bipolar vectors."""
    vecs = list(vectors)
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    stacked = np.stack(vecs, axis=0).astype(np.int32)
    summed = stacked.sum(axis=0)
    # Majority vote: sign of the sum (ties become 1)
    return np.where(summed >= 0, 1, -1).astype(np.int8)


# ---------------------------------------------------------------------------
# Hybrid components
# ---------------------------------------------------------------------------

ALPHA = 0.6
BETA = 0.4
GATE_CLIP_LO = 0.0
GATE_CLIP_HI = 1.0
DEFAULT_DIM = 10000


def ltc_gate(seq: Sequence[float]) -> float:
    """
    Liquid‑Time‑Constant inspired gate.
    Computes g = sigmoid(ALPHA * mean + BETA * variance) and clips to [0,1].
    """
    arr = np.asarray(seq, dtype=float)
    if arr.size == 0:
        return 0.5
    mean = float(np.mean(arr))
    var = float(np.var(arr))
    z = ALPHA * mean + BETA * var
    gate = 1.0 / (1.0 + math.exp(-z))
    return max(GATE_CLIP_LO, min(GATE_CLIP_HI, gate))


def multivector_to_hypervector(mv: Multivector, dim: int = DEFAULT_DIM) -> Vector:
    """
    Map each blade of a Multivector to a symbolic hypervector.
    The blade’s scalar magnitude scales the sign of the symbol vector;
    all blade vectors are bundled together.
    """
    hypervectors: List[Vector] = []
    for blade, value in mv.components.items():
        # Symbolic identifier for the blade
        symbol = f"blade_{'_'.join(map(str, sorted(blade)))}"
        base_vec = symbol_vector(symbol, dim)
        # Scale by sign of the component (ignore magnitude for bipolarity)
        scaled = base_vec if value >= 0 else -base_vec
        hypervectors.append(scaled)
    return bundle(hypervectors)


def hybrid_bind(
    seq_x: Sequence[float],
    seq_y: Sequence[float],
    dim: int = DEFAULT_DIM,
) -> Vector:
    """
    Core hybrid operation:
    1. Encode each sequence as a Multivector (Parent A).
    2. Convert each Multivector into a hypervector (Parent B).
    3. Compute an LTC gate from the concatenated statistics.
    4. Bind the two hypervectors and modulate by the gate.
    """
    mv_x = stats_to_multivector(seq_x)
    mv_y = stats_to_multivector(seq_y)

    hv_x = multivector_to_hypervector(mv_x, dim)
    hv_y = multivector_to_hypervector(mv_y, dim)

    # Gate based on combined statistics (simple concatenation)
    combined = list(seq_x) + list(seq_y)
    g = ltc_gate(combined)

    bound = bind(hv_x, hv_y).astype(np.int32)
    # Modulate: scaling bipolar values by gate retains bipolarity
    result = np.where(bound >= 0, 1, -1).astype(np.int8)
    # Apply gate as a probabilistic flip: each bit flips with probability (1-g)
    flip_mask = np.random.rand(dim) < (1.0 - g)
    result[flip_mask] *= -1
    return result


def hybrid_similarity(seq_a: Sequence[float], seq_b: Sequence[float]) -> float:
    """
    Compute a similarity score between two sequences using the hybrid
    representation. The score is the normalized dot product of the gated
    hypervector binding.
    """
    hv = hybrid_bind(seq_a, seq_b)
    # Normalized dot product for bipolar vectors equals 1 - 2*Hamming distance/dim
    return float(np.mean(hv == 1)) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate two random real‑valued sequences
    rng = np.random.default_rng(42)
    seq1 = rng.normal(loc=0.0, scale=1.0, size=128).tolist()
    seq2 = rng.normal(loc=0.5, scale=1.5, size=128).tolist()

    # Run core hybrid functions
    gate_val = ltc_gate(seq1 + seq2)
    print(f"LTC gate value: {gate_val:.4f}")

    mv1 = stats_to_multivector(seq1)
    mv2 = stats_to_multivector(seq2)
    print(f"Multivector scalar parts: {mv1.scalar_part():.4f}, {mv2.scalar_part():.4f}")

    hv1 = multivector_to_hypervector(mv1)
    hv2 = multivector_to_hypervector(mv2)
    print(f"Hypervectors shapes: {hv1.shape}, {hv2.shape}")

    hybrid_hv = hybrid_bind(seq1, seq2)
    print(f"Hybrid hypervector first 10 bits: {hybrid_hv[:10].tolist()}")

    sim = hybrid_similarity(seq1, seq2)
    print(f"Hybrid similarity score: {sim:.4f}")

    # Ensure the code runs without raising exceptions
    sys.exit(0)