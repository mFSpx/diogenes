# DARWIN HAMMER — match 4612, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_tropic_m2576_s4.py (gen6)
# parent_b: hybrid_rectified_flow_hybrid_ternary_lens__m404_s1.py (gen4)
# born: 2026-05-29T23:57:05Z

"""Hybrid Algorithm Fusion of Parent A and Parent B

Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_tropic_m2576_s4.py) provides:
- Tropical algebra primitives (t_add = max, t_mul = add, t_matmul, t_polyval)
- Gaussian‑beam based Fisher information (gaussian_beam, fisher_score)
- Trust‑weighted velocity utilities (trust_weighted_velocity, hybrid_trust_tropical_velocity)
- Euclidean distance (clifford_geometric_distance)

Parent B (hybrid_rectified_flow_hybrid_ternary_lens__m404_s1.py) provides:
- A simple resource‑aware model pool (ModelPool, ModelTier, Candidate)
- Linear interpolation / constant‑velocity flow (interpolant, flow_target)
- Exponential reconstruction risk (reconstruction_risk_score)

**Mathematical Bridge**
Both parents manipulate *vectors* (or tensors) and *weights* that modulate those vectors.
Parent A uses a *trust* scalar to weight a velocity, while Parent B computes a
*risk* scalar from model‑resource constraints.  The bridge is therefore a **scalar
weight** that can be derived from model risk and fed into the tropical‑algebraic
velocity composition of Parent A.  The hybrid algorithm:

1. Derives a unified *trust* factor `w = anti_slop_ratio * reconstruction_risk_score`.
2. Computes a base flow `v = flow_target(x0, x1)`.
3. Modulates `v` by the Fisher information of an angular parameter `theta`
   (Parent A) and by the unified weight `w`.
4. Combines the two weighted velocities using tropical addition (`t_add`) and
   tropical multiplication (`t_mul`) to obtain a *hybrid tropical‑Fisher flow*.

The same unified weight is also used to blend Euclidean and tropical distances,
producing a risk‑aware hybrid distance metric.

The module below implements three core hybrid functions that embody this
integration, together with a lightweight smoke test.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Iterable, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities (tropical algebra, Gaussian beam, Fisher score, etc.)
# ----------------------------------------------------------------------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def trust_weighted_velocity(x0: float, x1: float, trust: float) -> float:
    return trust * (x1 - x0)

def t_add(x, y):
    """Tropical addition (max). Works element‑wise for ndarrays."""
    return np.maximum(x, y)

def t_mul(x, y):
    """Tropical multiplication (add). Works element‑wise for ndarrays."""
    return np.add(x, y)

def clifford_geometric_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sqrt(np.sum((a - b) ** 2))

# ----------------------------------------------------------------------
# Parent B utilities (model pool, risk score, flow, etc.)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int

@dataclass
class Candidate:
    candidate_key: str
    family: str
    notes: str = ''
    classification: str = 'safe'
    fast_path_compatible: bool = True
    benchmark_required: bool = False
    benchmark_evidence: bool = False

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000, vram_ceiling_mb: int = 4096):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.vram_ceiling_mb = vram_ceiling_mb
        self.loaded: dict[str, ModelTier] = {}
        self.sensitive_records: List[Any] = []

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _used_vram(self) -> int:
        return sum(m.vram_mb for m in self.loaded.values())

    def load(self, model: ModelTier, candidate: Candidate) -> None:
        """Load a model only if it respects the resource ceilings and the candidate
        is marked as unsafe for fast‑path (mirroring the original logic)."""
        if candidate.classification == "unsafe_for_fastpath" and model.tier == "T3":
            if self._used_ram() + model.ram_mb > self.ram_ceiling_mb:
                raise RuntimeError("RAM ceiling exceeded")
            if self._used_vram() + model.vram_mb > self.vram_ceiling_mb:
                raise RuntimeError("VRAM ceiling exceeded")
            self.loaded[model.name] = model

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Exponential risk score – lower when identifiers are sparse."""
    if total_records == 0:
        return 1.0
    return math.exp(-unique_quasi_identifiers / total_records)

def interpolant(x0: np.ndarray, x1: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Linear interpolation Z_t = t * x1 + (1‑t) * x0."""
    t_arr = np.asarray(t, dtype=float)
    if t_arr.ndim == 0:
        t_arr = t_arr.reshape(1)
    return t_arr * x1 + (1.0 - t_arr) * x0

def flow_target(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Constant‑velocity field: v = x1 - x0."""
    return np.asarray(x1, dtype=float) - np.asarray(x0, dtype=float)

# ----------------------------------------------------------------------
# Hybrid core: unified weight, tropical‑Fisher flow, risk‑aware distance
# ----------------------------------------------------------------------
def unified_trust_weight(candidate: Candidate,
                         model_pool: ModelPool,
                         claims_with_evidence: int,
                         total_claims_emitted: int,
                         unique_quasi_identifiers: int,
                         total_records: int) -> float:
    """
    Combine the anti‑slop ratio from Parent A with the reconstruction risk
    score from Parent B into a single scalar trust factor in [0, 1].
    """
    slop = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    risk = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    # Both factors are in [0,1]; geometric mean keeps the result bounded.
    return math.sqrt(slop * risk)

def hybrid_tropical_fisher_flow(x0: np.ndarray,
                                x1: np.ndarray,
                                theta: float,
                                trust: float,
                                center: float = 0.0,
                                width: float = 1.0) -> np.ndarray:
    """
    Produce a flow vector that merges:
    * Tropical weighting (t_add, t_mul) of a trust‑scaled velocity,
    * Fisher‑information scaling derived from a Gaussian beam.
    Steps:
    1. Base velocity v = flow_target(x0, x1).
    2. Trust‑scaled velocity vt = trust_weighted_velocity(v, trust)   (element‑wise)
    3. Fisher scaling factor f = fisher_score(theta, center, width).
    4. Tropical combination:
       hybrid = t_add(vt, t_mul(v, f))
    """
    v = flow_target(x0, x1)                       # shape (d,)
    vt = trust * v                                 # trust_weighted_velocity for vectors
    f = fisher_score(theta, center, width)        # scalar
    vf = t_mul(v, f)                               # tropical multiplication (adds f)
    hybrid = t_add(vt, vf)                         # tropical addition (max)
    return hybrid

def hybrid_risk_aware_distance(p: np.ndarray,
                               q: np.ndarray,
                               candidate: Candidate,
                               model_pool: ModelPool,
                               claims_with_evidence: int,
                               total_claims_emitted: int,
                               unique_quasi_identifiers: int,
                               total_records: int) -> float:
    """
    Compute a distance that blends Euclidean and tropical metrics,
    weighted by the unified trust factor.
    """
    # Euclidean component
    euclid = clifford_geometric_distance(p, q)

    # Tropical component: treat vectors as points in tropical space,
    # distance = max_i |p_i - q_i|
    tropical = np.max(np.abs(p - q))

    # Unified trust weight
    w = unified_trust_weight(candidate, model_pool,
                             claims_with_evidence, total_claims_emitted,
                             unique_quasi_identifiers, total_records)

    # Blend: weighted harmonic mean keeps the result in a sensible range.
    blended = 1.0 / (w / euclid + (1 - w) / tropical) if euclid > 0 and tropical > 0 else max(euclid, tropical)
    return blended

def hybrid_energy(candidate: Candidate,
                  prev_candidate: Candidate,
                  x0: np.ndarray,
                  x1: np.ndarray,
                  theta: float,
                  model_pool: ModelPool,
                  claims_with_evidence: int,
                  total_claims_emitted: int,
                  unique_quasi_identifiers: int,
                  total_records: int) -> float:
    """
    Energy-like metric that combines:
    * The Jeopardy energy from Parent A (squared error between a predicted
      candidate value and the actual candidate value),
    * The hybrid tropical‑Fisher flow magnitude,
    * The unified trust weight as a scaling factor.
    """
    # Simulated scalar "candidate values" using hash‑derived floats
    def _scalar_from_key(key: str) -> float:
        random.seed(key)
        return random.random()

    cand_val = _scalar_from_key(candidate.candidate_key)
    prev_val = _scalar_from_key(prev_candidate.candidate_key)

    # Fisher‑scaled flow magnitude
    flow_vec = hybrid_tropical_fisher_flow(x0, x1, theta, trust=1.0)
    flow_mag = np.linalg.norm(flow_vec)

    # Jeopardy energy (Parent A)
    predictor = np.array([prev_val + flow_mag])
    encoder = np.array([cand_val])
    jeap = np.sum((encoder - predictor) ** 2)

    # Unified trust weight
    w = unified_trust_weight(candidate, model_pool,
                             claims_with_evidence, total_claims_emitted,
                             unique_quasi_identifiers, total_records)

    return w * jeap + (1 - w) * flow_mag

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny model pool and load a dummy model
    pool = ModelPool(ram_ceiling_mb=8000, vram_ceiling_mb=5000)
    dummy_model = ModelTier(name="tiny-model", ram_mb=1024, tier="T3", vram_mb=2048)
    unsafe_candidate = Candidate(candidate_key="c1", family="test", classification="unsafe_for_fastpath")
    pool.load(dummy_model, unsafe_candidate)

    # Random vectors
    dim = 5
    x0 = np.random.randn(dim)
    x1 = np.random.randn(dim)

    # Parameters for trust/score calculations
    claims_with_evidence = 42
    total_claims_emitted = 100
    unique_quasi_identifiers = 7
    total_records = 1000
    theta = random.uniform(-math.pi, math.pi)

    # Compute hybrid flow
    trust = unified_trust_weight(unsafe_candidate, pool,
                                 claims_with_evidence, total_claims_emitted,
                                 unique_quasi_identifiers, total_records)
    flow = hybrid_tropical_fisher_flow(x0, x1, theta, trust)
    print("Hybrid tropical‑Fisher flow:", flow)

    # Compute hybrid distance between x0 and x1
    dist = hybrid_risk_aware_distance(x0, x1, unsafe_candidate, pool,
                                      claims_with_evidence, total_claims_emitted,
                                      unique_quasi_identifiers, total_records)
    print("Hybrid risk‑aware distance:", dist)

    # Energy between two candidates
    prev_cand = Candidate(candidate_key="c0", family="test")
    energy = hybrid_energy(unsafe_candidate, prev_cand, x0, x1, theta, pool,
                           claims_with_evidence, total_claims_emitted,
                           unique_quasi_identifiers, total_records)
    print("Hybrid energy metric:", energy)