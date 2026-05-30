# DARWIN HAMMER — match 1658, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_fractional_hd_hybrid_korpus_text_h_m125_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s3.py (gen3)
# born: 2026-05-29T23:38:05Z

"""
Hybrid Hyperdimensional-Geometric Allocation (HHGA) Module
========================================================

This module fuses the core topologies of the two parent algorithms:

* **Parent A – hybrid_hybrid_fractional_hd_hybrid_korpus_text_h_m125_s1.py**  
  Provides hyperdimensional computing (HCHDC) primitives: random hypervectors,
  binding, fractional‑power binding and a compact text encoder (minhash).

* **Parent B – hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s3.py**  
  Provides a Liquid‑Time‑Constant (LTC) allocation scheduler and a Clifford
  geometric‑product based weight update mechanism.

**Mathematical Bridge**

The bridge is built by interpreting a *hypervector* (produced from text) as a
*multivector* in the 3‑dimensional Clifford algebra 𝔾³.  The first eight
components of a binary hypervector are reshaped into the eight basis elements
(scalar, e₁, e₂, e₃, e₁₂, e₁₃, e₂₃, e₁₂₃).  The weight matrix **W** of the geometric
product model is therefore a multivector **w** ∈ 𝔾³.

The LTC dynamics supply a time‑constant τ(d) that depends on the day‑of‑week
index *d*.  This τ(d) scales the geometric‑product update of **w** with the
encoded hypervector **h**:


w ← w + τ(d) · (w ⊛ h)            (⊛  = geometric product)


Thus the causal strength encoded by fractional‑power binding in the HCHDC
framework directly drives the geometric‑product evolution of the allocation
weights, while the LTC provides a smooth, day‑dependent modulation.

The three public functions below demonstrate this hybrid operation:
`init_hybrid_system`, `hybrid_allocate_by_dates`, and `summarize_hybrid_savings`.
"""

import re
import math
import random
import sys
import pathlib
import numpy as np
from datetime import date

# ---------------------------------------------------------------------------
# Hyperdimensional Computing primitives (Parent A)
# ---------------------------------------------------------------------------

def random_hv(dim: int = 10000) -> np.ndarray:
    """Generate a random binary hypervector with entries in {‑1, +1}."""
    return np.where(np.random.rand(dim) < 0.5, -1, 1).astype(np.int8)

def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Binding operator (element‑wise multiplication)."""
    return a * b

def unbind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unbinding is identical to binding for binary hypervectors."""
    return a * b

def fractional_power(hv: np.ndarray, alpha: float) -> np.ndarray:
    """
    Fractional power binding: raise the magnitude of each component to α
    while preserving the sign.
    """
    sign = np.sign(hv)
    return sign * (np.abs(hv) ** alpha)

def bundle(hvs: list[np.ndarray]) -> np.ndarray:
    """Superpose a list of hypervectors and binarise by majority vote."""
    sum_vec = np.sum(hvs, axis=0)
    return np.where(sum_vec >= 0, 1, -1).astype(np.int8)

def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two hypervectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def minhash_for_text(text: str, k: int = 64) -> list[int]:
    """
    Very lightweight minhash‑like sketch: hash each shingle and keep the k
    smallest hash values.
    """
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    if len(text) < 5:
        return [0] * k
    shingles = [text[i:i+5] for i in range(len(text) - 4)]
    hashes = [hash(s) & 0xffffffff for s in shingles]
    return sorted(hashes)[:k]

def encode_text_to_hv(text: str, dim: int = 10000) -> np.ndarray:
    """
    Encode text into a hypervector by seeding a RNG with the minhash sketch.
    """
    sketch = minhash_for_text(text, k=8)  # only eight values needed for 𝔾³
    seed = sum(sketch) & 0xffffffff
    rng = np.random.RandomState(seed)
    return np.where(rng.rand(dim) < 0.5, -1, 1).astype(np.int8)

# ---------------------------------------------------------------------------
# Geometric Algebra (Parent B) – 3‑D multivector implementation
# ---------------------------------------------------------------------------

# Basis order: [scalar, e1, e2, e3, e12, e13, e23, e123]
def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the geometric product a ⊛ b in 𝔾³.
    a and b are length‑8 arrays representing multivectors.
    """
    s1, e1, e2, e3, e12, e13, e23, e123 = a
    s2, f1, f2, f3, f12, f13, f23, f123 = b

    # Scalar part
    scalar = s1 * s2 + e1 * f1 + e2 * f2 + e3 * f3 - e12 * f12 - e13 * f13 - e23 * f23 - e123 * f123

    # Vector parts
    v1 = s1 * f1 + e1 * s2 + e2 * f12 - e3 * f13 + e12 * f2 - e13 * f3 - e23 * f123 + e123 * f23
    v2 = s1 * f2 - e1 * f12 + e2 * s2 + e3 * f23 + e12 * f1 - e13 * f123 - e23 * f3 - e123 * f13
    v3 = s1 * f3 + e1 * f13 - e2 * f23 + e3 * s2 + e12 * f123 + e13 * f1 - e23 * f2 - e123 * f12

    # Bivector parts
    b12 = s1 * f12 + e1 * f2 - e2 * f1 + e3 * f123 + e12 * s2 - e13 * f23 + e23 * f13 + e123 * f3
    b13 = s1 * f13 - e1 * f3 + e2 * f123 + e3 * f1 - e12 * f23 - e13 * s2 + e23 * f12 + e123 * f2
    b23 = s1 * f23 + e1 * f123 + e2 * f3 - e3 * f2 + e12 * f13 - e13 * f12 + e23 * s2 + e123 * f1

    # Trivector part
    t = s1 * f123 + e1 * f23 + e2 * f13 + e3 * f12 + e12 * f3 + e13 * f2 + e23 * f1 + e123 * s2

    return np.array([scalar, v1, v2, v3, b12, b13, b23, t], dtype=np.float64)

def hv_to_multivector(hv: np.ndarray) -> np.ndarray:
    """
    Map the first eight components of a binary hypervector to a multivector.
    The mapping preserves sign and treats the magnitude as weight.
    """
    # Ensure we have at least 8 components
    if hv.shape[0] < 8:
        raise ValueError("Hypervector must have at least 8 dimensions for multivector conversion.")
    # Convert {-1,1} to float and scale to [‑1,1]
    return hv[:8].astype(np.float64)

# ---------------------------------------------------------------------------
# Liquid‑Time‑Constant (LTC) dynamics (Parent B)
# ---------------------------------------------------------------------------

def ltc_time_constant(day_idx: int, base_tau: float = 1.0) -> float:
    """
    Day‑dependent effective time constant.
    day_idx ∈ {0,…,6} where 0 = Monday.
    """
    # Smooth sinusoidal modulation across the week
    omega = 2 * math.pi / 7
    return base_tau * (1.0 + 0.2 * math.sin(omega * day_idx))

# ---------------------------------------------------------------------------
# Hybrid System Construction
# ---------------------------------------------------------------------------

def init_hybrid_system(dim: int = 10000, seed: int | None = None):
    """
    Initialise the hybrid system:
    - random base hypervector (used as a seed for deterministic text encoding)
    - weight multivector w (initialized from a random hypervector)
    - store dimension for later use
    Returns a dict holding the state.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    base_hv = random_hv(dim)
    w = hv_to_multivector(base_hv)  # initial weight as multivector
    state = {
        "dim": dim,
        "base_hv": base_hv,
        "weight": w,
    }
    return state

def hybrid_update(state: dict, text: str, day_idx: int, alpha: float = 0.7):
    """
    Perform one hybrid update step:
    1. Encode the input text to a hypervector h.
    2. Apply fractional‑power binding to model causal strength.
    3. Convert h to a multivector h_mv.
    4. Update the weight multivector with LTC‑scaled geometric product.
    """
    dim = state["dim"]
    # 1. Encode text
    h = encode_text_to_hv(text, dim)

    # 2. Fractional power binding with the base hypervector (causal effect)
    h_fp = fractional_power(bind(state["base_hv"], h), alpha)

    # 3. Convert to multivector
    h_mv = hv_to_multivector(h_fp)

    # 4. LTC‑scaled geometric product update
    tau = ltc_time_constant(day_idx)
    delta = geometric_product(state["weight"], h_mv)
    state["weight"] = state["weight"] + tau * delta
    return state["weight"]

def hybrid_allocate_by_dates(state: dict, start_date: date, days: int = 7):
    """
    Compute per‑day allocation vectors for a span of `days` starting at `start_date`.
    The allocation for a given day d is the projection of the current weight onto
    the vector basis (e1, e2, e3) after an LTC‑scaled update with a dummy text.
    Returns a list of allocation vectors (length 3 each).
    """
    allocations = []
    dummy_text = "allocation placeholder"
    for offset in range(days):
        current_day = (start_date.weekday() + offset) % 7  # Monday=0
        # Update weight with dummy text to reflect day‑specific dynamics
        hybrid_update(state, dummy_text, current_day)
        # Projection onto vector part (components 1,2,3)
        vec_part = state["weight"][1:4]  # e1, e2, e3
        # Ensure non‑negative allocations (resource amounts)
        alloc = np.maximum(vec_part, 0.0)
        allocations.append(alloc.copy())
    return allocations

def summarize_hybrid_savings(state: dict, baseline_factor: float = 1.0):
    """
    Compare a baseline allocation (weight unchanged) with the hybrid LTC‑modulated
    allocation. Returns the percentage reduction in L1 norm of the allocation
    vector, interpreted as “savings”.
    """
    # Baseline weight (no LTC updates) – use initial weight copy
    baseline_weight = hv_to_multivector(state["base_hv"]).copy()
    # Current hybrid weight
    hybrid_weight = state["weight"]

    # Use L1 norm of vector components as a proxy for resource usage
    baseline_usage = np.sum(np.abs(baseline_weight[1:4])) * baseline_factor
    hybrid_usage = np.sum(np.abs(hybrid_weight[1:4]))

    if baseline_usage == 0:
        return 0.0
    savings = (baseline_usage - hybrid_usage) / baseline_usage * 100.0
    return savings

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialise system
    sys.setrecursionlimit(10000)
    state = init_hybrid_system(dim=10000, seed=42)

    # Perform a few updates with sample texts across a week
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Economic indicators suggest a modest growth this quarter.",
        "Quantum entanglement experiments achieve new fidelity.",
        "Artificial intelligence models become increasingly interpretable.",
        "Climate change mitigation policies gain global traction.",
        "Space exploration missions report successful landings.",
        "Healthcare innovations improve patient outcomes."
    ]

    today = date.today()
    for i, txt in enumerate(sample_texts):
        day_idx = (today.weekday() + i) % 7
        hybrid_update(state, txt, day_idx)

    # Allocate resources for the next 7 days
    allocations = hybrid_allocate_by_dates(state, today, days=7)
    print("Per‑day allocations (vector part e1‑e3):")
    for i, alloc in enumerate(allocations):
        day_name = (today + np.timedelta64(i, 'D')).astype('datetime64[D]').astype(str)
        print(f" Day {i} ({day_name}): {alloc}")

    # Summarize savings
    savings = summarize_hybrid_savings(state)
    print(f"\nEstimated resource savings vs. baseline: {savings:.2f}%")