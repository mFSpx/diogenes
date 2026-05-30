# DARWIN HAMMER — match 3144, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1660_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m414_s2.py (gen5)
# born: 2026-05-29T23:48:07Z

"""Hybrid Algorithm Fusion of Parent A (Morphology‑Righting‑ModelPool) and Parent B (HMTHDC‑Minimum‑Cost‑Bayesian).

Mathematical Bridge
-------------------
Parent A provides a scalar *recovery priority* `p = recovery_priority(m) ∈ [0,1]`
derived from the morphology’s righting‑time index.  
Parent B uses a hyperdimensional representation `h = encode(m, text)` and a
minimum‑cost tree whose edge weights `w` are later updated by a Bayesian rule.

The fusion uses `p` as a prior probability in the Bayesian update of each edge:

posterior_weight = (w * p) / (w * p + (1 - w) * (1 - p))

Thus the physical shape influences the probabilistic relevance of graph paths.
The hypervector `h` is also modulated by `p` to embed the same prior into the
semantic encoding:

h_fused = h * (1 + p)

The resulting system jointly reasons over morphology, text, and graph structure
in a single unified representation.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

# ---------- Unified Core Components ----------
@dataclass(frozen=True)
class Morphology:
    """Unified morphology descriptor used by both parent systems."""
    name: str
    length: float
    width: float
    height: float
    mass: float          # physical mass (kg)
    ram_mb: float = 0.0  # optional RAM footprint for model‑pool logic

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1] derived from righting time."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if den == 0 else float(np.dot(a, b) / den)

# ---------- Model Pool (Parent A) ----------
class ModelPool:
    """Manages loaded models respecting a RAM ceiling."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, Morphology] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used_ram(self) -> float:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_load(self, model: Morphology) -> bool:
        """True if the model fits within remaining RAM."""
        return (self._used_ram() + model.ram_mb) <= self.ram_ceiling_mb

    def load(self, model: Morphology) -> None:
        """Load a model if RAM permits; raise otherwise."""
        if self.is_loaded(model.name):
            return
        if not self.can_load(model):
            raise RuntimeError(f"Insufficient RAM to load model {model.name}")
        self.loaded[model.name] = model

# ---------- Hyperdimensional Utilities (Parent B) ----------
Vector = List[float]
HV = np.ndarray  # bipolar hypervector (+1 / -1)

def random_vector(dim: int = 10000, seed: Optional[int] = None) -> Vector:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]

def morphology_vector(m: Morphology, dim: int = 10000) -> Vector:
    """Embed morphology into a high‑dimensional real vector."""
    base = [m.length, m.width, m.height, m.mass]
    padding = [0.0] * (dim - len(base))
    return base + padding

def text_hypervector(text: str, dim: int = 10000) -> HV:
    """Deterministic bipolar hypervector from text via MD5 hashing."""
    bits = []
    for i in range(dim):
        h = hashlib.md5(f"{text}{i}".encode()).hexdigest()
        bits.append(1 if int(h, 16) % 2 == 0 else -1)
    return np.array(bits, dtype=np.int8)

def bind_vectors(v1: HV, v2: HV) -> HV:
    """Element‑wise multiplication (binding) of two bipolar hypervectors."""
    return v1 * v2

# ---------- Hybrid Operations ----------
def hybrid_encode(morph: Morphology, text: str, dim: int = 10000) -> HV:
    """
    Produce a fused hypervector that incorporates morphology, text,
    and the morphology‑derived prior `p`.
    """
    # Encode each modality
    morph_vec = np.array(morphology_vector(morph, dim), dtype=np.float32)
    # Binarize morphology vector to bipolar form
    morph_hv = np.where(morph_vec > 0.5, 1, -1).astype(np.int8)

    text_hv = text_hypervector(text, dim)

    # Bind modalities
    bound = bind_vectors(morph_hv, text_hv)

    # Modulate by recovery priority (acts as a scalar prior)
    p = recovery_priority(morph)
    fused = bound.astype(np.float32) * (1.0 + p)  # keep bipolar sign, scale magnitude
    return fused.astype(np.int8)

def hybrid_tree_score(
    nodes: List[str],
    edges: List[Tuple[str, str, float]],
    labels: Dict[Tuple[str, str], str],
    morph: Morphology,
    text: str,
    dim: int = 10000
) -> float:
    """
    Compute the total score of a minimum‑cost tree after Bayesian update.
    Each edge weight `w` is updated using the morphology‑derived prior `p`.
    The overall tree score is the sum of updated weights plus a semantic term
    derived from the fused hypervector.
    """
    if not nodes:
        raise ValueError("node list cannot be empty")

    p = recovery_priority(morph)  # prior probability in [0,1]

    # Bayesian update of each edge weight
    updated_weights = []
    for u, v, w in edges:
        # Ensure w is a probability‑like quantity in [0,1]
        w = max(0.0, min(1.0, w))
        posterior = (w * p) / (w * p + (1.0 - w) * (1.0 - p) + 1e-12)
        updated_weights.append(posterior)

    # Semantic contribution from fused hypervector
    fused_hv = hybrid_encode(morph, text, dim).astype(np.float32)
    # Simple semantic score: mean absolute value (reflects magnitude scaling)
    semantic_score = float(np.mean(np.abs(fused_hv)))

    total_score = sum(updated_weights) + semantic_score
    return total_score

def hybrid_effect_estimate(
    morph1: Morphology,
    text1: str,
    morph2: Morphology,
    text2: str,
    dim: int = 10000
) -> float:
    """
    Estimate a causal‑effect‑like quantity between two morphology‑text pairs.
    The estimate combines hypervector cosine similarity with the difference
    in recovery priorities.
    """
    hv1 = hybrid_encode(morph1, text1, dim).astype(np.float32)
    hv2 = hybrid_encode(morph2, text2, dim).astype(np.float32)

    sim = cosine_similarity(hv1, hv2)  # in [-1, 1]

    p1 = recovery_priority(morph1)
    p2 = recovery_priority(morph2)

    priority_diff = p2 - p1  # can be negative

    # Linear combination that respects both similarity and priority shift
    estimate = sim + priority_diff
    return estimate

# ---------- Smoke Test ----------
if __name__ == "__main__":
    # Create a simple model pool and two morphology instances
    pool = ModelPool(ram_ceiling_mb=2000)

    mA = Morphology(name="Alpha", length=2.0, width=1.0, height=0.5,
                    mass=3.0, ram_mb=500.0)
    mB = Morphology(name="Beta", length=1.5, width=1.2, height=0.6,
                    mass=2.5, ram_mb=700.0)

    # Load models (will raise if RAM exceeded)
    pool.load(mA)
    pool.load(mB)

    # Define a tiny graph (tree) for testing
    nodes = ["root", "leaf1", "leaf2"]
    edges = [
        ("root", "leaf1", 0.3),
        ("root", "leaf2", 0.6)
    ]
    labels = {("root", "leaf1"): "typeA", ("root", "leaf2"): "typeB"}

    # Run hybrid operations
    encA = hybrid_encode(mA, "sample text A")
    encB = hybrid_encode(mB, "sample text B")
    print("Encoded A (first 10 bits):", encA[:10])
    print("Encoded B (first 10 bits):", encB[:10])

    score = hybrid_tree_score(nodes, edges, labels, mA, "sample text A")
    print("Hybrid tree score (morph A):", score)

    effect = hybrid_effect_estimate(mA, "sample text A", mB, "sample text B")
    print("Hybrid effect estimate (A -> B):", effect)

    # Verify ModelPool RAM accounting
    print("Total RAM used:", pool._used_ram(), "MB")
    sys.exit(0)