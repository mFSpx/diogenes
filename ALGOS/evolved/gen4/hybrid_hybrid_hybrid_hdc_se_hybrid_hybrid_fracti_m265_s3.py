# DARWIN HAMMER — match 265, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hdc_serpentin_hybrid_hybrid_decisi_m162_s1.py (gen3)
# parent_b: hybrid_hybrid_fractional_hd_hybrid_korpus_text_h_m125_s1.py (gen3)
# born: 2026-05-29T23:27:57Z

"""
Hybrid Morphology-Text Hyperdimensional Computing (HMTHDC)

This module fuses the core of **Parent Algorithm A** (morphology → high‑dimensional
vector, shape indices, and entropy) with **Parent Algorithm B** (hyperdimensional
binding, fractional power weighting, min‑hash text encoding).  

The mathematical bridge is the use of a *scalar index* derived from morphology
(e.g. `righting_time_index`) as the exponent in the **fractional power binding**
operator of HDC.  A morphology is first turned into a dense vector (`morphology_vector`);
this vector is then converted to a bipolar hypervector and raised to the exponent
`α = f(morphology)`.  The resulting weighted hypervector is bound (element‑wise
multiplication) with a text‑derived hypervector obtained from a deterministic
min‑hash of the input text.  The bound hypervector encodes both physical shape
information and textual evidence in a single unified representation.

The module provides three high‑level hybrid operations:
* `hybrid_encode(morphology, text)` – produces the fused hypervector.
* `hybrid_similarity(vec1, vec2)` – cosine similarity between two fused vectors.
* `hybrid_effect_estimate(morph1, text1, morph2, text2)` – similarity‑based proxy for
  a causal effect estimate between two morphology‑text pairs.
"""

import re
import math
import random
import hashlib
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Tuple
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Vector = List[float]
HV = np.ndarray  # bipolar hypervector (+1 / -1)

# ---------------------------------------------------------------------------
# Parent A – Morphology utilities
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def morphology_vector(m: Morphology, dim: int = 10000) -> Vector:
    """Deterministic pseudo‑random vector seeded by the SHA‑256 hash of the morphology."""
    seed_bytes = hashlib.sha256(
        f"{m.length}{m.width}{m.height}{m.mass}".encode("utf-8")
    ).digest()[:8]
    seed_int = int.from_bytes(seed_bytes, "big")
    vec = np.array(random_vector(dim, seed_int), dtype=np.float64)

    scaling = np.array([m.length, m.width, m.height, m.mass], dtype=np.float64)
    # Pad scaling to match dimension (repeat pattern)
    repeats = dim // scaling.size + 1
    scaling = np.tile(scaling, repeats)[:dim]

    return (vec * scaling).tolist()


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def shannon_entropy(text: str) -> float:
    counter = Counter(text)
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((cnt / total) * math.log2(cnt / total) for cnt in counter.values())


# ---------------------------------------------------------------------------
# Parent B – Hyperdimensional computing & text utilities (trimmed & completed)
# ---------------------------------------------------------------------------

EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)


def minhash_for_text(text: str, k: int = 64) -> List[int]:
    """
    Deterministic min‑hash signature of the input text.
    """
    cleaned = re.sub(r"\s+", " ", text or "").strip().lower()
    if not cleaned:
        return [0] * k
    shingles = [cleaned[i : i + 5] for i in range(len(cleaned) - 4)]
    signature = []
    for i in range(k):
        min_hash = min(
            int(hashlib.sha256((sh + str(i)).encode("utf-8")).hexdigest(), 16)
            for sh in shingles
        )
        signature.append(min_hash)
    return signature


def random_hv(dim: int = 10000, seed: int | None = None) -> HV:
    """Bipolar (+1 / -1) hypervector."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1, 1], size=dim).astype(np.int8)


def bind(a: HV, b: HV) -> HV:
    """Element‑wise multiplication (binding) for bipolar vectors."""
    return a * b


def unbind(c: HV, b: HV) -> HV:
    """Inverse of bind (same operation for bipolar vectors)."""
    return c * b


def fractional_power(v: HV, exponent: float) -> HV:
    """
    Fractional power binding: sign(v) * |v|**exponent.
    For bipolar vectors |v| == 1, so this reduces to sign(v) * 1 == v,
    but we keep the formulation to allow future real‑valued extensions.
    """
    # Convert to float for exponentiation, then back to int8 bipolar
    float_v = v.astype(np.float64)
    powered = np.sign(float_v) * (np.abs(float_v) ** exponent)
    # Re‑bipolarise (threshold at 0)
    return np.where(powered >= 0, 1, -1).astype(np.int8)


def bundle(vectors: List[HV]) -> HV:
    """Superposition (majority vote) of bipolar vectors."""
    summed = np.sum(vectors, axis=0)
    return np.where(summed >= 0, 1, -1).astype(np.int8)


def similarity(a: HV, b: HV) -> float:
    """Cosine similarity for bipolar vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm != 0 else 0.0


def vector_literal(values: List[int]) -> HV:
    """Convert a list of ints (e.g., minhash) into a bipolar hypervector."""
    dim = len(values)
    # Map even -> -1, odd -> +1 (simple deterministic scheme)
    return np.array([1 if v % 2 else -1 for v in values], dtype=np.int8)


# ---------------------------------------------------------------------------
# Hybrid operations – mathematical bridge
# ---------------------------------------------------------------------------

def morphology_to_hv(m: Morphology, dim: int = 10000) -> HV:
    """
    Convert a Morphology into a bipolar hypervector.
    The dense vector from Parent A is thresholded at its median to obtain bipolar signs.
    """
    dense = np.array(morphology_vector(m, dim), dtype=np.float64)
    median = np.median(dense)
    return np.where(dense >= median, 1, -1).astype(np.int8)


def text_to_hv(text: str, dim: int = 10000) -> HV:
    """
    Encode text into a bipolar hypervector via deterministic min‑hash followed by
    vector_literal (binary mapping) and optional up‑sampling to the target dimension.
    """
    signature = minhash_for_text(text, k=dim // 100)  # produce a smaller signature
    hv = vector_literal(signature)
    # Upsample to requested dimension by repeating pattern
    repeats = dim // hv.size + 1
    hv = np.tile(hv, repeats)[:dim]
    return hv.astype(np.int8)


def hybrid_encode(m: Morphology, text: str, dim: int = 10000) -> HV:
    """
    Core hybrid encoding:
    1. Convert morphology → hv_m.
    2. Compute exponent α = righting_time_index(m).
    3. Apply fractional power: hv_mα = fractional_power(hv_m, α).
    4. Convert text → hv_t.
    5. Bind: hv = bind(hv_mα, hv_t).
    """
    hv_m = morphology_to_hv(m, dim)
    alpha = righting_time_index(m)
    hv_m_weighted = fractional_power(hv_m, alpha)
    hv_t = text_to_hv(text, dim)
    return bind(hv_m_weighted, hv_t)


def hybrid_similarity(vec1: HV, vec2: HV) -> float:
    """
    Cosine similarity between two hybrid hypervectors.
    """
    return similarity(vec1, vec2)


def hybrid_effect_estimate(
    m1: Morphology,
    txt1: str,
    m2: Morphology,
    txt2: str,
    dim: int = 10000,
) -> float:
    """
    A simple proxy for a causal/effect estimate between two (morphology, text)
    pairs: compute the similarity of their hybrid encodings.  Higher similarity
    suggests a stronger inferred relationship.
    """
    hv1 = hybrid_encode(m1, txt1, dim)
    hv2 = hybrid_encode(m2, txt2, dim)
    return hybrid_similarity(hv1, hv2)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Create two simple morphologies
    morph_a = Morphology(length=2.0, width=1.5, height=0.8, mass=3.2)
    morph_b = Morphology(length=2.1, width=1.4, height=0.85, mass=3.0)

    # Sample texts
    text_a = "The specimen was observed with clear evidence of structural integrity."
    text_b = "A verification of the sample shows confirmed stability and no damage."

    # Hybrid encodings
    hv_a = hybrid_encode(morph_a, text_a)
    hv_b = hybrid_encode(morph_b, text_b)

    # Similarity should be a number in [-1, 1]
    sim = hybrid_similarity(hv_a, hv_b)
    print(f"Hybrid similarity between A and B: {sim:.4f}")

    # Effect estimate (just the same similarity here)
    effect = hybrid_effect_estimate(morph_a, text_a, morph_b, text_b)
    print(f"Hybrid effect estimate: {effect:.4f}")

    # Self‑similarity sanity check (should be 1.0)
    self_sim = hybrid_similarity(hv_a, hv_a)
    print(f"Self similarity (should be 1.0): {self_sim:.4f}")