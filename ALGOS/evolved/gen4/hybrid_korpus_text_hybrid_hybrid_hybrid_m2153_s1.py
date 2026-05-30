# DARWIN HAMMER — match 2153, survivor 1
# gen: 4
# parent_a: korpus_text.py (gen0)
# parent_b: hybrid_hybrid_hybrid_endpoi_shap_attribution_m45_s0.py (gen3)
# born: 2026-05-29T23:41:03Z

"""Hybrid Text-Morphology SHAP Module

This module fuses the core mathematics of **korpus_text.py** (minhash signatures,
Shannon entropy, and quantized text embeddings) with the structures of
**hybrid_hybrid_hybrid_endpoi_shap_attribution_m45_s0.py** (Morphology,
EndpointCircuitBreaker, sphericity index, and SHAP kernel weighting).

Mathematical Bridge
------------------
* The *health* of an `EndpointCircuitBreaker` (open = 0, closed = 1) is used as a
  scalar weight that modulates every text‑derived feature.
* The geometric *sphericity* of a `Morphology` instance provides a second scalar
  that scales the SHAP kernel weight for each feature.
* The resulting hybrid attribution for a feature *i* is:


A_i = w_cb * w_sph * φ_i * K(|S_i|, F)


where
* `w_cb`  – circuit‑breaker health (0 or 1),
* `w_sph` – sphericity index,
* `φ_i`   – the i‑th raw text feature (minhash bit, entropy component, or embedding
            dimension),
* `K`     – SHAP kernel weight `shapley_kernel_weight(|S_i|, F)` with `F` the total
            number of features and `|S_i|` the size of the subset that contains
            feature *i*.

The three functions below illustrate this hybrid operation.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Simplified re‑implementation of Parent A utilities (stand‑alone)
# ----------------------------------------------------------------------
INT16_MAX = 32767  # mimic kernel.mini_embeddings.INT16_MAX


def _simple_shingles(text: str, width: int = 5) -> Set[str]:
    """Generate a set of contiguous substrings (shingles) of given width."""
    cleaned = text.replace("\n", " ").strip()
    return {cleaned[i : i + width] for i in range(len(cleaned) - width + 1)}


def _simple_minhash_signature(shingles: Set[str], k: int = 64) -> List[int]:
    """Produce a minhash signature using k pseudo‑random hash functions."""
    random.seed(0)  # deterministic for reproducibility
    signatures = []
    for _ in range(k):
        a, b = random.randint(1, 1 << 30), random.randint(0, 1 << 30)
        min_hash = min(((a * hash(s) + b) % (1 << 31)) for s in shingles) if shingles else 0
        signatures.append(min_hash)
    return signatures


def _simple_shannon_entropy(seq: List[str]) -> float:
    """Shannon entropy of a sequence of symbols."""
    if not seq:
        return 0.0
    freq = {}
    for s in seq:
        freq[s] = freq.get(s, 0) + 1
    total = len(seq)
    entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
    return entropy


def _simple_hash_quantized_embedding(text: str) -> List[int]:
    """Quantized embedding: hash each character and map to int16 range."""
    return [abs(hash(ch)) % (INT16_MAX + 1) for ch in text]


def minhash_for_text(text: str, k: int = 64) -> List[int]:
    """Parent A – MinHash signature for a text string."""
    return _simple_minhash_signature(_simple_shingles(text.lower(), width=5), k=k)


def entropy_for_text(text: str) -> float:
    """Parent A – Shannon entropy of the first 10 000 characters."""
    return float(_simple_shannon_entropy(list((text or "")[:10000]))) if text else 0.0


def vector_literal(text: str) -> str:
    """Parent A – String representation of a normalized quantized embedding."""
    embedding = _simple_hash_quantized_embedding(text or "")
    normalized = [float(v) / float(INT16_MAX) for v in embedding]
    return "[" + ",".join(f"{x:.8f}" for x in normalized) + "]"


# ----------------------------------------------------------------------
# Parent B structures (re‑implemented where needed)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    geometric_mean = (length * width * height) ** (1.0 / 3.0)
    longest = max(length, width, height)
    return geometric_mean / longest


def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """
    Classic SHAP kernel weight:
        w = (feature_count - 1) / (subset_size * (feature_count - subset_size))
    Handles edge cases where denominator would be zero.
    """
    if subset_size == 0 or subset_size == feature_count:
        return 0.0
    return (feature_count - 1) / (subset_size * (feature_count - subset_size))


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def compute_text_features(text: str) -> Dict[str, Any]:
    """
    Extract raw text features from Parent A:
      - minhash signature (list of ints)
      - Shannon entropy (float)
      - normalized embedding vector (numpy array)
    """
    mh = np.array(minhash_for_text(text), dtype=np.int64)
    ent = entropy_for_text(text)
    embed_str = vector_literal(text)
    # parse the string back to a numpy array for numeric work
    embed_vals = np.fromstring(embed_str.strip("[]"), sep=",", dtype=np.float64)
    return {"minhash": mh, "entropy": ent, "embedding": embed_vals}


def hybrid_feature_vector(
    text: str, morphology: Morphology, circuit_breaker: EndpointCircuitBreaker
) -> np.ndarray:
    """
    Build a single feature vector that fuses:
      * Text‑derived features (minhash bits, entropy, embedding)
      * Morphology‑derived sphericity scalar
      * Circuit‑breaker health scalar (1 = healthy, 0 = open)

    The vector is:
        w_cb * w_sph * [minhash, entropy, embedding]
    """
    feats = compute_text_features(text)
    # concatenate all numeric parts
    raw_vec = np.concatenate(
        [feats["minhash"].astype(np.float64), np.array([feats["entropy"]]), feats["embedding"]]
    )
    w_cb = 1.0 if circuit_breaker.allow() else 0.0
    w_sph = sphericity_index(morphology.length, morphology.width, morphology.height)
    return w_cb * w_sph * raw_vec


def hybrid_shap_attribution(
    text: str, morphology: Morphology, circuit_breaker: EndpointCircuitBreaker
) -> np.ndarray:
    """
    Compute a SHAP‑like attribution for each feature in the hybrid vector.
    For each feature i we use:
        attribution_i = w_cb * w_sph * feature_i * K(|S_i|, F)

    where |S_i| is taken as i+1 (a simple progressive subset size) and
    F is the total number of features.
    """
    vec = hybrid_feature_vector(text, morphology, circuit_breaker)
    F = vec.size
    attributions = np.empty_like(vec)
    for i in range(F):
        subset_size = i + 1  # simple deterministic subset size
        kernel = shapley_kernel_weight(subset_size, F)
        attributions[i] = vec[i] * kernel
    return attributions


def evaluate_hybrid_score(
    text: str, morphology: Morphology, circuit_breaker: EndpointCircuitBreaker
) -> float:
    """
    Produce a scalar health‑score that combines:
        * Average magnitude of the weighted hybrid vector,
        * Morphology sphericity,
        * Circuit‑breaker health.
    The formula is:
        score = (mean|v|) * w_sph * (1 if healthy else 0)
    """
    vec = hybrid_feature_vector(text, morphology, circuit_breaker)
    if vec.size == 0:
        return 0.0
    mean_abs = np.mean(np.abs(vec))
    w_sph = sphericity_index(morphology.length, morphology.width, morphology.height)
    w_cb = 1.0 if circuit_breaker.allow() else 0.0
    return mean_abs * w_sph * w_cb


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "The quick brown fox jumps over the lazy dog."
    sample_morph = Morphology(length=2.5, width=1.2, height=0.8, mass=3.4)

    cb = EndpointCircuitBreaker(failure_threshold=2)
    # Simulate one failure, then a success to keep circuit closed
    cb.record_failure()
    cb.record_success()

    print("=== Text Features ===")
    tf = compute_text_features(sample_text)
    print("Minhash (first 5):", tf["minhash"][:5])
    print("Entropy:", tf["entropy"])
    print("Embedding (first 5):", tf["embedding"][:5])

    print("\n=== Hybrid Vector ===")
    hv = hybrid_feature_vector(sample_text, sample_morph, cb)
    print("Hybrid vector shape:", hv.shape)
    print("First 5 entries:", hv[:5])

    print("\n=== Hybrid SHAP Attribution ===")
    attr = hybrid_shap_attribution(sample_text, sample_morph, cb)
    print("Attribution shape:", attr.shape)
    print("First 5 attributions:", attr[:5])

    print("\n=== Hybrid Score ===")
    score = evaluate_hybrid_score(sample_text, sample_morph, cb)
    print("Score:", score)