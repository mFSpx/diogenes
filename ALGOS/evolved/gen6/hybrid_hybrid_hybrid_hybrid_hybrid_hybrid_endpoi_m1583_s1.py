# DARWIN HAMMER — match 1583, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_semant_hybrid_hybrid_distri_m469_s0.py (gen5)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_krampus_brain_m24_s2.py (gen2)
# born: 2026-05-29T23:37:33Z

"""Hybrid Morphology‑Semantic‑Circuit Fusion

Parents:
- **Parent A** (`hybrid_hybrid_hybrid_semant_hybrid_hybrid_distri_m469_s0.py`): provides
  morphology‑based indices (sphericity, flatness), righting‑time based recovery
  priority, and a cosine‑similarity semantic memory score.
- **Parent B** (`hybrid_hybrid_endpoint_circ_hybrid_krampus_brain_m24_s2.py`): supplies
  an endpoint‑circuit‑breaker health model, a curvature‑modulated health factor,
  and a deterministic 24‑dimensional text feature → 3‑D brain coordinate map.

**Mathematical Bridge**

1. **Recovery priority** `ρ` (from A) quantifies how quickly an organism can
   self‑right.
2. **Health** `h` (from B) combines circuit‑breaker reliability with `ρ`:

   
   h = (1 - failures/threshold) * (1 - ρ)
   

3. **Morphology curvature** `κ` (from B) uses A’s sphericity `σ` and flatness `φ`:

   
   κ = σ * φ
   

4. **Curvature‑modulated factor** `c` injects health into curvature:

   
   c = h * (0.5 + 0.5 * tanh(κ))
   

5. **Semantic memory** `μ` (from A) is the cosine similarity between a document
   embedding and its semantic neighbours.

6. **Hybrid score** `S` multiplies the curvature factor with semantic memory,
   yielding a single scalar that drives the final 3‑D brain mapping:

   
   S = c * μ
   

The module implements this fused pipeline and returns a 3‑D coordinate together
with diagnostic values."""

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared utilities (timestamp)
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Parent A – morphology & semantic core
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Dimension‑based sphericity (unit‑less)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Dimension‑based flatness (unit‑less)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    """Physical righting‑time proxy."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0, 1] derived from righting time."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Robust cosine similarity (returns 0 when either vector is zero)."""
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Both inputs must be 1‑D vectors")
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if den == 0 else float(np.dot(a, b) / den)


def semantic_memory_score(
    doc_embedding: np.ndarray, semantic_neighbors: np.ndarray
) -> float:
    """
    Computes the semantic memory as the cosine similarity between a document
    embedding and the aggregated semantic neighbours vector.
    """
    if doc_embedding.shape != semantic_neighbors.shape:
        raise ValueError("Embedding and neighbours must share shape")
    return cosine_similarity(doc_embedding, semantic_neighbors)


# ----------------------------------------------------------------------
# Parent B – endpoint circuit‑breaker & brain‑map core
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures: int = 0

    def record_failure(self) -> None:
        """Increment failure count."""
        self.failures += 1

    @property
    def is_open(self) -> bool:
        """Breaker is open when failures exceed the threshold."""
        return self.failures >= self.failure_threshold

    def health(self) -> float:
        """Health score in [0, 1] based on remaining budget."""
        return max(0.0, 1.0 - self.failures / self.failure_threshold)


def compute_health(breaker: EndpointCircuitBreaker, morph: Morphology) -> float:
    """
    Combined health factor that mixes circuit‑breaker health with morphology‑derived
    recovery priority.

    health = (breaker health) * (1 - recovery_priority)
    """
    return breaker.health() * (1.0 - recovery_priority(morph))


def curvature_factor(morph: Morphology, health: float) -> float:
    """
    Morphology‑based curvature modulated by health.

    κ = sphericity * flatness
    c = health * (0.5 + 0.5 * tanh(κ))
    """
    sigma = sphericity_index(morph.length, morph.width, morph.height)
    phi = flatness_index(morph.length, morph.width, morph.height)
    kappa = sigma * phi
    return health * (0.5 + 0.5 * math.tanh(kappa))


def text_to_feature_vector(text: str, dim: int = 24) -> np.ndarray:
    """
    Deterministic 24‑dimensional feature vector derived from SHA‑256 hash.
    Values are normalised to the interval [0, 1].
    """
    if dim <= 0:
        raise ValueError("dim must be positive")
    # Produce enough bytes (dim * 4 bytes = 96 bytes) by re‑hashing iteratively.
    raw = text.encode("utf-8")
    digest = b""
    while len(digest) < dim * 4:
        h = hashlib.sha256(raw).digest()
        digest += h
        raw = h  # chain hash
    # Convert each 4‑byte chunk to a float in [0,1)
    floats = [
        int.from_bytes(digest[i : i + 4], "big") / 0xFFFFFFFF
        for i in range(0, dim * 4, 4)
    ]
    return np.array(floats[:dim], dtype=float)


def brain_xyz(feature_vec: np.ndarray, curvature: float) -> Tuple[float, float, float]:
    """
    Maps a 24‑dimensional feature vector to a 3‑D coordinate.
    The first three components are scaled by the curvature factor and passed
    through a smooth non‑linearity to keep coordinates bounded.

    xyz_i = tanh(curvature * v_i)   for i = 0,1,2
    """
    if feature_vec.shape[0] < 3:
        raise ValueError("feature vector must have at least 3 dimensions")
    scaled = curvature * feature_vec[:3]
    return tuple(math.tanh(v) for v in scaled)


# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_brain_map(
    text: str,
    morph: Morphology,
    doc_embedding: np.ndarray,
    semantic_neighbors: np.ndarray,
    breaker: EndpointCircuitBreaker,
) -> Dict[str, Any]:
    """
    End‑to‑end hybrid computation:

    1. Compute morphology‑derived recovery priority.
    2. Compute circuit‑breaker health and combine (health factor).
    3. Derive curvature factor from health and morphology.
    4. Compute semantic memory (cosine similarity).
    5. Fuse curvature and semantic memory into a hybrid score S.
    6. Generate a deterministic 24‑D feature vector from *text*.
    7. Produce a 3‑D brain coordinate using S as the curvature argument.

    Returns a dictionary with diagnostics and the final coordinate.
    """
    # Step 1 & 2
    health = compute_health(breaker, morph)

    # Step 3
    curv = curvature_factor(morph, health)

    # Step 4
    sem_mem = semantic_memory_score(doc_embedding, semantic_neighbors)

    # Step 5 – hybrid scalar that modulates the brain map
    hybrid_score = curv * sem_mem

    # Step 6
    feat_vec = text_to_feature_vector(text, dim=24)

    # Step 7
    coordinate = brain_xyz(feat_vec, curvature=hybrid_score)

    return {
        "timestamp": now_z(),
        "recovery_priority": recovery_priority(morph),
        "breaker_health": breaker.health(),
        "combined_health": health,
        "curvature_factor": curv,
        "semantic_memory": sem_mem,
        "hybrid_score": hybrid_score,
        "brain_coordinate": coordinate,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a representative morphology
    morph = Morphology(length=1.2, width=0.8, height=0.5, mass=3.4)

    # Random embeddings (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    doc_emb = rng.random(128).astype(float)
    neigh_emb = rng.random(128).astype(float)

    # Circuit breaker with a couple of simulated failures
    breaker = EndpointCircuitBreaker(failure_threshold=5)
    for _ in range(2):
        breaker.record_failure()

    # Sample text
    sample_text = "The quick brown fox jumps over the lazy dog."

    # Run the hybrid pipeline
    result = hybrid_brain_map(
        text=sample_text,
        morph=morph,
        doc_embedding=doc_emb,
        semantic_neighbors=neigh_emb,
        breaker=breaker,
    )

    # Pretty‑print the outcome
    for key, value in result.items():
        print(f"{key}: {value}")

    # Exit status
    sys.exit(0)