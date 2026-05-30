# DARWIN HAMMER — match 2427, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_privacy_sketc_hybrid_fractional_hd_m1084_s1.py (gen2)
# parent_b: hybrid_hybrid_hdc_serpentin_hybrid_hybrid_decisi_m162_s1.py (gen3)
# born: 2026-05-29T23:42:23Z

"""Hybrid CMS‑HDC‑Morphology Module
---------------------------------
Parents:
- hybrid_privacy_sketches_m15_s2.py (Count‑Min Sketch + complex hypervector encoding)
- hybrid_hdc_serpentin_hybrid_hybrid_decisi_m162_s1.py (Morphology‑driven vector generation,
  entropy utilities and textual hygiene scoring)

Mathematical Bridge
~~~~~~~~~~~~~~~~~~
Both parents operate on high‑dimensional vectors:

* Parent A converts a Count‑Min Sketch (CMS) into a *complex* hypervector
  `cms_hv` by assigning a random unit‑magnitude complex vector to each
  (row, column) token and aggregating them weighted by the cell counts.

* Parent B builds a *real‑valued* vector from a `Morphology` instance by seeding a
  pseudo‑random generator with a hash of the morphological parameters and then
  scaling the random components with those parameters.

The bridge is a **binding** operation defined in Parent A (element‑wise
multiplication) together with a **fractional‑power** modulation.  We extend the
binding to fuse three sources:


combined_hv = bind( cms_hv ⊙ morph_hv , causal_hv , α )


where `⊙` is element‑wise multiplication, `causal_hv` is an arbitrary
complex hypervector representing a treatment/policy, and `α ∈ ℝ` is the
fractional power exponent.  The resulting hypervector is used to modulate a
privacy‑risk score derived from the CMS (unique‑identifier ratio) via the
cosine similarity between `combined_hv` and `causal_hv`.

The module provides three representative hybrid functions:
1. `cms_to_hv` – CMS → complex hypervector.
2. `morphology_to_hv` – Morphology → complex hypervector.
3. `hybrid_risk_with_causal_effect` – blended risk estimate using the bound
   hypervector.
"""

import hashlib
import random
import math
import sys
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Dict, Set

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – Count‑Min Sketch utilities (adapted)
# ---------------------------------------------------------------------------

def _cms_hash(item: str, depth: int, width: int) -> List[int]:
    """Return a list of column indices, one per hash row."""
    return [
        int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
        for d in range(depth)
    ]

class CountMinSketch:
    """Simple Count‑Min Sketch with optional tracking of uniques."""
    def __init__(self, depth: int = 5, width: int = 2000):
        self.depth = depth
        self.width = width
        self.table = np.zeros((depth, width), dtype=np.int64)
        self.total_updates: int = 0
        self._unique_items: Set[str] = set()

    def update(self, item: str, inc: int = 1) -> None:
        cols = _cms_hash(item, self.depth, self.width)
        for r, c in enumerate(cols):
            self.table[r, c] += inc
        self.total_updates += inc
        self._unique_items.add(item)

    def estimate(self, item: str) -> int:
        cols = _cms_hash(item, self.depth, self.width)
        return min(self.table[r, c] for r, c in enumerate(cols))

    @property
    def unique_count(self) -> int:
        return len(self._unique_items)

# ---------------------------------------------------------------------------
# Parent B – Morphology utilities (adapted)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def _seed_from_morphology(m: Morphology) -> int:
    """Derive a deterministic integer seed from morphology attributes."""
    raw = f"{m.length:.6f}:{m.width:.6f}:{m.height:.6f}:{m.mass:.6f}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:16], 16)

def random_vector(dim: int = 10000, seed: int | str | None = None) -> np.ndarray:
    """Return a real‑valued random vector in [0,1)."""
    rng = random.Random(seed)
    return np.fromiter((rng.random() for _ in range(dim)), dtype=np.float64)

def morphology_to_real_vector(m: Morphology, dim: int = 10000) -> np.ndarray:
    """Generate a morphology‑conditioned real vector."""
    seed = _seed_from_morphology(m)
    vec = random_vector(dim, seed)
    # Scale by normalized morphological factors
    factors = np.array([m.length, m.width, m.height, m.mass], dtype=np.float64)
    # Pad to `dim` length (repeat the four factors cyclically)
    repeats = dim // 4
    scaling = np.tile(factors, repeats)
    vec = vec * scaling
    return vec

# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------

def cms_to_hv(cms: CountMinSketch, dim: int = 10000) -> np.ndarray:
    """
    Convert a Count‑Min Sketch into a unit‑magnitude complex hypervector.
    Each (row, col) token receives a deterministic random phase; the token
    hypervectors are summed weighted by the cell counts and finally normalized.
    """
    rng = np.random.default_rng(0)  # deterministic seed for reproducibility
    phases = rng.random((cms.depth, cms.width)) * 2 * math.pi
    token_hv = np.exp(1j * phases)  # complex unit vectors

    weighted_sum = np.zeros(dim, dtype=np.complex128)

    # Project each token hypervector onto a fixed `dim`‑dimensional space
    # using a simple linear projection (first `dim` components, repeat if needed).
    proj_len = token_hv.shape[0] * token_hv.shape[1]
    repeats = dim // proj_len + 1

    flat_tokens = token_hv.ravel()
    proj_matrix = np.tile(flat_tokens, repeats)[:dim]

    for r in range(cms.depth):
        for c in range(cms.width):
            cnt = cms.table[r, c]
            if cnt == 0:
                continue
            # Use the same projection for every token; scale by count.
            weighted_sum += cnt * proj_matrix

    # Normalize to unit magnitude (L2 norm = 1)
    norm = np.linalg.norm(weighted_sum)
    if norm == 0:
        return np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    return weighted_sum / norm

def morphology_to_hv(m: Morphology, dim: int = 10000) -> np.ndarray:
    """
    Encode a `Morphology` instance as a complex hypervector.
    The real vector from `morphology_to_real_vector` is interpreted as a phase
    (scaled to [0, 2π]) and combined with a unit magnitude.
    """
    real_vec = morphology_to_real_vector(m, dim)
    # Map real values to phases in [0, 2π)
    phases = (real_vec - real_vec.min()) / (real_vec.ptp() + 1e-12) * 2 * math.pi
    return np.exp(1j * phases)

def bind_hv(hv_a: np.ndarray, hv_b: np.ndarray, exponent: float = 1.0) -> np.ndarray:
    """
    Bind two complex hypervectors using element‑wise multiplication and apply
    a fractional‑power exponent to the phase of the result.
    """
    bound = hv_a * hv_b
    # Apply fractional power: z' = exp(α * log(z))
    log_z = np.log(bound)
    powered = np.exp(exponent * log_z)
    # Re‑normalize to unit length
    norm = np.linalg.norm(powered)
    if norm == 0:
        return powered
    return powered / norm

def cosine_similarity(hv1: np.ndarray, hv2: np.ndarray) -> float:
    """Cosine similarity for complex vectors (real part of inner product)."""
    dot = np.vdot(hv1, hv2).real
    norm = np.linalg.norm(hv1) * np.linalg.norm(hv2)
    return dot / norm if norm != 0 else 0.0

def hybrid_risk_with_causal_effect(
    cms: CountMinSketch,
    morph: Morphology,
    causal_hv: np.ndarray,
    exponent: float = 1.0,
    dim: int = 10000,
) -> float:
    """
    Compute a blended privacy‑risk estimate.
    - `privacy_ratio` = unique identifiers / total updates (from CMS).
    - `cms_hv` and `morph_hv` are bound together, then bound to `causal_hv`
      with fractional exponent `exponent`.
    - The final risk = privacy_ratio * (1 - similarity), where similarity is
      the cosine similarity between the combined hypervector and the causal
      hypervector.
    """
    if cms.total_updates == 0:
        privacy_ratio = 0.0
    else:
        privacy_ratio = cms.unique_count / cms.total_updates

    cms_hv = cms_to_hv(cms, dim)
    morph_hv = morphology_to_hv(morph, dim)

    # First bind CMS and morphology, then bind to causal with exponent
    intermediate = cms_hv * morph_hv
    combined = bind_hv(intermediate, causal_hv, exponent)

    similarity = cosine_similarity(combined, causal_hv)
    blended_risk = privacy_ratio * (1.0 - similarity)
    return blended_risk

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a tiny CMS and insert sample items
    cms = CountMinSketch(depth=4, width=256)
    items = ["alice", "bob", "alice", "carol", "dave", "bob", "eve", "alice"]
    for it in items:
        cms.update(it)

    # Define a morphology instance
    morph = Morphology(length=2.5, width=1.8, height=0.9, mass=0.45)

    # Create a causal hypervector (random complex unit vector)
    rng = np.random.default_rng(42)
    phases = rng.random(10000) * 2 * math.pi
    causal_hv = np.exp(1j * phases)

    # Compute hybrid risk
    risk = hybrid_risk_with_causal_effect(
        cms=cms,
        morph=morph,
        causal_hv=causal_hv,
        exponent=0.75,
        dim=10000,
    )
    print(f"Hybrid risk estimate: {risk:.6f}")

    # Simple sanity checks
    assert 0.0 <= risk <= 1.0, "Risk should be within [0,1]"
    print("Smoke test completed successfully.")