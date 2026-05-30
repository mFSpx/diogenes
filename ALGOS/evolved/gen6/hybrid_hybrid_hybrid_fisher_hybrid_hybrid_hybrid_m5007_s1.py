# DARWIN HAMMER — match 5007, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_hybrid_m1864_s5.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m1818_s3.py (gen5)
# born: 2026-05-29T23:59:14Z

"""Hybrid Algorithm integrating:
- Parent A: Gaussian beam, Fisher information, SSIM similarity (hybrid_fisher_locali_hybrid_hybrid_m1864_s5)
- Parent B: MinHash Serpentina morphology, bind operation, ternary lens feature count (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m1818_s3)

Mathematical Bridge:
The Fisher information derived from a packet’s textual surface is used as a continuous
scaling factor for the hyper‑dimensional morphology representation.  The SSIM similarity
between two token‑frequency “images” supplies a raw pheromone signal; this signal is
modulated by a Caputo‑style fractional decay evaluated at the current StoreState level.
The resulting pheromone weight multiplies the bind product of the morphology vector
and the feature‑count vector, yielding a unified pheromone‑augmented morphology that can
drive routing/decision making in a single loop.
"""

import sys
import math
import random
import pathlib
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Gaussian beam, Fisher score, SSIM
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index Measure for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))

    numerator = (2 * mx * my + C1) * (2 * cov + C2)
    denominator = (mx * mx + my * my + C1) * (vx + vy + C2)
    return numerator / denominator

# ----------------------------------------------------------------------
# Parent B – MinHash, morphology vector, bind, feature count
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1
Vector = List[float]

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float
    tokens: List[str]

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        hashlib.blake2b(data, digest_size=8).digest(), "big"
    )

def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def morphology_vector(m: Morphology, dim: int = 10000) -> Vector:
    seed_bytes = hashlib.sha256(
        f"{m.length}{m.width}{m.height}{m.mass}".encode("utf-8")
    ).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    random.seed(seed)
    base = np.array([random.random() for _ in range(dim)], dtype=np.float64)
    scale = np.array(
        [m.length, m.width, m.height, m.mass] * (dim // 4 + 1)
    )[:dim]
    return (base * scale).tolist()

def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    return [x * y for x, y in zip(a, b)]

def extract_feature_count_vector(text: str, vocab: List[str] = None) -> List[int]:
    """Count occurrences of each token in *vocab* (or of all words if vocab is None)."""
    words = re.findall(r"\b\w+\b", text.lower())
    counter = Counter(words)
    if vocab is None:
        vocab = sorted(counter.keys())
    return [counter.get(tok, 0) for tok in vocab]

# ----------------------------------------------------------------------
# Fractional decay (Caputo‑style kernel) used as the bridge
# ----------------------------------------------------------------------
def fractional_decay(t: float, alpha: float = 0.5) -> float:
    """Simple Caputo‑like kernel: t^{alpha‑1} / Gamma(alpha).  Returns 0 for t≤0."""
    if t <= 0:
        return 0.0
    return (t ** (alpha - 1)) / math.gamma(alpha)

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_pheromone_signal(
    text_a: str,
    text_b: str,
    theta: float,
    center: float,
    width: float,
    store_level: float,
) -> float:
    """
    Compute a pheromone inflow using:
    - Fisher information (continuous scaling)
    - SSIM similarity between token‑frequency histograms
    - Fractional decay evaluated at the current StoreState level
    """
    # Fisher scaling
    fisher = fisher_score(theta, center, width)

    # Build token‑frequency histograms and treat them as 1‑D “images”
    vocab = sorted(set(re.findall(r"\b\w+\b", (text_a + " " + text_b).lower())))
    hist_a = np.array(extract_feature_count_vector(text_a, vocab), dtype=np.float64)
    hist_b = np.array(extract_feature_count_vector(text_b, vocab), dtype=np.float64)

    # Normalise to dynamic range 255 for SSIM compatibility
    if hist_a.max() == 0 and hist_b.max() == 0:
        ssim_val = 1.0
    else:
        scale = 255.0 / max(hist_a.max(), hist_b.max())
        ssim_val = ssim(hist_a * scale, hist_b * scale)

    decay = fractional_decay(store_level)

    pheromone = fisher * ssim_val * decay
    return pheromone

def hybrid_morphology_update(
    morph: Morphology,
    text: str,
    pheromone: float,
    dim: int = 4096,
) -> Vector:
    """
    Produce a pheromone‑augmented morphology vector:
    bind(morphology_vector, feature_count_vector) * pheromone
    """
    # Base morphology vector
    base_vec = np.array(morphology_vector(morph, dim=dim), dtype=np.float64)

    # Feature‑count vector (projected to same dimensionality by simple repetition)
    vocab = sorted(set(re.findall(r"\b\w+\b", text.lower())))
    feat_counts = np.array(extract_feature_count_vector(text, vocab), dtype=np.float64)
    # repeat or truncate to match dim
    if len(feat_counts) < dim:
        repeats = dim // len(feat_counts) + 1
        feat_counts = np.tile(feat_counts, repeats)[:dim]
    else:
        feat_counts = feat_counts[:dim]

    bound = base_vec * feat_counts
    return (bound * pheromone).tolist()

def hybrid_route_decision(
    candidate_vectors: List[Vector],
    pheromone_vector: Vector,
) -> int:
    """
    Choose the index of the candidate whose cosine similarity with the
    pheromone‑augmented morphology vector is maximal.
    """
    p = np.array(pheromone_vector, dtype=np.float64)
    p_norm = np.linalg.norm(p) + 1e-12
    best_idx = -1
    best_score = -float("inf")
    for idx, cand in enumerate(candidate_vectors):
        c = np.array(cand, dtype=np.float64)
        score = np.dot(p, c) / ((np.linalg.norm(c) + 1e-12) * p_norm)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx

# ----------------------------------------------------------------------
# Simple StoreState representation
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    level: float = 1.0
    dance: float = 0.0

    def update(self, delta: float) -> None:
        """Update level and recompute dance as bounded change."""
        old = self.level
        self.level = max(0.0, self.level + delta)
        self.dance = math.tanh(self.level - old)  # bounded between -1 and 1

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample texts
    txt1 = "Evidence suggests that the particle exhibits wave‑like behavior."
    txt2 = "Observations indicate a dual nature of the quantum entity."

    # StoreState initialization
    state = StoreState(level=2.0)

    # Hybrid pheromone computation
    pher = hybrid_pheromone_signal(
        txt1,
        txt2,
        theta=0.5,
        center=0.0,
        width=1.0,
        store_level=state.level,
    )
    print(f"Pheromone signal: {pher:.6f}")

    # Morphology definition
    morph = Morphology(
        length=1.2,
        width=0.8,
        height=0.5,
        mass=3.4,
        tokens=re.findall(r"\b\w+\b", txt1.lower()),
    )

    # Update morphology with pheromone
    morph_vec = hybrid_morphology_update(morph, txt1, pheromone=pher)
    print(f"Morphology vector norm: {np.linalg.norm(morph_vec):.3f}")

    # Create candidate vectors (randomly for demo)
    candidates = [np.random.rand(len(morph_vec)).tolist() for _ in range(5)]

    # Decision routing
    chosen = hybrid_route_decision(candidates, morph_vec)
    print(f"Chosen candidate index: {chosen}")

    # Update StoreState based on pheromone (example rule)
    state.update(delta=pher * 0.1)
    print(f"New StoreState level: {state.level:.3f}, dance: {state.dance:.3f}")