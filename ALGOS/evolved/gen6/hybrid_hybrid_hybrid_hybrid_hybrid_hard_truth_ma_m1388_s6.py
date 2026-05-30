# DARWIN HAMMER — match 1388, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_ternary_route_m928_s1.py (gen5)
# parent_b: hybrid_hard_truth_math_model_pool_m8_s2.py (gen1)
# born: 2026-05-29T23:35:49Z

"""Hybrid algorithm combining binary hyperdimensional routing (Parent A) with
resource‑aware model selection (Parent B).

Mathematical bridge:
- Parent A provides high‑dimensional binary vectors **v**∈{−1,+1}ⁿ, similarity
  measures, and a Fisher‑score based weighting function.
- Parent B defines a low‑dimensional model resource vector **m**∈ℝ² (RAM,
  tier) and evaluates compatibility via a bilinear form **s = v̂ᵀ P m**,
  where **v̂** is a 2‑dimensional summary of the hyperdimensional vector
  (mean and variance) and **P** is a 2×2 projection matrix built from the
  Fisher‑score kernel.

The fusion therefore maps the high‑dimensional representation onto the
model‑space using the Fisher kernel, then scores each model with the
bilinear form. The three core functions below illustrate this pipeline."""
import math
import random
import hashlib
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – hyperdimensional primitives
# ----------------------------------------------------------------------
Vector = List[int]

def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: List[Vector]) -> Vector:
    if not vectors:
        raise ValueError("bundle requires at least one vector")
    dim = len(vectors[0])
    for v in vectors:
        if len(v) != dim:
            raise ValueError("all vectors must have same dimension")
    summed = [sum(comp) for comp in zip(*vectors)]
    return [1 if s >= 0 else -1 for s in summed]

def similarity(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    return dot / len(a)

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """Derivative of the Gaussian beam w.r.t. theta (used as a weighting kernel)."""
    g = gaussian_beam(theta, center, width)
    return - (theta - center) / (width ** 2 + eps) * g

# ----------------------------------------------------------------------
# Parent B – stylometry / model‑resource utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

def words(text: str) -> List[str]:
    import re
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def stylometry_features(text: str) -> np.ndarray:
    """Return a 2‑element vector: [mean_word_length, word_ratio]."""
    w = words(text)
    if not w:
        return np.array([0.0, 0.0])
    mean_len = sum(len(tok) for tok in w) / len(w)
    word_ratio = len(w) / max(len(text.split()), 1)
    return np.array([mean_len, word_ratio], dtype=float)

def lsm_vector(text: str) -> Dict[str, float]:
    """Simple lexical‑semantic‑mapping: frequency per FUNCTION_CATS."""
    w = words(text)
    total = len(w) or 1
    cat_counts = {cat: 0 for cat in FUNCTION_CATS}
    for token in w:
        for cat, vocab in FUNCTION_CATS.items():
            if token in vocab:
                cat_counts[cat] += 1
    return {cat: cnt / total for cat, cnt in cat_counts.items()}

@dataclass
class Model:
    name: str
    ram_gb: float   # scaled RAM
    tier: int       # integer tier

    def resource_vector(self) -> np.ndarray:
        """2‑dimensional resource vector."""
        return np.array([self.ram_gb, float(self.tier)], dtype=float)

# ----------------------------------------------------------------------
# Fusion core – mapping hyperdimensional vectors onto model space
# ----------------------------------------------------------------------
def projection_matrix(center: float = 0.0, width: float = 1.0) -> np.ndarray:
    """
    Build a 2×2 projection matrix P using Fisher‑score kernel values.
    The diagonal entries are Fisher scores evaluated at the two
    canonical axes (0 and 1); off‑diagonal entries are set to zero for
    simplicity.
    """
    p11 = fisher_score(0.0, center, width)
    p22 = fisher_score(1.0, center, width)
    return np.array([[p11, 0.0],
                     [0.0, p22]], dtype=float)

def hyper_summary(vec: Vector) -> np.ndarray:
    """
    Reduce a high‑dimensional binary vector to a 2‑dimensional summary:
    - first component: mean value (range −1…+1)
    - second component: variance (always non‑negative)
    """
    arr = np.array(vec, dtype=float)
    mean = arr.mean()
    var = arr.var()
    return np.array([mean, var], dtype=float)

def compute_hybrid_score(text: str, symbol: str, model: Model,
                         proj_center: float = 0.0,
                         proj_width: float = 1.0) -> float:
    """
    Full hybrid pipeline:
    1. Build a hyperdimensional representation of the *symbol*.
    2. Bind it with a random context vector derived from the *text*.
    3. Summarise the bound vector to a 2‑D statistic vector v̂.
    4. Build projection matrix P from Fisher‑score parameters.
    5. Compute bilinear compatibility s = v̂ᵀ P m.
    """
    # 1‑2: hyperdimensional encoding
    sym_vec = symbol_vector(symbol)
    ctx_vec = random_vector(dim=len(sym_vec), seed=hash(text))
    bound = bind(sym_vec, ctx_vec)

    # 3: 2‑D summary
    v_hat = hyper_summary(bound)

    # 4: projection
    P = projection_matrix(proj_center, proj_width)

    # 5: bilinear form with model resources
    m = model.resource_vector()
    score = float(v_hat @ P @ m)
    return score

def select_best_model(text: str, symbol: str, models: List[Model]) -> Tuple[Model, float]:
    """Return the model with the highest hybrid compatibility score."""
    best = None
    best_score = -math.inf
    for mdl in models:
        sc = compute_hybrid_score(text, symbol, mdl)
        if sc > best_score:
            best_score = sc
            best = mdl
    if best is None:
        raise RuntimeError("No models provided")
    return best, best_score

def hybrid_representation(text: str, symbol: str) -> Vector:
    """
    Produce a bundled hyperdimensional vector that fuses textual context
    (via a random vector seeded by the text) with a symbolic vector.
    """
    sym_vec = symbol_vector(symbol)
    ctx_vec = random_vector(dim=len(sym_vec), seed=hash(text))
    return bundle([sym_vec, ctx_vec])

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The quick brown fox jumps over the lazy dog while the "
        "observer watches silently."
    )
    sample_symbol = "α"

    models = [
        Model(name="Tiny", ram_gb=2.0, tier=1),
        Model(name="Small", ram_gb=8.0, tier=2),
        Model(name="Medium", ram_gb=16.0, tier=3),
        Model(name="Large", ram_gb=32.0, tier=4),
    ]

    # Demonstrate hybrid representation
    hv = hybrid_representation(sample_text, sample_symbol)
    print(f"Hybrid vector length: {len(hv)}; first 10 bits: {hv[:10]}")

    # Compute scores for each model
    for m in models:
        s = compute_hybrid_score(sample_text, sample_symbol, m)
        print(f"Model {m.name:6s} → score {s:.4f}")

    # Select best model
    best_model, best_score = select_best_model(sample_text, sample_symbol, models)
    print(f"\nBest model: {best_model.name} with score {best_score:.4f}")

    # Verify similarity between two symbol vectors (sanity)
    v1 = symbol_vector("α")
    v2 = symbol_vector("β")
    sim = similarity(v1, v2)
    print(f"\nSimilarity between symbols α and β: {sim:.4f}")