# DARWIN HAMMER — match 519, survivor 3
# gen: 4
# parent_a: hybrid_rectified_flow_hybrid_hybrid_hard_t_m184_s0.py (gen3)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_krampus_brain_m24_s2.py (gen2)
# born: 2026-05-29T23:29:30Z

import numpy as np
import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# 1. Lexical function categories (unchanged)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

# ----------------------------------------------------------------------
# 2. Utility functions
# ----------------------------------------------------------------------
def words(text: str) -> List[str]:
    """Tokenise a string into alphabetic lower‑case words."""
    return [w for w in (text or "").lower().split() if w.isalpha()]

def lsm_vector(text: str) -> Dict[str, float]:
    """
    Compute a “lexical style matrix” (LSM) vector.
    Returns a probability for each FUNCTION_CATS entry; missing categories get 0.0.
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    lsm: Dict[str, float] = {}
    for cat, vocab in FUNCTION_CATS.items():
        lsm[cat] = sum(cnt[w] for w in vocab) / total
    return lsm

def stable_hash(text: str) -> int:
    """Deterministic 64‑bit hash based on SHA‑256."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)

# ----------------------------------------------------------------------
# 3. Endpoint health subsystem (fixed & stateful)
# ----------------------------------------------------------------------
@dataclass
class EndpointCircuitBreaker:
    """Tracks failures and yields a bounded health score."""
    failure_threshold: int = 3
    failures: int = 0
    _last_endpoint_len: int = field(default=0, init=False, repr=False)

    def record_failure(self) -> None:
        self.failures = min(self.failures + 1, self.failure_threshold)

    def reset(self) -> None:
        self.failures = 0

    def compute_health(self, endpoint: str, breaker: str) -> float:
        """
        Health ∈ [0, 1].
        - Failure penalty: linear decay to 0 at the threshold.
        - Length penalty: ratio of breaker length to endpoint length, capped at 1.
        """
        if not endpoint:
            return 0.0

        failure_factor = max(0.0, 1.0 - self.failures / self.failure_threshold)
        length_ratio = min(1.0, len(breaker) / len(endpoint))
        health = failure_factor * (1.0 - length_ratio)

        # Clamp for numerical safety
        return float(np.clip(health, 0.0, 1.0))

# ----------------------------------------------------------------------
# 4. Curvature & morphology integration (more principled)
# ----------------------------------------------------------------------
def compute_curvature_score(morph: str, health: float) -> float:
    """
    Returns a curvature modulation ∈ [0, 1].
    Uses a normalized “sphericity” measure and a smooth tanh blend.
    """
    if not morph:
        return 0.0

    sphericity = len(morph) / (1.0 + len(morph))          # ∈ (0,1)
    flatness = 1.0 - sphericity                          # complementary
    raw_curvature = sphericity * flatness               # peaks at sphericity=0.5
    curvature = health * (0.5 + 0.5 * math.tanh(4.0 * raw_curvature))
    return float(np.clip(curvature, 0.0, 1.0))

# ----------------------------------------------------------------------
# 5. Minimal Kolmogorov‑Arnold Network (KAN) layer
# ----------------------------------------------------------------------
class SimpleKAN:
    """
    A toy KAN that maps an input vector (size = n_features) to an output
    vector (size = n_out) via piecewise‑linear basis functions.
    Parameters are randomly initialised; in a real system they would be learned.
    """

    def __init__(self, n_features: int, n_out: int, n_knots: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.n_out = n_out
        self.n_knots = n_knots

        # Knot positions are shared across features for simplicity
        self.knots = np.linspace(0.0, 1.0, n_knots)

        # Linear coefficients for each (feature, knot, out) triple
        self.coeffs = rng.uniform(-1.0, 1.0, size=(n_features, n_knots, n_out))

    def _basis(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the piecewise‑linear basis matrix B ∈ ℝ^{n_features×n_knots}
        for a column vector x ∈ [0,1]^{n_features}.
        """
        # Broadcast x against knots → shape (n_features, n_knots)
        diff = x[:, None] - self.knots[None, :]
        # Linear interpolation weight (triangular hat)
        basis = np.maximum(0.0, 1.0 - np.abs(diff) * self.n_knots)
        return basis

    def __call__(self, x_dict: Dict[str, float]) -> np.ndarray:
        """
        Forward pass: input dict → ordered numpy vector → KAN output.
        Output is a 3‑dimensional coordinate.
        """
        # Preserve deterministic ordering of categories
        ordered_vals = np.array([x_dict[cat] for cat in FUNCTION_CATS.keys()], dtype=float)
        # Clamp to [0,1] (LSM already does this, but safety first)
        ordered_vals = np.clip(ordered_vals, 0.0, 1.0)

        B = self._basis(ordered_vals)                     # (n_features, n_knots)
        # Tensor contraction: (n_features, n_knots) · (n_features, n_knots, n_out)
        # → sum over feature & knot axes → (n_out,)
        out = np.tensordot(B, self.coeffs, axes=([0, 1], [0, 1]))
        return out

# ----------------------------------------------------------------------
# 6. Core fusion pipeline (deepened integration)
# ----------------------------------------------------------------------
# Global KAN instance – deterministic seed ensures reproducibility
KAN_LAYER = SimpleKAN(n_features=len(FUNCTION_CATS), n_out=3, seed=42)

def hybrid_brain_map(text: str, endpoint: str, breaker: str, morph: str) -> Tuple[float, float, float]:
    """
    Compute a 3‑D brain coordinate from:
    - LSM of `text`
    - Health derived from endpoint/breaker
    - Morphology‑curvature modulation
    The KAN mixes all three signals in a non‑linear fashion.
    """
    lsm = lsm_vector(text)
    health = EndpointCircuitBreaker().compute_health(endpoint, breaker)
    curvature = compute_curvature_score(morph, health)

    # Blend curvature into each LSM entry (acts as a global gain)
    blended = {cat: val * curvature for cat, val in lsm.items()}

    # KAN produces raw coordinates; we scale them to a unit cube for interpretability
    raw_coords = KAN_LAYER(blended)
    coords = np.tanh(raw_coords)                     # squash to (‑1,1)
    return tuple(coords.tolist())

def _sample_word_by_category(words_pool: List[str], prob: float) -> List[str]:
    """Return a subset of `words_pool` where each word is kept with probability `prob`."""
    return [w for w in words_pool if random.random() < prob]

def hybrid_text_generator(
    source_text: str,
    target_text: str,
    endpoint: str,
    breaker: str,
    morph: str,
    seed: int = 0,
) -> str:
    """
    Produce a synthetic sentence that interpolates between the LSMs of
    `source_text` and `target_text`. The interpolation factor is driven by
    curvature‑modulated health, and the final word selection respects the
    categorical probabilities via a simple stochastic sampler.
    """
    random.seed(seed)
    np.random.seed(seed)

    src_lsm = lsm_vector(source_text)
    tgt_lsm = lsm_vector(target_text)

    health = EndpointCircuitBreaker().compute_health(endpoint, breaker)
    curvature = compute_curvature_score(morph, health)

    # Linear interpolation in LSM space, weighted by curvature
    interp_lsm = {
        cat: src_lsm[cat] + curvature * (tgt_lsm[cat] - src_lsm[cat])
        for cat in FUNCTION_CATS
    }

    # Build a pool of candidate words from both source and target
    src_words = words(source_text)
    tgt_words = words(target_text)
    candidate_pool = src_words + tgt_words

    # Group candidates by category for more faithful style transfer
    categorized: Dict[str, List[str]] = {cat: [] for cat in FUNCTION_CATS}
    for w in candidate_pool:
        for cat, vocab in FUNCTION_CATS.items():
            if w in vocab:
                categorized[cat].append(w)

    # Sample words per category according to the interpolated probabilities
    selected_words: List[str] = []
    for cat, prob in interp_lsm.items():
        selected_words.extend(_sample_word_by_category(categorized[cat], prob))

    # If sampling yields nothing, fall back to a minimal deterministic output
    if not selected_words:
        selected_words = ["generated", "text"]

    # Shuffle to avoid category‑order bias while keeping reproducibility
    random.shuffle(selected_words)
    return " ".join(selected_words)

# ----------------------------------------------------------------------
# 7. Demonstration entry‑point (kept for manual testing)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    src = "The quick brown fox jumps over the lazy dog."
    tgt = "A diligent programmer writes clean and efficient code."
    endpoint = "api/v1/resource"
    breaker = "timeout"
    morph = "neuroplasticity"

    x, y, z = hybrid_brain_map(src, endpoint, breaker, morph)
    print(f"Brain coordinates: ({x:.4f}, {y:.4f}, {z:.4f})")

    gen = hybrid_text_generator(src, tgt, endpoint, breaker, morph, seed=123)
    print(f"Generated text: {gen}")