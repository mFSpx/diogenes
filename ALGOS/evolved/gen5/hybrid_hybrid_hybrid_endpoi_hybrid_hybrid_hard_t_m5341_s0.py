# DARWIN HAMMER — match 5341, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s4.py (gen4)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s3.py (gen3)
# born: 2026-05-30T00:01:19Z

"""Hybrid Stylometric‑Morphological Circuit Model
================================================

This module fuses the two parent algorithms:

* **Parent A** – ``hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s4.py``  
  provides an :class:`EndpointCircuitBreaker`, a ``Morphology`` data class and
  statistical tools (Fisher score, SSIM) used to adapt a failure threshold.

* **Parent B** – ``hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s3.py``  
  extracts a high‑dimensional stylometric fingerprint from text, builds a
  Voronoi partition of the feature space and represents linguistic function
  categories as Clifford‑algebra blades.

**Mathematical bridge**

Both parents operate on *vectors* in a Euclidean space:

* the stylometric fingerprint is a point **vₛ ∈ ℝⁿ** (n≈96),
* the morphology description can be written as a point **vₘ ∈ ℝ⁴**,
* the Fisher‑score and SSIM are scalar functions of such vectors.

The hybrid algorithm concatenates the two vectors into a single point  

``v = [vₛ , vₘ] ∈ ℝⁿ⁺⁴``  

and uses this unified representation for three coupled operations:

1. **Voronoi assignment** – the nearest seed determines a base failure
   threshold for the circuit breaker.
2. **Clifford blade construction** – the set of present function‑category
   tokens is mapped to a binary blade; blade multiplication provides an
   algebraic signature that is combined with a Gaussian‑beam matrix derived
   from the morphology.
3. **Statistical adaptation** – the Fisher score of ``v`` and the SSIM
   between ``v`` and the chosen seed modify the threshold dynamically.

The result is a single, mathematically coherent system that can be used for
robust decision‑making on textual‑plus‑physical inputs.

"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Set, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology & Circuit Breaker utilities
# ----------------------------------------------------------------------


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


def fisher_score(vector: np.ndarray) -> float:
    """
    Simple proxy for a Fisher score:
    ratio of between‑class variance (here approximated by total variance)
    to within‑class variance (here approximated by mean squared deviation).
    """
    if vector.size == 0:
        return 0.0
    total_var = np.var(vector, ddof=1)
    within_var = np.mean((vector - np.mean(vector)) ** 2)
    return total_var / (within_var + 1e-12)


def ssim_like(x: np.ndarray, y: np.ndarray) -> float:
    """
    Structural‑Similarity‑Index‑like measure for two 1‑D vectors.
    Uses the classic SSIM formula with small constants.
    """
    C1 = 1e-4
    C2 = 9e-4
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x, ddof=1)
    sigma_y = np.var(y, ddof=1)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / (denominator + 1e-12)


# ----------------------------------------------------------------------
# Parent B – Stylometry & Geometric toolkit
# ----------------------------------------------------------------------


FUNCTION_CATS: Dict[str, Set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set("no not never none neither cannot".split()),
}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"\b\w+\b", text)]


def stylometry_features(text: str) -> np.ndarray:
    """
    Build a 96‑dimensional frequency fingerprint.
    For reproducibility the hash of each token decides its bucket.
    """
    tokens = _tokenize(text)
    counts = np.zeros(96, dtype=np.float64)
    for t in tokens:
        idx = int(hashlib.sha256(t.encode()).hexdigest(), 16) % 96
        counts[idx] += 1.0
    # Normalise to unit L2 norm
    norm = np.linalg.norm(counts)
    return counts / (norm + 1e-12)


def extract_function_categories(text: str) -> Set[str]:
    """
    Return the set of function‑category names that appear at least once in the text.
    """
    tokens = set(_tokenize(text))
    present = set()
    for cat, vocab in FUNCTION_CATS.items():
        if tokens & vocab:
            present.add(cat)
    return present


# ----------------------------------------------------------------------
# Hybrid core – unified vector space & Clifford algebra
# ----------------------------------------------------------------------


# Mapping from function category to basis index (for a 6‑dimensional blade)
_CAT_TO_IDX = {cat: i for i, cat in enumerate(sorted(FUNCTION_CATS.keys()))}
_BLADE_DIM = len(_CAT_TO_IDX)


def canonical_blade(categories: Set[str]) -> np.ndarray:
    """
    Represent a set of function categories as a binary blade vector.
    1 at position i ⇔ basis vector e_i is present.
    """
    blade = np.zeros(_BLADE_DIM, dtype=np.int8)
    for cat in categories:
        idx = _CAT_TO_IDX.get(cat)
        if idx is not None:
            blade[idx] = 1
    return blade


def multiply_blades(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Simplified Clifford blade multiplication for orthogonal basis vectors:
    the result is the XOR (mod‑2 sum) of the binary representations.
    """
    return (b1 + b2) % 2


def morphology_vector(m: Morphology) -> np.ndarray:
    """Convert a Morphology instance to a 4‑element float vector."""
    return np.array([m.length, m.width, m.height, m.mass], dtype=np.float64)


def combined_feature_vector(text: str, m: Morphology) -> np.ndarray:
    """
    Concatenate the stylometric fingerprint with the morphology vector,
    yielding a single point in ℝⁿ⁺⁴.
    """
    return np.concatenate([stylometry_features(text), morphology_vector(m)])


# ----------------------------------------------------------------------
# Voronoi partition (seed generation is deterministic)
# ----------------------------------------------------------------------


_SEED_COUNT = 8
_RANDOM_STATE = np.random.RandomState(42)
_SEEDS: np.ndarray = _RANDOM_STATE.randn(_SEED_COUNT, 96 + 4)  # shape (8,100)


def voronoi_partition(point: np.ndarray) -> int:
    """
    Return the index of the nearest seed (0‑based).
    """
    distances = np.linalg.norm(_SEEDS - point, axis=1)
    return int(np.argmin(distances))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def compute_hybrid_score(text: str, m: Morphology) -> Tuple[float, float]:
    """
    Produce a pair ``(adjusted_threshold, similarity)`` where:

    * ``adjusted_threshold`` = base threshold from Voronoi cell
      + int(Fisher score) – int(SSIM * 10)
    * ``similarity`` is the SSIM‑like similarity between the combined vector
      and its assigned seed.
    """
    v = combined_feature_vector(text, m)
    cell = voronoi_partition(v)
    base_threshold = 3 + cell  # simple linear mapping

    f_score = fisher_score(v)
    seed = _SEEDS[cell]
    sim = ssim_like(v, seed)

    adjusted = base_threshold + int(f_score) - int(sim * 10)
    adjusted = max(1, adjusted)  # threshold must stay positive
    return float(adjusted), float(sim)


def hybrid_circuit_breaker(text: str, m: Morphology) -> EndpointCircuitBreaker:
    """
    Instantiate an :class:`EndpointCircuitBreaker` whose failure threshold is
    derived from the hybrid score.  The breaker records a failure if the
    SSIM similarity falls below 0.5, otherwise a success.
    """
    thresh, sim = compute_hybrid_score(text, m)
    cb = EndpointCircuitBreaker(failure_threshold=int(thresh))

    if sim < 0.5:
        cb.record_failure()
    else:
        cb.record_success()
    return cb


def hybrid_algebraic_signature(text: str, m: Morphology) -> np.ndarray:
    """
    Build a Clifford‑algebra signature for the input and combine it with a
    Gaussian‑beam matrix derived from the morphology.

    The Gaussian beam matrix G is a 4×4 covariance‑like matrix with entries
    exp(‑(Δ_i² + Δ_j²)/2σ²) where Δ_i are the normalized morphology components.
    The final signature is ``blade @ G`` (matrix‑vector product).
    """
    categories = extract_function_categories(text)
    blade = canonical_blade(categories).astype(np.float64)

    # Normalise morphology to unit scale
    v_m = morphology_vector(m)
    v_norm = v_m / (np.linalg.norm(v_m) + 1e-12)

    sigma = 0.5
    G = np.exp(- (np.add.outer(v_norm, v_norm) ** 2) / (2 * sigma ** 2))

    # Pad blade to length 4 (since G is 4×4) by truncating or zero‑padding
    if blade.size < 4:
        pad = np.zeros(4 - blade.size, dtype=np.float64)
        blade4 = np.concatenate([blade, pad])
    else:
        blade4 = blade[:4]

    signature = G @ blade4
    return signature


# ----------------------------------------------------------------------
# Demonstration functions (required ≥3)
# ----------------------------------------------------------------------


def demo_hybrid_score():
    """Print the hybrid threshold and similarity for a sample input."""
    txt = "The quick brown fox jumps over the lazy dog while it cannot be ignored."
    morph = Morphology(1.2, 0.8, 0.5, 2.3)
    thresh, sim = compute_hybrid_score(txt, morph)
    print(f"Hybrid adjusted threshold: {thresh:.2f}, SSIM‑like similarity: {sim:.4f}")


def demo_circuit_breaker():
    """Show the state of the circuit breaker after processing a sample."""
    txt = "I cannot deny that the results are significant, and they are not random."
    morph = Morphology(0.9, 0.4, 0.3, 1.1)
    cb = hybrid_circuit_breaker(txt, morph)
    print("Circuit breaker state:", cb.as_dict())


def demo_algebraic_signature():
    """Compute and display the algebraic signature vector."""
    txt = "She and he will go but they might not arrive."
    morph = Morphology(2.0, 1.5, 1.0, 3.0)
    sig = hybrid_algebraic_signature(txt, morph)
    print("Algebraic signature:", sig)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    print("=== Hybrid Stylometric‑Morphological Demo ===")
    demo_hybrid_score()
    demo_circuit_breaker()
    demo_algebraic_signature()
    print("Smoke test completed without errors.")