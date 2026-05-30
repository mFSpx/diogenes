# DARWIN HAMMER — match 3912, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_doomsd_m857_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1673_s4.py (gen5)
# born: 2026-05-29T23:52:33Z

import numpy as np
import hashlib
import math
import random
import string
from dataclasses import dataclass, field
from typing import Dict, Set, Sequence, Tuple, List

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Return numerator/denominator, falling back to *fallback* if denominator is zero."""
    return numerator / denominator if denominator != 0 else fallback


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1] for non‑negative vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity centred at *center* with standard deviation *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a single‑parameter Gaussian model.

    I(θ) = (∂ℓ/∂θ)² / ℓ, where ℓ = Gaussian intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def gaussian_filter(data: np.ndarray, sigma: float) -> np.ndarray:
    """1‑D Gaussian smoothing with standard deviation *sigma*."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = int(math.ceil(3 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(data, kernel, mode="same")


def _hash_to_seed(s: str) -> int:
    """Deterministic integer seed from a short string."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    # Use first 8 bytes as a 64‑bit integer
    return int.from_bytes(h[:8], "big", signed=False)


def _bipolar_hypervector(dim: int, seed: int) -> np.ndarray:
    """Generate a bipolar (+1 / -1) hypervector of dimension *dim* using *seed*."""
    rng = random.Random(seed)
    return np.array([1 if rng.random() < 0.5 else -1 for _ in range(dim)], dtype=np.int8)


# ----------------------------------------------------------------------
# Core algorithm
# ----------------------------------------------------------------------
@dataclass
class HybridStylometry:
    """
    A deeper fusion of stylometry, Bayesian‑style feature extraction,
    hyperdimensional computing and Gaussian‑beam filtering.

    The class is deliberately stateless apart from the hypervector
    dictionary, which is generated once per instance for reproducibility.
    """

    function_cats: Dict[str, Set[str]]
    dim: int = 10_000                     # dimensionality of hypervectors
    sigma: float = 1.0                    # smoothing parameter for Gaussian filter
    beam_center: float = 0.5              # centre of Gaussian beam
    beam_width: float = 0.1               # width of Gaussian beam
    reference_hv: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # Build deterministic hypervectors for each function category
        self._cat_hvs: Dict[str, np.ndarray] = {
            cat: _bipolar_hypervector(self.dim, _hash_to_seed(cat))
            for cat in self.function_cats
        }
        # Reference hypervector = majority vote of all category hypervectors
        stacked = np.stack(list(self._cat_hvs.values()))
        self.reference_hv = np.where(stacked.mean(axis=0) >= 0, 1, -1).astype(np.int8)

    # ------------------------------------------------------------------
    # Text preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_text(text: str) -> str:
        """Lower‑case and strip punctuation."""
        translator = str.maketrans("", "", string.punctuation)
        return text.lower().translate(translator)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Very light tokeniser – split on whitespace after cleaning."""
        return text.split()

    # ------------------------------------------------------------------
    # Stylometry feature extraction
    # ------------------------------------------------------------------
    def _category_proportions(self, tokens: List[str]) -> np.ndarray:
        """
        Return a vector *p* where p[i] = proportion of tokens belonging to
        the i‑th function category. Zero‑division is guarded.
        """
        total = len(tokens)
        if total == 0:
            return np.zeros(len(self.function_cats), dtype=float)

        proportions = np.empty(len(self.function_cats), dtype=float)
        for i, (cat, vocab) in enumerate(self.function_cats.items()):
            count = sum(1 for t in tokens if t in vocab)
            proportions[i] = _safe_div(count, total)
        return proportions

    # ------------------------------------------------------------------
    # Hyperdimensional encoding
    # ------------------------------------------------------------------
    def _encode_hypervector(self, proportions: np.ndarray) -> np.ndarray:
        """
        Weighted sum of category hypervectors, followed by a sign‑binarisation.
        This yields a bipolar hypervector representing the whole document.
        """
        weighted = np.zeros(self.dim, dtype=float)
        for weight, (cat, hv) in zip(proportions, self._cat_hvs.items()):
            weighted += weight * hv
        return np.where(weighted >= 0, 1, -1).astype(np.int8)

    # ------------------------------------------------------------------
    # Scoring utilities
    # ------------------------------------------------------------------
    def _similarity_score(self, doc_hv: np.ndarray) -> float:
        """Cosine similarity between document hypervector and reference."""
        # Convert bipolar to float for cosine computation
        return _cosine_similarity(doc_hv.astype(float), self.reference_hv.astype(float))

    def _smooth_score(self, score: float) -> float:
        """
        Apply a 1‑D Gaussian filter to a scalar by treating it as a length‑1
        signal. This is mathematically equivalent to an identity operation,
        but we keep the call for API symmetry and future extension.
        """
        return float(gaussian_filter(np.array([score]), self.sigma)[0])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def hybrid_beam(self, text: str) -> float:
        """
        Compute the Gaussian‑beam feature for *text*.
        Steps:
          1. Clean & tokenise.
          2. Compute category proportions.
          3. Encode a hypervector.
          4. Measure similarity to reference.
          5. Smooth the similarity (optional, currently identity).
          6. Apply Gaussian beam.
        """
        cleaned = self._clean_text(text)
        tokens = self._tokenize(cleaned)
        props = self._category_proportions(tokens)
        doc_hv = self._encode_hypervector(props)
        sim = self._similarity_score(doc_hv)
        smooth_sim = self._smooth_score(sim)
        return gaussian_beam(smooth_sim, self.beam_center, self.beam_width)

    def hybrid_fisher(self, text: str) -> float:
        """
        Compute the Fisher‑information based score for *text*,
        using the same pipeline as :meth:`hybrid_beam` but ending with
        :func:`fisher_score`.
        """
        cleaned = self._clean_text(text)
        tokens = self._tokenize(cleaned)
        props = self._category_proportions(tokens)
        doc_hv = self._encode_hypervector(props)
        sim = self._similarity_score(doc_hv)
        smooth_sim = self._smooth_score(sim)
        return fisher_score(smooth_sim, self.beam_center, self.beam_width)

    # ------------------------------------------------------------------
    # Convenience wrapper
    # ------------------------------------------------------------------
    def evaluate(self, text: str) -> Tuple[float, float]:
        """
        Return a tuple ``(beam_score, fisher_score)`` for *text*.
        """
        return self.hybrid_beam(text), self.hybrid_fisher(text)


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    FUNCTION_CATS: Dict[str, Set[str]] = {
        "pronoun": {
            "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
            "he", "him", "his", "himself", "she", "her", "hers", "herself",
            "they", "them", "their", "theirs", "themselves", "we", "us", "our",
            "ours", "ourselves"
        },
        "article": {"a", "an", "the"},
        "preposition": {
            "about", "above", "after", "against", "around", "as", "at", "before",
            "behind", "below", "between", "by", "during", "for", "from", "in",
            "into", "of", "off", "on", "onto", "over", "through", "to", "under",
            "with", "without"
        },
        "auxiliary": {
            "am", "are", "be", "been", "being", "can", "could", "did", "do",
            "does", "had", "has", "have", "is", "may", "might", "must", "shall",
            "should", "was", "were", "will", "would"
        },
        "conjunction": {
            "and", "but", "or", "nor", "so", "yet", "because", "although",
            "while", "if", "when", "where", "whereas", "unless", "until"
        },
        "negation": {
            "no", "not", "never", "none", "neither", "cannot", "can't", "won't",
            "don't", "didn't", "isn't", "aren't", "wasn't", "weren't"
        },
        "quantifier": set(),
    }

    hybrid = HybridStylometry(
        function_cats=FUNCTION_CATS,
        dim=8000,
        sigma=1.5,
        beam_center=0.5,
        beam_width=0.12,
    )

    sample = "This is a sample text for testing the hybrid algorithm."
    beam, fisher = hybrid.evaluate(sample)
    print(f"Beam score   : {beam:.6f}")
    print(f"Fisher score : {fisher:.6f}")