# DARWIN HAMMER — match 2229, survivor 0
# gen: 6
# parent_a: hybrid_geometric_product_hybrid_hybrid_fisher_m1171_s1.py (gen3)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_hybrid_m515_s1.py (gen5)
# born: 2026-05-29T23:41:33Z

"""Hybrid Geometric‑Linguistic Scoring Module

Parents:
- hybrid_geometric_product_hybrid_hybrid_fisher_m1171_s1 (geometric product via Clifford algebra,
  Bayesian/Fisher‑information based tree scoring)
- hybrid_hybrid_hard_truth_ma_hybrid_hybrid_hybrid_m515_s1 (lexical‑statistical‑model (LSM) scoring
  with posterior edge belief)

Mathematical Bridge:
Both parents manipulate probability‑like quantities.  In the linguistic side a word‑frequency
distribution 𝑝⃗ is a vector in ℝ^|V| (V = vocabulary).  In the geometric‑algebra side a multivector
can encode the same distribution by assigning each vocabulary token to a distinct basis blade.
The geometric product of two such multivectors corresponds to a bilinear interaction of the
underlying distributions.  The Fisher information of a distribution is a scalar that measures
its intrinsic uncertainty; we embed this scalar into the geometric product by scaling the
resulting multivector.  The hybrid score is therefore

    S(text, edge_belief) = ⟨ MV(p_text) ⊙ MV(edge_belief) ⟩  ·  I_F(p_text)

where ⊙ is the geometric product, ⟨·⟩ extracts the scalar (grade‑0) part, and I_F is the
Fisher‑information scalar of the text distribution.  This single expression fuses the core
topologies of both parents.
"""

import math
import random
import sys
import pathlib
from datetime import datetime
from collections import Counter
import re
from typing import Dict, List, Tuple, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Geometric Algebra core (Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades.

    components: dict mapping frozenset(basis_indices) -> float coefficient.
                frozenset() is the scalar (grade‑0) blade.
    n: dimension of the base vector space.
    """

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result: Dict[FrozenSet[int], float] = {}
        for k1, v1 in self.components.items():
            for k2, v2 in other.components.items():
                combined, sign = self._multiply_blades(k1, k2)
                result[combined] = result.get(combined, 0.0) + sign * v1 * v2
        return Multivector(result, self.n)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = self.components.copy()
        for k, v in other.components.items():
            result[k] = result.get(k, 0.0) + v
        return Multivector(result, self.n)

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) coefficient."""
        return self.components.get(frozenset(), 0.0)

    # ------------------------------------------------------------------
    # internal blade handling
    # ------------------------------------------------------------------
    def _multiply_blades(self, blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
        combined = list(blade_a) + list(blade_b)
        result, sign = self._blade_sign(combined)
        return frozenset(result), sign

    def _blade_sign(self, indices: List[int]) -> Tuple[List[int], int]:
        """Sort indices, applying sign changes for swaps and removing pairs (e_i^2 = 1)."""
        sign = 1
        i = 0
        while i < len(indices):
            j = i + 1
            while j < len(indices):
                if indices[i] > indices[j]:
                    indices[i], indices[j] = indices[j], indices[i]
                    sign *= -1
                elif indices[i] == indices[j]:
                    # e_i * e_i = 1 removes the pair
                    indices.pop(j)
                    indices.pop(i)
                    i -= 1  # step back because list shrank
                    break
                j += 1
            i += 1
        return indices, sign

# ----------------------------------------------------------------------
# Simple Gaussian beam (completion of Parent A)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Return a Gaussian profile evaluated at angle theta."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

# ----------------------------------------------------------------------
# Linguistic utilities (Parent B)
# ----------------------------------------------------------------------
def words(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())

def lsm_vector(text: str) -> Dict[str, float]:
    """Linear‑scaled model: normalized word frequencies."""
    wc = Counter(words(text))
    total = float(sum(wc.values()))
    return {w: c / total for w, c in wc.items()}

# ----------------------------------------------------------------------
# Hybrid utilities – the mathematical bridge
# ----------------------------------------------------------------------
def dict_to_multivector(dist: Dict[str, float],
                        vocab_index: Dict[str, int],
                        n: int) -> Multivector:
    """
    Encode a probability distribution over a vocabulary as a multivector.
    Each word w is mapped to basis blade e_{vocab_index[w]} with coefficient p(w).
    """
    comps: Dict[FrozenSet[int], float] = {}
    for word, prob in dist.items():
        idx = vocab_index.get(word)
        if idx is None:
            continue  # ignore out‑of‑vocabulary words
        comps[frozenset({idx})] = prob
    # add scalar part = 0 (optional)
    return Multivector(comps, n)

def fisher_information_scalar(dist: Dict[str, float]) -> float:
    """
    Simple scalar Fisher information for a discrete distribution.
    I = Σ p_i (log p_i - μ)^2   where μ = Σ p_i log p_i (the entropy term).
    """
    if not dist:
        return 0.0
    log_ps = np.array([math.log(p) if p > 0 else 0.0 for p in dist.values()])
    probs = np.array(list(dist.values()))
    mu = np.sum(probs * log_ps)  # expected log‑probability (negative entropy)
    return float(np.sum(probs * (log_ps - mu) ** 2))

def hybrid_geometric_fisher_score(text: str,
                                  edge_belief: Dict[str, float],
                                  vocab_index: Dict[str, int],
                                  n: int) -> float:
    """
    Core hybrid operation:
      1. Build LSM distribution from the text.
      2. Encode both the text distribution and the edge belief as multivectors.
      3. Compute geometric product, extract scalar part.
      4. Multiply by Fisher information of the text distribution.
    """
    # 1. LSM distribution of the text
    p_text = lsm_vector(text)

    # 2. Multivector encodings
    mv_text = dict_to_multivector(p_text, vocab_index, n)
    mv_edge = dict_to_multivector(edge_belief, vocab_index, n)

    # 3. Geometric product and scalar extraction
    product = mv_text * mv_edge
    scalar = product.scalar_part()

    # 4. Fisher information weighting
    fisher = fisher_information_scalar(p_text)

    return scalar * fisher

# ----------------------------------------------------------------------
# Additional demonstration functions
# ----------------------------------------------------------------------
def build_vocab_index(corpus: List[str]) -> Tuple[Dict[str, int], int]:
    """Create a deterministic word→basis‑index map from a corpus."""
    vocab = sorted({w for txt in corpus for w in words(txt)})
    index = {w: i for i, w in enumerate(vocab)}
    return index, len(vocab)

def hybrid_lsm_score(text: str,
                     posterior_edge_belief: Dict[str, float]) -> float:
    """
    Original LSM‑based score from Parent B, retained for comparison.
    It computes the dot product between the text LSM vector and the edge belief.
    """
    lsm_vec = lsm_vector(text)
    return sum(lsm_vec.get(w, 0.0) * posterior_edge_belief.get(w, 0.0)
               for w in set(lsm_vec) | set(posterior_edge_belief))

def combined_hybrid_score(text: str,
                          edge_belief: Dict[str, float],
                          corpus: List[str]) -> float:
    """
    Produce a blended score:
        α * hybrid_lsm_score + (1‑α) * hybrid_geometric_fisher_score
    where α balances pure statistical similarity vs. geometric‑Fisher interaction.
    """
    vocab_index, n = build_vocab_index(corpus)
    alpha = 0.4
    s_lsm = hybrid_lsm_score(text, edge_belief)
    s_geo = hybrid_geometric_fisher_score(text, edge_belief, vocab_index, n)
    return alpha * s_lsm + (1 - alpha) * s_geo

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_corpus = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps above a sleepy canine",
        "Lorem ipsum dolor sit amet consectetur adipiscing elit"
    ]

    test_text = "The quick brown fox"
    edge_belief = {
        "the": 0.2,
        "quick": 0.15,
        "brown": 0.15,
        "fox": 0.2,
        "jumps": 0.1,
        "over": 0.1,
        "lazy": 0.05,
        "dog": 0.05
    }

    # compute each component
    lsm_score = hybrid_lsm_score(test_text, edge_belief)
    vocab_index, n = build_vocab_index(sample_corpus + [test_text])
    geo_score = hybrid_geometric_fisher_score(test_text, edge_belief, vocab_index, n)
    combined = combined_hybrid_score(test_text, edge_belief, sample_corpus)

    print(f"LSM score:               {lsm_score:.6f}")
    print(f"Geometric‑Fisher score: {geo_score:.6f}")
    print(f"Combined hybrid score:  {combined:.6f}")

    # Demonstrate Gaussian beam (auxiliary)
    print(f"Gaussian beam (θ=0.5, μ=0, σ=1): {gaussian_beam(0.5, 0.0, 1.0):.6f}")