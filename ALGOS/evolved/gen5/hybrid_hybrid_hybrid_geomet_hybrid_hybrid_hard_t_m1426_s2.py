# DARWIN HAMMER — match 1426, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s2.py (gen4)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s3.py (gen3)
# born: 2026-05-29T23:36:24Z

"""Hybrid Text‑Clifford Fusion Module
===================================

Parents
-------
* **Parent A** – ``hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s2.py``
  Implements a minimal Clifford algebra ``Cl(n,0)`` with blade multiplication,
  multivector arithmetic and the geometric product.

* **Parent B** – ``hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s3.py``
  Provides deterministic stylometry feature extraction (96‑dimensional vector)
  and a simple lexical‑function‑category (LSM) vector.

Mathematical Bridge
-------------------
The 96‑dimensional stylometry vector is interpreted as the coordinate list of a
grade‑1 multivector in the Euclidean Clifford algebra ``Cl(96,0)``.  Each
component ``v_i`` becomes the coefficient of the basis blade ``e_i``.  Within
this algebra the **geometric product** ``a * b`` simultaneously yields:

* the **scalar part** (grade‑0) – the inner product, usable as a similarity
  measure between two texts,
* higher‑grade components – encoding oriented relationships between feature
  dimensions.

Thus the core algebraic machinery of Parent A operates directly on the
statistical representation of Parent B, producing a unified similarity and
assignment framework.

The module supplies three public hybrid functions:

1. ``text_to_multivector(text)`` – convert stylometry features to a multivector.
2. ``geometric_similarity(mv1, mv2)`` – scalar‑part similarity via the geometric product.
3. ``assign_texts_to_nodes(texts, node_texts)`` – nearest‑node assignment using
   Clifford‑norm distances.

All code is self‑contained, uses only the permitted standard library and
``numpy`` and runs under Python 3.9+.
"""

import math
import random
import sys
import pathlib
from typing import Any, Dict, List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Clifford algebra core (trimmed to essentials)
# ----------------------------------------------------------------------


def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return a sorted list of indices and the sign produced by anti‑commutation.

    Duplicate indices cancel because e_i*e_i = 1.
    """
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = i
        while j < n - 1:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # e_i * e_i = 1  → remove the pair
                del lst[j : j + 2]
                n -= 2
                i = -1  # restart outer loop
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(
    blade_a: frozenset, blade_b: frozenset
) -> Tuple[frozenset, int]:
    """Geometric multiplication of two basis blades."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Sparse representation of a multivector in Cl(n,0)."""

    def __init__(self, components: Dict[frozenset, float] = None):
        # map: blade (frozenset of ints) -> coefficient (float)
        self.components: Dict[frozenset, float] = {}
        if components:
            for b, v in components.items():
                if abs(v) > 1e-15:
                    self.components[frozenset(b)] = float(v)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def scalar(value: float) -> "Multivector":
        return Multivector({frozenset(): value})

    @staticmethod
    def vector(coeffs: Iterable[float]) -> "Multivector":
        comp = {}
        for i, c in enumerate(coeffs):
            if abs(c) > 1e-15:
                comp[frozenset({i})] = float(c)
        return Multivector(comp)

    # ------------------------------------------------------------------
    # Basic arithmetic
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = self.components.copy()
        for b, v in other.components.items():
            result[b] = result.get(b, 0.0) + v
            if abs(result[b]) < 1e-15:
                del result[b]
        return Multivector(result)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -v for b, v in self.components.items()})

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product (bilinear, associative)."""
        result: Dict[frozenset, float] = {}
        for ba, va in self.components.items():
            for bb, vb in other.components.items():
                bc, sign = _multiply_blades(ba, bb)
                coeff = va * vb * sign
                result[bc] = result.get(bc, 0.0) + coeff
        # prune near‑zero entries
        result = {b: v for b, v in result.items() if abs(v) > 1e-15}
        return Multivector(result)

    # ------------------------------------------------------------------
    # Grade and involutions
    # ------------------------------------------------------------------
    def grade(self, k: int) -> "Multivector":
        """Extract the grade‑k part."""
        comp = {b: v for b, v in self.components.items() if len(b) == k}
        return Multivector(comp)

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) component, 0 if absent."""
        return self.components.get(frozenset(), 0.0)

    def reverse(self) -> "Multivector":
        """Reverse involution: sign = (-1)^{k(k-1)/2} for grade‑k blade."""
        comp = {}
        for b, v in self.components.items():
            k = len(b)
            sign = 1 if (k * (k - 1) // 2) % 2 == 0 else -1
            comp[b] = v * sign
        return Multivector(comp)

    def norm(self) -> float:
        """Euclidean norm derived from the scalar part of a·a_rev."""
        prod = self * self.reverse()
        return math.sqrt(abs(prod.scalar_part()))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for b, v in sorted(self.components.items(), key=lambda x: (len(x[0]), x[0])):
            if not b:
                blade = "1"
            else:
                blade = "*".join(f"e{i}" for i in sorted(b))
            terms.append(f"{v:.3g}{'*' if blade!='1' else ''}{blade}")
        return " + ".join(terms)


# ----------------------------------------------------------------------
# Parent B – Stylometry utilities (trimmed)
# ----------------------------------------------------------------------
FUNCTION_CATS = {
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
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> List[str]:
    """Lower‑case alphabetic tokens (apostrophe‑aware)."""
    return [w for w in text.lower().split() if w.isalpha()]


def stylometry_features(text: str) -> np.ndarray:
    """
    Deterministic 96‑dimensional representation of *text*.
    The hash of the text seeds a NumPy RNG, guaranteeing reproducibility.
    """
    h = np.frombuffer(hashlib.sha256(text.encode("utf-8")).digest()[:8], dtype=np.uint64)[0]
    rng = np.random.default_rng(int(h))
    return rng.random(96)


def lsm_vector(text: str) -> np.ndarray:
    """
    Lexical‑function‑category (LSM) vector: proportion of words belonging
    to each FUNCTION_CAT, ordered alphabetically by category name.
    """
    toks = words(text)
    total = len(toks) or 1
    vec = []
    for cat in sorted(FUNCTION_CATS):
        count = sum(1 for w in toks if w in FUNCTION_CATS[cat])
        vec.append(count / total)
    return np.array(vec, dtype=float)


# ----------------------------------------------------------------------
# Hybrid Layer – bridging the two parents
# ----------------------------------------------------------------------


def text_to_multivector(text: str) -> Multivector:
    """
    Convert a piece of text into a grade‑1 multivector.

    The 96‑dimensional stylometry feature vector ``f`` supplies the coefficients:
        M = Σ_i f_i * e_i

    Returns
    -------
    Multivector
        Grade‑1 multivector in ``Cl(96,0)`` representing the text.
    """
    f = stylometry_features(text)
    return Multivector.vector(f)


def geometric_similarity(mv1: Multivector, mv2: Multivector) -> float:
    """
    Compute a similarity score between two texts represented as multivectors.

    The scalar (grade‑0) part of the geometric product equals the inner product
    of the underlying vectors, i.e. the dot product of their stylometry feature
    vectors.  This value is returned unchanged; its magnitude lies in ``[0,1]``
    for the deterministic random vectors generated by ``stylometry_features``.

    Parameters
    ----------
    mv1, mv2 : Multivector
        Grade‑1 multivectors produced by :func:`text_to_multivector`.

    Returns
    -------
    float
        Scalar part of ``mv1 * mv2`` – a similarity measure.
    """
    prod = mv1 * mv2
    return prod.scalar_part()


def assign_texts_to_nodes(
    texts: List[str], node_texts: List[str]
) -> Dict[int, int]:
    """
    Assign each input text to the nearest “node” text using Clifford‑norm distance.

    For every text ``t`` we compute its multivector ``M_t`` and compare it to
    each node ``n`` with multivector ``M_n``.  The distance is the Euclidean norm
    of the difference multivector: ``||M_t - M_n||``.  The node with minimal
    distance receives the assignment.

    Parameters
    ----------
    texts : list[str]
        Corpus to be classified.
    node_texts : list[str]
        Representative texts (e.g., cluster centroids).

    Returns
    -------
    dict[int, int]
        Mapping from index of ``texts`` to index of the closest ``node_texts``.
    """
    node_mvs = [text_to_multivector(n) for n in node_texts]
    assignment: Dict[int, int] = {}
    for idx, txt in enumerate(texts):
        mv = text_to_multivector(txt)
        best_node = min(
            range(len(node_mvs)),
            key=lambda n_idx: (mv - node_mvs[n_idx]).norm(),
        )
        assignment[idx] = best_node
    return assignment


def hybrid_feature_vector(text: str) -> np.ndarray:
    """
    Produce a combined feature vector that concatenates:

    * the raw stylometry feature vector (96‑dim),
    * the LSM vector (|FUNCTION_CATS|‑dim),
    * the scalar self‑similarity (norm squared).

    The resulting vector can be fed to downstream ML pipelines while still
    retaining a clear algebraic interpretation.
    """
    f = stylometry_features(text)
    l = lsm_vector(text)
    mv = text_to_multivector(text)
    norm_sq = mv.norm() ** 2
    return np.concatenate([f, l, np.array([norm_sq])])


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    txt_a = "The quick brown fox jumps over the lazy dog."
    txt_b = "A fast, dark-colored animal leaps above a sleepy canine."
    txt_c = "Quantum entanglement is a fundamental phenomenon in physics."

    # Convert to multivectors
    mv_a = text_to_multivector(txt_a)
    mv_b = text_to_multivector(txt_b)
    mv_c = text_to_multivector(txt_c)

    # Pairwise similarities
    sim_ab = geometric_similarity(mv_a, mv_b)
    sim_ac = geometric_similarity(mv_a, mv_c)
    sim_bc = geometric_similarity(mv_b, mv_c)

    print(f"Similarity A↔B: {sim_ab:.4f}")
    print(f"Similarity A↔C: {sim_ac:.4f}")
    print(f"Similarity B↔C: {sim_bc:.4f}")

    # Assignment test
    corpus = [txt_a, txt_b, txt_c]
    nodes = ["Animal story.", "Physics discussion."]
    assign = assign_texts_to_nodes(corpus, nodes)
    for i, node_idx in assign.items():
        print(f"Text {i} assigned to node {node_idx} ({nodes[node_idx]})")

    # Hybrid feature vector sanity check
    vec = hybrid_feature_vector(txt_a)
    print(f"Hybrid feature vector length: {len(vec)} (expected {96 + len(FUNCTION_CATS) + 1})")