# DARWIN HAMMER — match 4911, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m958_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1394_s2.py (gen6)
# born: 2026-05-29T23:58:50Z

import numpy as np
import math
import random
import sys
import hashlib
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Set, Callable, Any

# ----------------------------------------------------------------------
# Data structures for labeling
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int

@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float

@dataclass(frozen=True)
class LabelError:
    doc_id: str
    given_label: int
    suggested_label: int
    error_probability: float

def labeling_function(name: str | None = None):
    def deco(fn: Callable[[dict], int]):
        fn.lf_name = name or fn.__name__
        return fn
    return deco

def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    """Weighted majority vote where each vote is weighted by its confidence."""
    votes: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                # default confidence 1.0 for raw LF output; can be overridden later
                votes[r.doc_id].append((r.label, 1.0))
    out: List[ProbabilisticLabel] = []
    for doc_id, weighted_votes in votes.items():
        if not weighted_votes:
            out.append(ProbabilisticLabel(doc_id, 0, 0.5))
            continue
        counter: Counter[int] = Counter()
        weight_sum: Counter[int] = Counter()
        for lbl, w in weighted_votes:
            counter[lbl] += 1
            weight_sum[lbl] += w
        # choose label with highest weighted sum; break ties by raw count
        best_label = max(weight_sum.items(), key=lambda kv: (kv[1], counter[kv[0]]))[0]
        confidence = weight_sum[best_label] / sum(weight_sum.values())
        out.append(ProbabilisticLabel(doc_id, best_label, confidence))
    return out

# ----------------------------------------------------------------------
# Count‑Min Sketch for frequency estimation
# ----------------------------------------------------------------------
class CountMinSketch:
    """A simple Count‑Min Sketch with pairwise‑independent hash functions."""
    def __init__(self, width: int = 2**10, depth: int = 5, seed: int = 0):
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.int64)
        rng = np.random.default_rng(seed)
        self.hash_seeds = rng.integers(1, 2**31 - 1, size=depth, dtype=np.int64)

    def _hash(self, item: bytes, i: int) -> int:
        h = hashlib.blake2b(item, digest_size=8, person=self.hash_seeds[i].to_bytes(8, "little"))
        return int.from_bytes(h.digest(), "little") % self.width

    def add(self, item: str, count: int = 1) -> None:
        b = item.encode("utf-8")
        for i in range(self.depth):
            idx = self._hash(b, i)
            self.tables[i, idx] += count

    def estimate(self, item: str) -> int:
        b = item.encode("utf-8")
        mins = []
        for i in range(self.depth):
            idx = self._hash(b, i)
            mins.append(self.tables[i, idx])
        return min(mins)

    def total_count(self) -> int:
        return int(self.tables.sum() / self.depth)

# Global sketch instance (could be swapped per experiment)
global_sketch = CountMinSketch()

def update_global_sketch(docs: Iterable[dict]) -> None:
    """Populate the global sketch with word frequencies from a corpus."""
    for doc in docs:
        words = doc.get("content", "").split()
        for w in words:
            global_sketch.add(w.lower())

def approx_log_likelihood(word: str) -> float:
    """Approximate log‑likelihood of a word under the empirical distribution."""
    freq = global_sketch.estimate(word.lower())
    total = global_sketch.total_count()
    if freq == 0 or total == 0:
        return -np.inf
    return math.log(freq / total)

# ----------------------------------------------------------------------
# Geometric Algebra utilities (multivector)
# ----------------------------------------------------------------------
def _blade_sign_and_sort(indices: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """Return a sorted tuple of indices with the sign of the permutation.
    Duplicate indices cancel (they are removed) because e_i ∧ e_i = 0."""
    # Count occurrences
    counts: Dict[int, int] = {}
    for idx in indices:
        counts[idx] = counts.get(idx, 0) + 1
    # Remove even occurrences (they cancel)
    remaining = [idx for idx, cnt in counts.items() if cnt % 2 == 1]
    # Sort while tracking parity (sign)
    sign = 1
    lst = list(remaining)
    # Simple bubble‑sort parity computation
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign = -sign
    return tuple(lst), sign

def _multiply_blades(blade_a: Tuple[int, ...], blade_b: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    combined = blade_a + blade_b
    return _blade_sign_and_sort(combined)

class Multivector:
    """Sparse representation of a multivector in a Euclidean Clifford algebra."""
    def __init__(self, components: Dict[Tuple[int, ...], float] | None = None, n: int = 0):
        self.n = int(n)
        self.components: Dict[Tuple[int, ...], float] = {}
        if components:
            for blade, coef in components.items():
                if abs(coef) > 1e-12:
                    self.components[tuple(sorted(blade))] = float(coef)

    def grade(self, k: int) -> "Multivector":
        return Multivector({b: c for b, c in self.components.items() if len(b) == k}, self.n)

    def scalar_part(self) -> float:
        return self.components.get((), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
            if abs(result[blade]) < 1e-12:
                del result[blade]
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-1 * other)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({b: c * scalar for b, c in self.components.items()}, self.n)

    __rmul__ = __mul__

    def geometric_product(self, other: "Multivector") -> "Multivector":
        result: Dict[Tuple[int, ...], float] = defaultdict(float)
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                blade, sign = _multiply_blades(blade_a, blade_b)
                result[blade] += coef_a * coef_b * sign
        # prune near‑zero entries
        result = {b: c for b, c in result.items() if abs(c) > 1e-12}
        return Multivector(result, self.n)

    def norm(self) -> float:
        """Euclidean norm induced by the scalar part of self * reverse(self)."""
        reverse = Multivector({b: ((-1) ** (len(b) * (len(b) - 1) // 2)) * c for b, c in self.components.items()}, self.n)
        gp = self.geometric_product(reverse)
        return math.sqrt(gp.scalar_part())

    def __repr__(self) -> str:
        terms = [" + ".join(f"{c:.3g}{''.join(str(i) for i in b) if b else ''}" for b, c in self.components.items())]
        return f"Multivector({terms[0]})"

# ----------------------------------------------------------------------
# Feature extraction → multivector encoding
# ----------------------------------------------------------------------
def _word_to_basis(word: str, vocab_size: int = 2**16) -> int:
    """Deterministically map a word to a basis index using a 64‑bit hash."""
    h = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % vocab_size

def document_to_multivector(doc: dict, vocab_size: int = 2**16) -> Multivector:
    """Encode a document as a multivector where each unique word contributes a basis blade."""
    words = doc.get("content", "").lower().split()
    coeffs: Dict[Tuple[int, ...], float] = defaultdict(float)
    for w in words:
        idx = _word_to_basis(w, vocab_size)
        coeffs[(idx,)] += 1.0  # simple term frequency; could be tf‑idf
    return Multivector(dict(coeffs), n=vocab_size)

def document_similarity(doc1: dict, doc2: dict) -> float:
    """Similarity based on the cosine of the geometric product."""
    mv1 = document_to_multivector(doc1)
    mv2 = document_to_multivector(doc2)
    gp = mv1.geometric_product(mv2)
    # Use scalar part as inner product analogue
    inner = gp.scalar_part()
    denom = mv1.norm() * mv2.norm()
    if denom == 0:
        return 0.0
    return inner / denom

# ----------------------------------------------------------------------
# Labeling that incorporates sketch‑based confidence
# ----------------------------------------------------------------------
def label_document(doc: dict) -> ProbabilisticLabel:
    """Generate a probabilistic label whose confidence is modulated by word‑level log‑likelihood."""
    words = doc.get("content", "").lower().split()
    # Simple heuristic: if the average log‑likelihood of words exceeds a threshold, label as relevant
    if not words:
        return ProbabilisticLabel(doc["id"], 0, 0.5)
    avg_ll = np.mean([approx_log_likelihood(w) for w in words])
    # Transform log‑likelihood to a [0,1] confidence via sigmoid
    confidence = 1 / (1 + math.exp(-avg_ll))
    label = 1 if confidence > 0.6 else 0
    return ProbabilisticLabel(doc["id"], label, confidence)

def extract_features(doc: dict) -> Dict[str, Any]:
    """Return a richer feature dict, including sketch‑estimated frequencies."""
    words = doc.get("content", "").lower().split()
    freq_features = {w: global_sketch.estimate(w) for w in set(words)}
    length_feature = {"char_len": len(doc.get("content", ""))}
    return {"word_freqs": freq_features, **length_feature}

# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny corpus to seed the sketch
    corpus = [
        {"id": "1", "content": "relevant data for testing"},
        {"id": "2", "content": "irrelevant noise"},
        {"id": "3", "content": "more relevant examples"},
    ]
    update_global_sketch(corpus)

    doc_a = {"id": "A", "content": "This is a relevant document with relevant words"}
    doc_b = {"id": "B", "content": "Another document that is not so relevant"}

    print("Similarity:", document_similarity(doc_a, doc_b))
    print("Label A :", label_document(doc_a))
    print("Label B :", label_document(doc_b))
    print("Features A:", extract_features(doc_a))