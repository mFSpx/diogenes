# DARWIN HAMMER — match 113, survivor 1
# gen: 3
# parent_a: indy_learning_vector.py (gen0)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s1.py (gen2)
# born: 2026-05-29T23:25:56Z

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import math
import numpy as np

# ----------------------------------------------------------------------
# INDY vector utilities (parent A)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"\S+")
DEFAULT_TERMS = (
    "ENTITY", "ATTRIBUTE", "RELATIONSHIP", "ACTION", "EVENT", "TIME", "EVIDENCE",
    "CLAIM", "HYPOTHESIS", "SIGNAL", "PATTERN", "TOOL", "ALGORITHM", "BOOK",
    "SOURCE", "LEAD", "LOCATION", "LAW", "RULE",
)


def sha256_json(value: Any) -> str:
    """Deterministic SHA‑256 of a JSON‑serialisable value."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def load_go_terms(root: Path = ROOT) -> List[str]:
    """Load ontology terms; fall back to DEFAULT_TERMS."""
    p = root / "OFFICIAL_ONTOLOGY.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        terms = data.get("active_terms") or []
        return [str(t).upper() for t in terms if str(t).strip()]
    except Exception:
        return list(DEFAULT_TERMS)


def tokenize(text: str) -> List[Dict[str, Any]]:
    """Return a list of token dicts with start/end character offsets."""
    return [
        {"token": m.group(0), "start": m.start(), "end": m.end()}
        for m in WORD_RE.finditer(text)
    ]


def chunk_text_tokens(
    text: str,
    *,
    max_tokens: int = 200,
    overlap_tokens: int = 0,
    source_ref: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Split text into overlapping token chunks."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not (0 <= overlap_tokens < max_tokens):
        raise ValueError("overlap_tokens must be >=0 and < max_tokens")
    toks = tokenize(text)
    source_ref = dict(source_ref or {})
    if not toks:
        cid = "chunk:" + sha256_json({"source_ref": source_ref, "empty": True})[:24]
        return [
            {
                "chunk_id": cid,
                "chunk_index": 0,
                "token_start": 0,
                "token_end": 0,
                "char_start": 0,
                "char_end": 0,
                "token_count": 0,
                "text": "",
                "source_ref": source_ref,
            }
        ]

    chunks: List[Dict[str, Any]] = []
    idx = 0
    token_start = 0
    while token_start < len(toks):
        token_end = min(len(toks), token_start + max_tokens)
        char_start = toks[token_start]["start"]
        char_end = toks[token_end - 1]["end"]
        chunk_text = text[char_start:char_end]
        cid = "chunk:" + sha256_json(
            {
                "source_ref": source_ref,
                "token_start": token_start,
                "token_end": token_end,
                "text": chunk_text,
            }
        )[:24]
        chunks.append(
            {
                "chunk_id": cid,
                "chunk_index": idx,
                "token_start": token_start,
                "token_end": token_end,
                "char_start": char_start,
                "char_end": char_end,
                "token_count": token_end - token_start,
                "text": chunk_text,
                "source_ref": source_ref,
            }
        )
        idx += 1
        token_start = token_end - overlap_tokens
    return chunks


def chunk_to_term_vector(chunk: Dict[str, Any], terms: List[str]) -> np.ndarray:
    """Convert a chunk into a frequency vector over the supplied ontology terms."""
    token_counts = Counter(
        token["token"].upper() for token in tokenize(chunk["text"])
    )
    vec = np.zeros(len(terms), dtype=float)
    term_index = {t: i for i, t in enumerate(terms)}
    for tok, cnt in token_counts.items():
        if tok in term_index:
            vec[term_index[tok]] = cnt
    # Normalise to unit L2 norm to make distances comparable
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ----------------------------------------------------------------------
# Geometric algebra & Voronoi utilities (parent B)
# ----------------------------------------------------------------------
Point = Tuple[float, float]  # used only for 2‑D examples; our vectors are nd‑arrays


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def nearest(point: np.ndarray, seeds: List[np.ndarray]) -> int:
    if not seeds:
        raise ValueError("seeds required")
    distances = [euclidean(point, s) for s in seeds]
    return int(np.argmin(distances))


def assign(
    points: List[np.ndarray], seeds: List[np.ndarray]
) -> Dict[int, List[np.ndarray]]:
    """Assign each point to the Voronoi region of its nearest seed."""
    regions: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(seeds))}
    for p in points:
        idx = nearest(p, seeds)
        regions[idx].append(p)
    return regions


def _blade_sign(indices: Tuple[int, ...]) -> Tuple[List[int], int]:
    """Return sorted indices and the sign of the permutation (Grassmann sign)."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate index cancels (e_i ^ e_i = 0)
                lst.pop(j)
                lst.pop(j)
                return lst, sign
    return lst, sign


def _multiply_blades(
    blade_a: Tuple[int, ...], blade_b: Tuple[int, ...]
) -> Tuple[frozenset, int]:
    combined = tuple(blade_a) + tuple(blade_b)
    sorted_idxs, sign = _blade_sign(combined)
    return frozenset(sorted_idxs), sign


class Multivector:
    """Simple multivector limited to grades up to 2 (scalar + vector + bivector)."""

    def __init__(self, components: Dict[frozenset, float], n: int):
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-12
        }
        self.n = int(n)
        self.com = self.components.get(frozenset(), 0)

    @staticmethod
    def from_vector(v: np.ndarray) -> "Multivector":
        """Create a grade‑1 multivector from a numpy vector."""
        comps = {frozenset({i}): float(v[i]) for i in range(len(v)) if v[i] != 0}
        return Multivector(comps, n=len(v))

    def scalar_part(self) -> float:
        return self.com

    def __add__(self, other: "Multivector") -> "Multivector":
        if self.n != other.n:
            raise ValueError("Dimensions must match")
        comps = self.components.copy()
        comps.update({k: comps.get(k, 0) + v for k, v in other.components.items()})
        return Multivector(comps, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        if self.n != other.n:
            raise ValueError("Dimensions must match")
        comps = {}
        for k1, v1 in self.components.items():
            for k2, v2 in other.components.items():
                k, s = _multiply_blades(tuple(k1), tuple(k2))
                comps[k] = comps.get(k, 0) + s * v1 * v2
        return Multivector(comps, self.n)


def geometric_product(mv1: Multivector, mv2: Multivector) -> Multivector:
    return mv1 * mv2


def aggregate_multivectors(region_vectors: List[Multivector]) -> Multivector:
    mv = Multivector({}, n=region_vectors[0].n)
    for v in region_vectors:
        mv += v
    return mv


class HybridModel:
    def __init__(self, terms: List[str], seeds: List[np.ndarray]):
        self.terms = terms
        self.seeds = seeds

    def _chunk_to_multivector(self, chunk: Dict[str, Any]) -> Multivector:
        vec = chunk_to_term_vector(chunk, self.terms)
        return Multivector.from_vector(vec)

    def fit(self, chunks: List[Dict[str, Any]]) -> Dict[int, Multivector]:
        vectors = [self._chunk_to_multivector(c) for c in chunks]
        regions = assign([v.components[frozenset({i})] * np.eye(len(self.terms)) for v in vectors], self.seeds)
        region_multivectors = {}
        for idx, region_vectors in regions.items():
            region_multivectors[idx] = aggregate_multivectors([Multivector.from_vector(v) for v in region_vectors])
        return region_multivectors

    def ollivier_ricci_curvature(self, region_multivectors: Dict[int, Multivector], region_centroids: List[np.ndarray]) -> Dict[Tuple[int, int], float]:
        curvatures = {}
        for i in region_multivectors:
            for j in region_multivectors:
                if i != j:
                    transport_cost = euclidean(region_centroids[i], region_centroids[j])
                    curvatures[(i, j)] = transport_cost * region_multivectors[i].scalar_part() * region_multivectors[j].scalar_part()
        return curvatures


def main():
    terms = load_go_terms()
    seeds = [np.random.rand(len(terms)) for _ in range(10)]
    model = HybridModel(terms, seeds)

    text = "This is a sample text."
    chunks = chunk_text_tokens(text)
    region_multivectors = model.fit(chunks)

    region_centroids = [np.mean([model._chunk_to_multivector(c).components[frozenset({i})] * np.eye(len(terms)) for c in chunks if model.nearest(model._chunk_to_multivector(c).components[frozenset({i})] * np.eye(len(terms)), model.seeds) == i], axis=0) for i in range(len(model.seeds))]
    curvatures = model.ollivier_ricci_curvature(region_multivectors, region_centroids)

    print(curvatures)


if __name__ == "__main__":
    main()