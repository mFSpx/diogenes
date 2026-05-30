# DARWIN HAMMER — match 113, survivor 0
# gen: 3
# parent_a: indy_learning_vector.py (gen0)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s1.py (gen2)
# born: 2026-05-29T23:25:56Z

"""Hybrid module combining INDY vector chunking (indy_learning_vector.py) with
geometric algebra Voronoi partitioning and Ollivier‑Ricci curvature
(hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s1.py).

Mathematical bridge:
- Each text chunk is mapped to a high‑dimensional frequency vector whose
  axes are the ontology terms loaded by the INDY routine.
- Frequency vectors are treated as grade‑1 blades of a geometric algebra.
  The geometric product of two such blades yields a multivector containing a
  scalar (dot product) and a bivector (wedge product).
- A set of seed vectors defines a Voronoi diagram in the same space; each
  chunk‑vector is assigned to the nearest seed region.
- For every region we aggregate the multivectors of its members (pairwise
  geometric product) to obtain a region‑level multivector.
- Ollivier‑Ricci curvature between two neighboring regions is estimated
  from the transport cost of moving the normalized weight distributions of
  their vectors, using the Euclidean distance between region centroids.
  The scalar part of the region multivector (dot product) supplies the
  “mass” needed for the transport calculation.

The module therefore fuses deterministic text tokenisation with
geometric‑algebraic connectivity analysis in a single, self‑contained
pipeline.
"""

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

    @staticmethod
    def from_vector(v: np.ndarray) -> "Multivector":
        """Create a grade‑1 multivector from a numpy vector."""
        comps = {frozenset({i}): float(v[i]) for i in range(len(v)) if v[i] != 0}
        return Multivector(comps, n=len(v))

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        new = self.components.copy()
        for k, v in other.components.items():
            new[k] = new.get(k, 0.0) + v
        return Multivector(new, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product (limited to scalar, vector, bivector terms)."""
        result: Dict[frozenset, float] = {}
        for a_blade, a_coef in self.components.items():
            for b_blade, b_coef in other.components.items():
                new_blade, sign = _multiply_blades(tuple(a_blade), tuple(b_blade))
                result[new_blade] = result.get(new_blade, 0.0) + sign * a_coef * b_coef
        return Multivector(result, self.n)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(
            self.components.items(),
            key=lambda x: (len(x[0]), sorted(x[0])),
        ):
            if not blade:
                term = f"{coef:.3g}"
            else:
                basis = "".join(f"e{i}" for i in sorted(blade))
                term = f"{coef:.3g}{basis}"
            terms.append(term)
        return " + ".join(terms)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def text_to_region_multivectors(
    text: str,
    max_tokens: int = 200,
    overlap_tokens: int = 20,
    seed_count: int = 3,
) -> Tuple[Dict[int, Multivector], List[int]]:
    """
    End‑to‑end pipeline:
    1. Chunk the input text.
    2. Convert each chunk into a normalized term frequency vector.
    3. Choose `seed_count` random vectors as Voronoi seeds.
    4. Assign chunks to Voronoi regions.
    5. For each region, compute the aggregated multivector obtained by
       pairwise geometric products of its member vectors.
    Returns a mapping region_index → aggregated Multivector and the list of
    region indices that contain at least one chunk.
    """
    terms = load_go_terms()
    chunks = chunk_text_tokens(
        text, max_tokens=max_tokens, overlap_tokens=overlap_tokens
    )
    vectors = [chunk_to_term_vector(c, terms) for c in chunks]

    if not vectors:
        return {}, []

    # Random seed selection (ensure reproducibility if needed)
    rng = random.Random(42)
    seed_vectors = rng.sample(vectors, min(seed_count, len(vectors)))

    regions = assign(vectors, seed_vectors)

    region_mvs: Dict[int, Multivector] = {}
    nonempty_regions: List[int] = []

    for idx, vecs in regions.items():
        if not vecs:
            continue
        nonempty_regions.append(idx)
        # Start with the multivector of the first vector
        agg_mv = Multivector.from_vector(vecs[0])
        for v in vecs[1:]:
            agg_mv = agg_mv * Multivector.from_vector(v)
        region_mvs[idx] = agg_mv
    return region_mvs, nonempty_regions


def curvature_between_regions(
    region_a: List[np.ndarray], region_b: List[np.ndarray]
) -> float:
    """
    Approximate Ollivier‑Ricci curvature between two regions.
    Let μ_A and μ_B be uniform probability measures on the vectors of each region.
    The 1‑Wasserstein distance W₁(μ_A, μ_B) is approximated by the average
    Euclidean distance between all cross‑pairs.
    Curvature κ is then defined as:
        κ = 1 - W₁ / d(c_A, c_B)
    where c_A and c_B are the centroids of the two regions.
    The value lies in (-∞, 1]; κ≈1 indicates tightly coupled regions.
    """
    if not region_a or not region_b:
        return 0.0

    # Centroids
    c_a = np.mean(region_a, axis=0)
    c_b = np.mean(region_b, axis=0)
    d_centroid = euclidean(c_a, c_b)
    if d_centroid == 0.0:
        return 1.0

    # Approximate transport cost by mean pairwise distance
    pairwise = [
        euclidean(pa, pb) for pa in region_a for pb in region_b
    ]
    w1 = float(np.mean(pairwise))
    curvature = 1.0 - w1 / d_centroid
    return curvature


def region_curvature_map(
    text: str,
    max_tokens: int = 200,
    overlap_tokens: int = 20,
    seed_count: int = 3,
) -> Dict[Tuple[int, int], float]:
    """
    Compute curvature for every unordered pair of non‑empty Voronoi regions.
    Returns a dictionary keyed by (region_i, region_j) with curvature values.
    """
    terms = load_go_terms()
    chunks = chunk_text_tokens(
        text, max_tokens=max_tokens, overlap_tokens=overlap_tokens
    )
    vectors = [chunk_to_term_vector(c, terms) for c in chunks]

    if not vectors:
        return {}

    rng = random.Random(42)
    seed_vectors = rng.sample(vectors, min(seed_count, len(vectors)))
    regions = assign(vectors, seed_vectors)

    # Build list of vectors per region (filter empties)
    region_lists = {i: lst for i, lst in regions.items() if lst}
    indices = sorted(region_lists.keys())
    curv_map: Dict[Tuple[int, int], float] = {}

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            ri, rj = indices[i], indices[j]
            curv = curvature_between_regions(region_lists[ri], region_lists[rj])
            curv_map[(ri, rj)] = curv
    return curv_map


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "Entity relationship algorithms can model actions and events. "
        "Evidence and hypothesis form the basis of scientific claims. "
        "Time, location, and law regulate the interaction of tools."
    )
    # 1. Obtain region multivectors
    region_mvs, nonempty = text_to_region_multivectors(sample_text, seed_count=2)
    print("Aggregated multivectors per region:")
    for idx, mv in region_mvs.items():
        print(f" Region {idx}: {mv}")

    #