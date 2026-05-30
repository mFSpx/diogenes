# DARWIN HAMMER — match 113, survivor 2
# gen: 3
# parent_a: indy_learning_vector.py (gen0)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s1.py (gen2)
# born: 2026-05-29T23:25:56Z

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


def calculate_ollivier_ricci_curvature(
    region1: List[np.ndarray], region2: List[np.ndarray]
) -> float:
    """Calculate Ollivier-Ricci curvature between two regions."""
    # Calculate the centroids of the two regions
    centroid1 = np.mean(region1, axis=0)
    centroid2 = np.mean(region2, axis=0)

    # Calculate the transport cost
    transport_cost = 0.0
    for point in region1:
        transport_cost += np.linalg.norm(point - centroid2)
    for point in region2:
        transport_cost += np.linalg.norm(point - centroid1)

    # Calculate the Ollivier-Ricci curvature
    curvature = 1 - (transport_cost / (len(region1) + len(region2)))

    return curvature


def improved_hybrid_algorithm(
    text: str,
    max_tokens: int = 200,
    overlap_tokens: int = 0,
    source_ref: Dict[str, Any] | None = None,
) -> Dict[int, List[np.ndarray]]:
    """Improved hybrid algorithm combining INDY vector chunking with geometric algebra Voronoi partitioning and Ollivier-Ricci curvature."""
    terms = load_go_terms()
    chunks = chunk_text_tokens(
        text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, source_ref=source_ref
    )
    vectors = [chunk_to_term_vector(chunk, terms) for chunk in chunks]

    # Calculate the seeds for the Voronoi diagram
    seeds = [vectors[i] for i in range(0, len(vectors), len(vectors) // 10)]

    # Assign each vector to its nearest seed
    regions = assign(vectors, seeds)

    # Calculate the Ollivier-Ricci curvature between each pair of regions
    curvatures = {}
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            curvature = calculate_ollivier_ricci_curvature(regions[i], regions[j])
            curvatures[(i, j)] = curvature

    return curvatures


# Example usage:
text = "This is an example text."
curvatures = improved_hybrid_algorithm(text)
for pair, curvature in curvatures.items():
    print(f"Curvature between regions {pair}: {curvature}")