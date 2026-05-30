# DARWIN HAMMER — match 34, survivor 2
# gen: 2
# parent_a: hybrid_rbf_surrogate_tri_algo_conduit_m8_s1.py (gen1)
# parent_b: indy_learning_vector.py (gen0)
# born: 2026-05-29T23:23:14Z

"""Hybrid RBF‑Surrogate + INDY_READs Learning Vector.

Parent A: rbf_surrogate.py – radial‑basis‑function surrogate model that learns a mapping
          from a low‑dimensional feature vector (signal, noise, recovery) to a scalar
          output by solving a dense linear system K·w = y.

Parent B: indy_learning_vector.py – deterministic construction of a “learning vector”
          consisting of ontology‑term hit counts extracted from text chunks.

Mathematical bridge:
    The surrogate’s input space is extended with the ontology‑hit count vector.
    For a given payload we compute
        x = [signal, noise, recovery] ⊕ h
    where h ∈ ℝ^T is the term‑frequency vector for the T ontology terms
    (ordered deterministically).  The RBF kernel K_ij = φ(‖x_i−x_j‖) is then
    built on these augmented vectors, preserving the exact linear‑system solve
    of the original surrogate while injecting the high‑dimensional semantic
    information from INDY_READs.  The hybrid therefore learns a joint mapping
    f(signal, noise, recovery, ontology_counts) → target.

The module provides:
    * hybrid_fit – fit an RBFSurrogate on augmented vectors.
    * hybrid_predict – evaluate a new payload + its text chunks.
    * rank_chunks – rank individual chunks by their contribution to the surrogate
      prediction (by feeding each chunk’s ontology vector through the model).

All code is pure Python, using only the standard library and numpy.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple

import numpy as np
import re
import json
import hashlib
from collections import Counter

# ----------------------------------------------------------------------
# Utility functions shared by both parents
# ----------------------------------------------------------------------
Vector = Sequence[float]


def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b with Gaussian elimination (no external deps)."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]


# ----------------------------------------------------------------------
# Parent A – RBF surrogate and tri‑algo conduit
# ----------------------------------------------------------------------
def _byte_entropy(data: bytes, sample: int = 8192) -> float:
    if not data:
        return 0.0
    chunk = data[:sample]
    entropy = 0.0
    for x in set(chunk):
        p = chunk.count(x) / len(chunk)
        entropy -= p * math.log(p, 2)
    return entropy / 8.0


def signal_scores(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> Tuple[float, float]:
    size = len(data)
    entropy = _byte_entropy(data)
    status_bonus = 0.18 if status_code and 200 <= status_code < 300 else -0.10
    mime_bonus = 0.12 if any(x in (mime or "").lower() for x in ("html", "json", "text", "xml")) else 0.02
    size_bonus = min(0.22, math.log1p(size) / 60.0)
    keyword_bonus = min(0.20, keyword_hits * 0.05)
    structure_bonus = min(0.16, structural_links * 0.01)
    signal = max(
        0.0,
        min(
            1.0,
            0.20
            + status_bonus
            + mime_bonus
            + size_bonus
            + keyword_bonus
            + structure_bonus
            + 0.12 * entropy,
        ),
    )
    noise = max(
        0.0,
        min(
            1.0,
            0.58 - 0.22 * entropy - keyword_bonus - structure_bonus + (0.12 if size < 64 else 0.0),
        ),
    )
    return signal, noise


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def recovery_priority(morph: Morphology, max_index: float) -> float:
    return (morph.length + morph.width + morph.height + morph.mass) / (4 * max_index)


def recovery_from_payload(data: bytes, max_bytes: int, parse_error: bool = False) -> float:
    size_ratio = min(4.0, len(data) / max(1, max_bytes))
    morph = Morphology(
        length=1.0 + size_ratio * 8.0,
        width=2.0 + (2.0 if parse_error else 0.5),
        height=max(0.5, 3.0 - size_ratio),
        mass=1.0 + size_ratio * 6.0 + (3.0 if parse_error else 0.0),
    )
    return recovery_priority(morph, max_index=12.0)


def decide(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
    max_bytes: int = 1_500_000,
    parse_error: bool = False,
) -> Tuple[float, float, float]:
    signal, noise = signal_scores(data, status_code, mime, keyword_hits, structural_links)
    recovery = recovery_from_payload(data, max_bytes=max_bytes, parse_error=parse_error)
    return signal, noise, recovery


# ----------------------------------------------------------------------
# Parent B – INDY_READs learning vector utilities
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TERMS = (
    "ENTITY",
    "ATTRIBUTE",
    "RELATIONSHIP",
    "ACTION",
    "EVENT",
    "TIME",
    "EVIDENCE",
    "CLAIM",
    "HYPOTHESIS",
    "SIGNAL",
    "PATTERN",
    "TOOL",
    "ALGORITHM",
    "BOOK",
    "SOURCE",
    "LEAD",
    "LOCATION",
    "LAW",
    "RULE",
)

WORD_RE = re.compile(r"\S+")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def load_go_terms(root: Path = ROOT) -> List[str]:
    p = root / "OFFICIAL_ONTOLOGY.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        terms = data.get("active_terms") or []
        return [str(t).upper() for t in terms if str(t).strip()]
    except Exception:
        return list(DEFAULT_TERMS)


def tokenize(text: str) -> List[dict]:
    return [{"token": m.group(0), "start": m.start(), "end": m.end()} for m in WORD_RE.finditer(text)]


def chunk_text_tokens(
    text: str,
    *,
    max_tokens: int = 500,
    overlap_tokens: int = 0,
    source_ref: dict | None = None,
) -> List[dict]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
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
    chunks: List[dict] = []
    token_start = 0
    idx = 0
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
        if token_end == len(toks):
            break
        token_start = token_end - overlap_tokens
        idx += 1
    return chunks


def ontology_hits_for_text(text: str, terms: List[str] | None = None) -> List[dict]:
    terms = terms or load_go_terms()
    upper = text.upper()
    hits: List[dict] = []
    for term in terms:
        if re.search(r"\b" + re.escape(term.upper()) + r"\b", upper):
            cnt = len(re.findall(r"\b" + re.escape(term.upper()) + r"\b", upper))
            hits.append({"term": term.upper(), "count": cnt})
    return hits


def build_term_vector(chunks: List[dict], terms: List[str] | None = None) -> List[int]:
    """Return a fixed‑length count vector aligned with `terms`."""
    terms = terms or load_go_terms()
    counter: Counter[str] = Counter()
    for c in chunks:
        for hit in ontology_hits_for_text(c.get("text", ""), terms):
            counter[hit["term"]] += int(hit["count"])
    return [counter.get(t, 0) for t in terms]


# ----------------------------------------------------------------------
# Hybrid data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon) for w, c in zip(self.weights, self.centers)
        )


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_feature_vector(
    data: bytes,
    chunks: List[dict],
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
    terms: List[str] | None = None,
) -> List[float]:
    """Compose the augmented feature vector [signal, noise, recovery] + ontology counts."""
    signal, noise, recovery = (*decide(data, status_code, mime, keyword_hits, structural_links),)
    term_vec = build_term_vector(chunks, terms)
    return [signal, noise, recovery] + term_vec


def hybrid_fit(
    data_samples: List[bytes],
    chunk_lists: List[List[dict]],
    targets: List[float],
    epsilon: float = 1.0,
    ridge: float = 1e-9,
    terms: List[str] | None = None,
) -> RBFSurrogate:
    """
    Fit an RBFSurrogate on the hybrid feature space.

    Parameters
    ----------
    data_samples : list of raw payload bytes
    chunk_lists  : list of chunk lists (parallel to data_samples)
    targets      : list of scalar targets (e.g., relevance scores)
    epsilon      : RBF shape parameter
    ridge        : small diagonal regularisation
    terms        : optional fixed ontology term ordering
    """
    if not (len(data_samples) == len(chunk_lists) == len(targets)):
        raise ValueError("All input sequences must have the same length")
    # Build feature matrix
    X = [hybrid_feature_vector(d, ch, terms=terms) for d, ch in zip(data_samples, chunk_lists)]
    Y = [float(v) for v in targets]
    n = len(X)
    # Kernel matrix with ridge regularisation
    K = np.empty((n, n), dtype=np.float64)
    for i, xi in enumerate(X):
        for j, xj in enumerate(X):
            K[i, j] = gaussian(euclidean(xi, xj), epsilon)
        K[i, i] += ridge
    # Solve for weights
    w = np.linalg.solve(K, np.array(Y, dtype=np.float64))
    # Convert centers to tuples for immutability
    centers = [tuple(map(float, xi)) for xi in X]
    return RBFSurrogate(centers, list(map(float, w)), epsilon)


def hybrid_predict(
    surrogate: RBFSurrogate,
    data: bytes,
    chunks: List[dict],
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
    terms: List[str] | None = None,
) -> float:
    """Predict a scalar using the hybrid surrogate."""
    x = hybrid_feature_vector(
        data,
        chunks,
        status_code,
        mime,
        keyword_hits,
        structural_links,
        terms=terms,
    )
    return surrogate.predict(x)


def rank_chunks(
    surrogate: RBFSurrogate,
    data: bytes,
    chunks: List[dict],
    terms: List[str] | None = None,
) -> List[Tuple[str, float]]:
    """
    Rank individual chunks by their marginal contribution.
    For each chunk we construct a feature vector that contains only that
    chunk’s ontology term counts (others are zero) together with the
    tri‑algo conduit scores, then evaluate the surrogate.
    Returns a list of (chunk_id, score) sorted descending.
    """
    # Base conduit scores are identical for all chunks
    base_signal, base_noise, base_recovery = decide(data)
    term_list = terms or load_go_terms()
    zero_counts = [0] * len(term_list)

    rankings = []
    for c in chunks:
        # Build a term vector that only reflects the current chunk
        vec = build_term_vector([c], term_list)
        feature = [base_signal, base_noise, base_recovery] + vec
        score = surrogate.predict(feature)
        rankings.append((c["chunk_id"], score))

    rankings.sort(key=lambda kv: kv[1], reverse=True)
    return rankings


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample payload
    payload = b"The quick brown fox jumps over the lazy dog."
    status = 200
    mime_type = "text/plain"
    kw_hits = 1
    struct_links = 0

    # Create a tiny document and chunk it
    sample_text