# DARWIN HAMMER — match 4741, survivor 2
# gen: 6
# parent_a: hybrid_indy_learning_vector_hybrid_hybrid_geomet_m113_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1586_s3.py (gen5)
# born: 2026-05-29T23:57:50Z

"""Hybrid Text‑Chunk Regret‑Weighted Ternary Analyzer

This module fuses two distinct parents:

* **Parent A** – `indy_learning_vector` utilities for tokenising text,
  chunking it into overlapping token windows and deterministically hashing
  metadata.
* **Parent B** – a regret‑weighted ternary decision engine that builds a
  signature over a token set, maps the signature to a ternary decision
  vector and scores it with `MathAction` objects, finally applying a
  sigmoid‑based probabilistic transform.

**Mathematical bridge**

1. A text chunk is turned into a *set* of tokens (Parent A).
2. From that token set a *min‑hash* signature `σ ∈ ℕ^k` is built
   (`signature` from Parent B).
3. Each integer `σ_i` is mapped to a ternary decision `τ_i ∈ {‑1,0,1}`
   via `τ_i = (σ_i % 3) - 1`.  This creates a ternary decision vector
   `τ` that lives in the same space as the Hybrid Ternary‑Decision Hygiene
   Analyzer of Parent B.
4. A regret‑weighted score `S = Σ_a (E_a − C_a)·τ·R_a` is computed,
   where `E_a` is the expected value, `C_a` the cost and `R_a` the risk of
   a `MathAction` (Parent B).
5. The raw score is finally transformed with a numerically stable sigmoid
   (`sigmoid`) to obtain a probability‑like confidence value.

The three core functions below demonstrate the end‑to‑end hybrid pipeline:
chunking → ternary signature → regret‑weighted scoring → sigmoid
transformation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – INDY vector utilities
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
                "text": "",
                "tokens": [],
                "source_ref": source_ref,
            }
        ]

    chunks: List[Dict[str, Any]] = []
    i = 0
    while i < len(toks):
        window = toks[i : i + max_tokens]
        chunk_text = " ".join(tok["token"] for tok in window)
        cid = "chunk:" + sha256_json(
            {"source_ref": source_ref, "start": i, "end": i + len(window)}
        )[:24]
        chunks.append(
            {
                "chunk_id": cid,
                "text": chunk_text,
                "tokens": [tok["token"] for tok in window],
                "source_ref": source_ref,
            }
        )
        i += max_tokens - overlap_tokens
    return chunks


# ----------------------------------------------------------------------
# Parent B – Regret‑Weighted Ternary Engine
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action descriptor used in regret‑weighted scoring."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome used for alternative‑scenario analysis."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit integer hash based on a seed and a token."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: List[str], k: int = 128) -> List[int]:
    """
    Min‑hash style signature of a token set.

    For each row `i` in `0..k-1` the smallest hash value over the token set
    is kept.  The result is a deterministic integer vector of length `k`.
    """
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        # Return a vector of maximal 64‑bit values – neutral for similarity.
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """
    Jaccard‑like similarity for min‑hash signatures.

    It is the proportion of positions where the two signatures agree.
    """
    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must be of equal length")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def chunk_signature(chunk: Dict[str, Any], k: int = 64) -> List[int]:
    """
    Compute a min‑hash signature for a single text chunk.
    """
    return signature(chunk["tokens"], k=k)


def signature_to_ternary(sig: List[int]) -> np.ndarray:
    """
    Map a signature vector to a ternary decision vector τ ∈ {‑1,0,1}ⁿ.

    The mapping uses `τ_i = (sig_i % 3) - 1`, yielding:
        0 → -1, 1 → 0, 2 → +1
    """
    arr = np.fromiter((s % 3 for s in sig), dtype=np.int8, count=len(sig))
    return arr - 1  # now in {-1,0,1}


def regret_weighted_score(
    ternary_vec: np.ndarray,
    actions: List[MathAction],
) -> float:
    """
    Compute a regret‑weighted scalar score.

    For each action `a` we form a contribution:
        (E_a - C_a) * (τ·r_a)

    where `τ` is the ternary decision vector and `r_a` is a risk‑scaled
    version of the action (risk factor defaults to 1.0 if `risk` is 0).
    The final score is the sum of contributions.
    """
    if ternary_vec.ndim != 1:
        raise ValueError("ternary_vec must be a 1‑D array")
    # Normalise ternary vector length to match number of actions if needed.
    # If the signature length differs from the number of actions we repeat
    # or truncate the vector to align dimensions.
    n = len(actions)
    if ternary_vec.size != n:
        # Simple repeat/truncate strategy
        repeats = (n + ternary_vec.size - 1) // ternary_vec.size
        vec = np.tile(ternary_vec, repeats)[:n]
    else:
        vec = ternary_vec

    total = 0.0
    for coeff, act in zip(vec, actions):
        risk_factor = act.risk if act.risk != 0 else 1.0
        contribution = (act.expected_value - act.cost) * coeff * risk_factor
        total += contribution
    return total


def transform_score(score: float) -> float:
    """
    Apply a sigmoid transform to a scalar score and return a confidence
    value in the interval (0, 1).  The input is first cast to a NumPy
    array for vectorised stability.
    """
    arr = np.array([score], dtype=np.float64)
    return float(sigmoid(arr)[0])


def process_text_hybrid(
    text: str,
    *,
    max_tokens: int = 200,
    overlap: int = 0,
    signature_k: int = 64,
    actions: List[MathAction] | None = None,
) -> List[Tuple[str, float, float]]:
    """
    End‑to‑end hybrid pipeline.

    Returns a list of tuples per chunk:
        (chunk_id, raw_regret_score, transformed_confidence)
    """
    if actions is None:
        # Default toy actions – one per signature component.
        actions = [
            MathAction(id=f"a{i}", expected_value=random.uniform(0, 10), cost=random.uniform(0, 5), risk=random.uniform(0.5, 1.5))
            for i in range(signature_k)
        ]

    chunks = chunk_text_tokens(text, max_tokens=max_tokens, overlap_tokens=overlap)
    results: List[Tuple[str, float, float]] = []

    for ch in chunks:
        sig = chunk_signature(ch, k=signature_k)
        tern = signature_to_ternary(sig)
        raw = regret_weighted_score(tern, actions)
        conf = transform_score(raw)
        results.append((ch["chunk_id"], raw, conf))
    return results


def compute_similarity_between_chunks(
    chunk_a: Dict[str, Any],
    chunk_b: Dict[str, Any],
    k: int = 64,
) -> float:
    """
    Compute signature‑based similarity between two chunks using the
    Jaccard‑like metric defined in Parent B.
    """
    sig_a = chunk_signature(chunk_a, k=k)
    sig_b = chunk_signature(chunk_b, k=k)
    return similarity(sig_a, sig_b)


def apply_regret_transform(
    raw_scores: List[float],
) -> List[float]:
    """
    Vectorised sigmoid transformation of a list of raw regret scores.
    """
    arr = np.array(raw_scores, dtype=np.float64)
    return list(sigmoid(arr))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "In computational physics we often need to discretise continuous "
        "domains, apply numerical solvers, and analyse error bounds. "
        "Hybrid algorithms that combine symbolic reasoning with statistical "
        "methods can improve robustness."
    )

    # Run the full hybrid pipeline
    results = process_text_hybrid(sample_text, max_tokens=30, overlap=5, signature_k=32)

    for cid, raw, conf in results:
        print(f"Chunk {cid[:8]}… | Raw score: {raw: .3f} | Confidence: {conf:.3f}")

    # Demonstrate similarity between first two chunks (if at least two exist)
    if len(results) >= 2:
        chunks = chunk_text_tokens(sample_text, max_tokens=30, overlap_tokens=5)
        sim = compute_similarity_between_chunks(chunks[0], chunks[1], k=32)
        print(f"\nSimilarity between first two chunks: {sim:.3f}")

    # Vectorised transform demo
    raw_vals = [r for _, r, _ in results]
    transformed = apply_regret_transform(raw_vals)
    print("\nVectorised transformed confidences:", ["{:.3f}".format(v) for v in transformed])