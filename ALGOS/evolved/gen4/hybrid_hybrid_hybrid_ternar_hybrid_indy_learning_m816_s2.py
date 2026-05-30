# DARWIN HAMMER — match 816, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_ternary_lens__hybrid_path_signatur_m34_s1.py (gen2)
# parent_b: hybrid_indy_learning_vector_hybrid_hybrid_geomet_m113_s2.py (gen3)
# born: 2026-05-29T23:31:11Z

import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants & utilities (shared)
# ----------------------------------------------------------------------
CLASSIFICATIONS = {
    "usable_now": 1,
    "research_only": 0,
    "needs_conversion": -1,
    "unsafe_for_fastpath": -1,
    "unsupported": -1,
}
WORD_RE = re.compile(r"\S+")

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

def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

# ----------------------------------------------------------------------
# Parent A – Ternary Lens Audit utilities
# ----------------------------------------------------------------------
def load_manifest(path: Path) -> Dict[str, Any]:
    """Load a vendor manifest JSON and validate classifications."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for cand in data.get("vendors", []):
        cls = cand.get("classification")
        if cls not in CLASSIFICATIONS:
            raise SystemExit(f"invalid classification {cls!r} for {cand.get('candidate_key')}")
    return data

def ternary_audit_vector(candidate: Dict[str, Any]) -> np.ndarray:
    """
    Convert a candidate dict into a 3‑dimensional ternary vector.
    The three dimensions encode (usable, research, other) as +1/0/‑1.
    """
    cls = candidate.get("classification", "unsupported")
    base = CLASSIFICATIONS.get(cls, -1)
    # Simple mapping: replicate base across three axes, but allow future extension.
    return np.array([base, base, base], dtype=np.int8)

def build_audit_matrix(candidates: List[Dict[str, Any]]) -> np.ndarray:
    """Stack ternary vectors of all candidates → shape (M,3)."""
    return np.vstack([ternary_audit_vector(c) for c in candidates])

# ----------------------------------------------------------------------
# Parent B – Tokenisation & Ontology vector utilities
# ----------------------------------------------------------------------
def load_go_terms(root: Path) -> List[str]:
    """Load ontology terms from OFFICIAL_ONTOLOGY.json; fall back to defaults."""
    p = root / "OFFICIAL_ONTOLOGY.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        terms = data.get("active_terms") or []
        return [str(t).upper() for t in terms if str(t).strip()]
    except Exception:
        return list(DEFAULT_TERMS)

def tokenize(text: str) -> List[Dict[str, Any]]:
    """Return token dicts with start/end offsets."""
    return [
        {"token": m.group(0), "start": m.start(), "end": m.end()}
        for m in WORD_RE.finditer(text)
    ]

def term_frequency_vector(text: str, terms: List[str]) -> np.ndarray:
    """
    Build a frequency vector f ∈ ℝᴺ where N = len(terms).
    Matching is case‑insensitive exact term equality.
    """
    tokens = [t["token"].upper() for t in tokenize(text)]
    counter = {term: 0 for term in terms}
    for tok in tokens:
        if tok in counter:
            counter[tok] += 1
    vec = np.array([counter[t] for t in terms], dtype=np.float32)
    # Normalise to unit L2 length to avoid scale issues.
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# ----------------------------------------------------------------------
# Hybrid operations (mathematical fusion)
# ----------------------------------------------------------------------
def path_signature_approx(audit_mat: np.ndarray) -> np.ndarray:
    """
    Approximate a path‑signature of the audit matrix.
    For a matrix A∈ℤ^{M×3} we compute the element‑wise cumulative product
    across rows (i.e. ∏_{k=1}^{M} A_{k,·}) yielding a 3‑vector.
    """
    if audit_mat.size == 0:
        return np.ones(3, dtype=np.int8)
    prod = np.prod(audit_mat, axis=0, dtype=np.int64)
    # Clip to ternary range to keep interpretation stable.
    prod = np.clip(prod, -1, 1)
    return prod.astype(np.int8)

def hybrid_score_matrix(
    audit_mat: np.ndarray,
    freq_vec: np.ndarray,
    term_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute hybrid scores for each candidate.

    Let:
        A  ∈ ℤ^{M×3}   – audit matrix,
        s  = Π_rows(A) ∈ ℤ^{3}   – path‑signature,
        f  ∈ ℝ^{N}    – term‑frequency vector,
        w  ∈ ℝ^{N}    – optional term weights (default: uniform).

    The hybrid score vector h ∈ ℝ^{M} is:
        h_i = (a_i · s) * (f · w)

    This is a contraction of the tensor product a_i⊗f with the scalar
    signatures (a_i·s) and (f·w).
    """
    if term_weights is None:
        term_weights = np.ones_like(freq_vec, dtype=np.float32)
    # Compute L2 norm of term frequency vector
    freq_norm = np.linalg.norm(freq_vec)
    
    # Regularize term_weights to have similar scale
    term_weights = term_weights / np.linalg.norm(term_weights)
    
    # scalar from text side
    text_scalar = float(np.dot(freq_vec, term_weights))
    
    # Down-weight text scalar by freq_norm to prevent magnitude mismatch
    text_scalar *= (freq_norm + 1e-8)
    
    # signature from audit side
    signature = path_signature_approx(audit_mat).astype(np.float32)  # shape (3,)
    # dot each candidate row with signature → shape (M,)
    audit_scalar = audit_mat.astype(np.float32) @ signature
    # final hybrid scores
    return audit_scalar * text_scalar

def evaluate_candidates(
    manifest_path: Path,
    document: str,
    root_dir: Path = Path(__file__).resolve().parents[1],
) -> List[Tuple[str, float]]:
    """
    End‑to‑end hybrid evaluation:
      1. Load manifest → candidate list.
      2. Build audit matrix.
      3. Load ontology terms.
      4. Build term‑frequency vector from the supplied document.
      5. Compute hybrid scores.
      6. Return list of (candidate_key, score) sorted descending.
    """
    data = load_manifest(manifest_path)
    candidates = data.get("vendors", [])
    if not candidates:
        return []

    audit_mat = build_audit_matrix(candidates)                     # (M,3)
    terms = load_go_terms(root_dir)                               # N terms
    freq_vec = term_frequency_vector(document, terms)             # (N,)

    scores = hybrid_score_matrix(audit_mat, freq_vec)             # (M,)
    result = [
        (candidates[i].get("candidate_key", f"cand_{i}"), float(scores[i]))
        for i in range(len(candidates))
    ]
    result.sort(key=lambda x: x[1], reverse=True)
    return result

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    manifest_path = Path("example_manifest.json")
    document = "This is an example document."
    root_dir = Path(__file__).resolve().parents[1]
    result = evaluate_candidates(manifest_path, document, root_dir)
    print(result)