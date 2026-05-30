# DARWIN HAMMER — match 816, survivor 3
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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

# ----------------------------------------------------------------------
# Parent A – Ternary Lens Audit utilities
# ----------------------------------------------------------------------
def load_manifest(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for cand in data.get("vendors", []):
        cls = cand.get("classification")
        if cls not in CLASSIFICATIONS:
            raise SystemExit(f"invalid classification {cls!r} for {cand.get('candidate_key')}")
    return data

def ternary_audit_vector(candidate: Dict[str, Any]) -> np.ndarray:
    cls = candidate.get("classification", "unsupported")
    base = CLASSIFICATIONS.get(cls, -1)
    return np.array([base, base, base], dtype=np.int8)

def build_audit_matrix(candidates: List[Dict[str, Any]]) -> np.ndarray:
    return np.vstack([ternary_audit_vector(c) for c in candidates])

# ----------------------------------------------------------------------
# Parent B – Tokenisation & Ontology vector utilities
# ----------------------------------------------------------------------
def load_go_terms(root: Path) -> List[str]:
    p = root / "OFFICIAL_ONTOLOGY.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        terms = data.get("active_terms") or []
        return [str(t).upper() for t in terms if str(t).strip()]
    except Exception:
        return list(DEFAULT_TERMS)

def tokenize(text: str) -> List[Dict[str, Any]]:
    return [
        {"token": m.group(0), "start": m.start(), "end": m.end()}
        for m in WORD_RE.finditer(text)
    ]

def term_frequency_vector(text: str, terms: List[str]) -> np.ndarray:
    tokens = [t["token"].upper() for t in tokenize(text)]
    counter = {term: 0 for term in terms}
    for tok in tokens:
        if tok in counter:
            counter[tok] += 1
    vec = np.array([counter[t] for t in terms], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# ----------------------------------------------------------------------
# Hybrid operations (mathematical fusion)
# ----------------------------------------------------------------------
def path_signature_approx(audit_mat: np.ndarray) -> np.ndarray:
    if audit_mat.size == 0:
        return np.ones(3, dtype=np.int8)
    prod = np.prod(audit_mat, axis=0, dtype=np.int64)
    prod = np.clip(prod, -1, 1)
    return prod.astype(np.int8)

def hybrid_score_matrix(
    audit_mat: np.ndarray,
    freq_vec: np.ndarray,
    term_weights: np.ndarray | None = None,
) -> np.ndarray:
    if term_weights is None:
        term_weights = np.ones_like(freq_vec, dtype=np.float32)
    text_scalar = np.dot(freq_vec, term_weights)
    signature = path_signature_approx(audit_mat).astype(np.float32)
    audit_scalar = audit_mat.astype(np.float32) @ signature
    return audit_scalar * text_scalar

def evaluate_candidates(
    manifest_path: Path,
    document: str,
    root_dir: Path = Path(__file__).resolve().parents[1],
) -> List[Tuple[str, float]]:
    data = load_manifest(manifest_path)
    candidates = data.get("vendors", [])
    if not candidates:
        return []

    audit_mat = build_audit_matrix(candidates)
    terms = load_go_terms(root_dir)
    freq_vec = term_frequency_vector(document, terms)

    scores = hybrid_score_matrix(audit_mat, freq_vec)
    result = [
        (candidates[i].get("candidate_key", f"cand_{i}"), float(scores[i]))
        for i in range(len(candidates))
    ]
    result.sort(key=lambda x: x[1], reverse=True)
    return result

def improved_hybrid_score_matrix(
    audit_mat: np.ndarray,
    freq_vec: np.ndarray,
    term_weights: np.ndarray | None = None,
) -> np.ndarray:
    if term_weights is None:
        term_weights = np.ones_like(freq_vec, dtype=np.float32)
    text_scalar = np.dot(freq_vec, term_weights)
    signature = path_signature_approx(audit_mat).astype(np.float32)
    audit_scalar = audit_mat.astype(np.float32) @ signature
    return np.maximum(audit_scalar * text_scalar, 0)

def improved_evaluate_candidates(
    manifest_path: Path,
    document: str,
    root_dir: Path = Path(__file__).resolve().parents[1],
) -> List[Tuple[str, float]]:
    data = load_manifest(manifest_path)
    candidates = data.get("vendors", [])
    if not candidates:
        return []

    audit_mat = build_audit_matrix(candidates)
    terms = load_go_terms(root_dir)
    freq_vec = term_frequency_vector(document, terms)

    scores = improved_hybrid_score_matrix(audit_mat, freq_vec)
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
    manifest_path = Path("manifest.json")
    document = "This is a sample document."
    result = improved_evaluate_candidates(manifest_path, document)
    print(result)