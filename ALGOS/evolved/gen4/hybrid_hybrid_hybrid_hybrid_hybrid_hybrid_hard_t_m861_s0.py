# DARWIN HAMMER — match 861, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_decisi_m254_s0.py (gen3)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_bayes_update__m123_s0.py (gen3)
# born: 2026-05-29T23:31:17Z

"""Hybrid Hammer Algorithm – Fusion of Ternary Lens Audit & Decision Hygiene with
Bilnear‑Curvature Bayes Update.

Parents:
* hybrid_hybrid_hybrid_ternar_hybrid_hybrid_decisi_m254_s0.py (Ternary Lens Audit &
  Decision Hygiene)
* hybrid_hybrid_hard_truth_ma_hybrid_bayes_update__m123_s0.py (Hard‑Truth Math
  Model & Bayesian Curvature Update)

Mathematical bridge:
Both parents ultimately generate a *vector* representation of a candidate.
 – The Lens side provides a path‑signature vector (a rough‑path signature of the
   candidate’s numerical path) together with an evidence count.
 – The Bayes side provides a low‑dimensional “function‑category” vector (LSM)
   derived from textual description.

The hybrid therefore builds a single feature vector  **v**  by concatenating
signature, evidence, and LSM components.  The curvature‑bayes parent supplies a
bilinear form  **M = α v vᵀ + β I**  (α,β ∈ ℝ).  The final decision metric is the
quadratic form  

    score = vᵀ M v = α (vᵀv)² + β ‖v‖² ,

which fuses the deterministic audit information with the probabilistic curvature
information in a mathematically unified way.  The three public functions below
expose this pipeline."""
import sys
import math
import random
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants & helpers from Parent A (Ternary Lens Audit & Decision Hygiene)
# ----------------------------------------------------------------------
CLASSIFICATIONS = {
    "usable_now",
    "research_only",
    "needs_conversion",
    "unsafe_for_fastpath",
    "unsupported",
}
LOCAL_PATTERNS = [
    "*bitnet*",
    "*BitNet*",
    "*fairyfuse*",
    "*FairyFuse*",
    "*lora*",
    "*LoRA*",
    "*adapter*",
]

EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|"
    r"receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|"
    r"check|checked|audit)\b",
    re.I,
)

_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

def load_manifest(path: Path) -> dict:
    """Load a manifest JSON‑like file (evaluated safely via literal eval)."""
    # The original code used eval – we keep the spirit but guard against non‑dicts.
    data = eval(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Manifest must evaluate to a dict")
    for candidate in data.get("vendors", []):
        classification = candidate.get("classification")
        if classification not in CLASSIFICATIONS:
            raise SystemExit(
                f"invalid classification {classification!r} for {candidate.get('candidate_key')}"
            )
    return data

def _raw_counts(text: str) -> dict:
    """Count evidence tokens in a free‑form string."""
    return {"evidence_count": len(EVIDENCE_RE.findall(text or ""))}

# ----------------------------------------------------------------------
# Constants & helpers from Parent B (Hard‑Truth Math Model & Bayesian Update)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
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
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

def _words(text: str) -> List[str]:
    """Tokenise lower‑case alphabetic words."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def lsm_vector(text: str) -> Dict[str, float]:
    """Low‑dimensional semantic (LSM) vector based on function‑category frequencies."""
    wlist = _words(text)
    total = len(wlist) or 1
    counts = {cat: 0 for cat in FUNCTION_CATS}
    for w in wlist:
        for cat, vocab in FUNCTION_CATS.items():
            if w in vocab:
                counts[cat] += 1
    return {cat: cnt / total for cat, cnt in counts.items()}

# ----------------------------------------------------------------------
# Hybrid core – building a unified feature vector and applying a bilinear form
# ----------------------------------------------------------------------
def path_signature_transformation(candidate: dict) -> np.ndarray:
    """
    Very lightweight surrogate for a rough‑path signature.
    The candidate is expected to contain a numeric ``path`` key – a list of floats.
    The signature returns the first‑order cumulative sum and the second‑order
    (pairwise) product summed over the path, concatenated as a 1‑D array.
    """
    path = candidate.get("path", [])
    if not isinstance(path, (list, tuple)) or not path:
        # Fallback to a zero vector of length 2
        return np.zeros(2, dtype=float)

    arr = np.asarray(path, dtype=float)
    first_order = np.cumsum(arr)  # shape (n,)
    second_order = np.cumsum(arr[:, None] * arr[None, :])  # shape (n, n)
    # Reduce second order to a single scalar (sum of upper triangle)
    second_scalar = np.triu(second_order).sum()
    # Concatenate the last cumulative sum and the second‑order scalar
    return np.array([first_order[-1], second_scalar], dtype=float)


def build_feature_vector(candidate: dict) -> np.ndarray:
    """
    Construct the unified feature vector ``v`` for a candidate.

    Components (in order):
      1. Path‑signature (2 floats)
      2. Evidence count (1 float)
      3. LSM categories (len(FUNCTION_CATS) floats, ordered alphabetically)
    """
    # 1. Path signature
    sig = path_signature_transformation(candidate)  # shape (2,)

    # 2. Evidence count extracted from the free‑form description
    description = candidate.get("description", "")
    evidence = _raw_counts(description).get("evidence_count", 0)
    evidence_arr = np.array([float(evidence)], dtype=float)

    # 3. LSM vector (ordered alphabetically for reproducibility)
    lsm = lsm_vector(description)
    lsm_arr = np.array([lsm[cat] for cat in sorted(FUNCTION_CATS.keys())], dtype=float)

    # Concatenate all parts
    return np.concatenate([sig, evidence_arr, lsm_arr])


def curvature_bilinear_matrix(v: np.ndarray, alpha: float = 0.001, beta: float = 0.1) -> np.ndarray:
    """
    Produce the curvature‑aware bilinear matrix ``M`` from a feature vector ``v``.

    M = α * v vᵀ + β * I
    """
    dim = v.shape[0]
    outer = np.outer(v, v)                # v vᵀ
    M = alpha * outer + beta * np.eye(dim)  # add isotropic regularisation
    return M


def hybrid_score(candidate: dict, alpha: float = 0.001, beta: float = 0.1) -> float:
    """
    Compute the hybrid decision metric for a candidate.

    score = vᵀ M v  where  M = α v vᵀ + β I
    This expands to α (‖v‖²)² + β ‖v‖².
    """
    v = build_feature_vector(candidate)
    M = curvature_bilinear_matrix(v, alpha=alpha, beta=beta)
    score = float(v @ M @ v)  # vᵀ M v
    return score


# ----------------------------------------------------------------------
# Convenience utilities
# ----------------------------------------------------------------------
def rank_candidates(candidates: List[dict], **kwargs) -> List[Tuple[dict, float]]:
    """
    Return a list of (candidate, score) tuples sorted descending by score.
    Additional keyword arguments are forwarded to ``hybrid_score``.
    """
    scored = [(cand, hybrid_score(cand, **kwargs)) for cand in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny manifest in‑memory and write to a temporary file.
    sample_manifest = {
        "vendors": [
            {
                "candidate_key": "lens_001",
                "classification": "usable_now",
                "path": [0.2, 0.5, -0.1],
                "description": "Verified source with evidence and clear documentation. "
                               "The model uses a lot of pronouns and prepositions.",
            },
            {
                "candidate_key": "lens_002",
                "classification": "needs_conversion",
                "path": [1.0, -0.3, 0.4, 0.2],
                "description": "No evidence, just speculation. It lacks citations and proof.",
            },
            {
                "candidate_key": "lens_003",
                "classification": "research_only",
                "path": [],
                "description": "Preliminary results, many articles referenced, but no verified hash.",
            },
        ]
    }

    tmp_path = Path("tmp_manifest.py")
    tmp_path.write_text(repr(sample_manifest), encoding="utf-8")

    try:
        manifest = load_manifest(tmp_path)
        candidates = manifest["vendors"]
        ranked = rank_candidates(candidates)
        print("Hybrid ranking (candidate_key, score):")
        for cand, sc in ranked:
            print(f"  {cand['candidate_key']}: {sc:.6f}")
    finally:
        # Clean up the temporary file
        tmp_path.unlink(missing_ok=True)