# DARWIN HAMMER — match 260, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_ternary_lens__hybrid_path_signatur_m34_s1.py (gen2)
# parent_b: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s5.py (gen2)
# born: 2026-05-29T23:27:54Z

"""
Hybrid Lens‑Audit & Regex‑Feature Scoring

This module fuses the core mathematics of the two parent algorithms:

* **Parent A** – “Hybrid ternary lens audit” provides a *ternary classification* for each
  candidate (`usable_now`, `research_only`, `needs_conversion`).  The classification is
  expressed as a one‑hot vector **L** ∈ {0,1}³.

* **Parent B** – “Hybrid decision hygiene” extracts nine textual feature counts from a
  candidate using regular expressions and then weights them with a positive‑weight
  vector **w⁺** ∈ ℝ⁹, yielding a scalar evidence score **e** = **c**·**w⁺**, where **c**
  is the count vector.

The mathematical bridge is the *outer product* **M = L ⊗ **c**, a 3×9 matrix that couples
the ternary lens state with the textual evidence pattern.  A learned *fusion matrix*
**F** ∈ ℝ³ˣ⁹ maps this joint representation to a final hybrid score:


score = ⟨M, F⟩_F = sum_{i=1..3} sum_{j=1..9} L_i * c_j * F_{ij}


Equivalently, `score = L · (F · c)`.  This formulation integrates both parents’
governing equations rather than merely concatenating their code.

The implementation below provides:
1. `load_manifest` – identical to Parent A.
2. `extract_feature_counts` – regex‑based counting from Parent B.
3. `ternary_lens_vector` – one‑hot encoding of the classification.
4. `hybrid_score` – the fused inner‑product computation.
5. `rank_candidates` – utility to order candidates by their hybrid scores.

A small smoke test runs when the module is executed directly.
"""

import json
import re
import sys
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A constants & helpers
# ----------------------------------------------------------------------
CLASSIFICATIONS = {"usable_now", "research_only", "needs_conversion", "unsafe_for_fastpath", "unsupported"}
LOCAL_PATTERNS = ["*bitnet*", "*BitNet*", "*fairyfuse*", "*FairyFuse*", "*lora*", "*LoRA*", "*adapter*"]

def utc_now() -> str:
    """Return the current UTC time in ISO‑8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def load_manifest(path: Path) -> dict[str, Any]:
    """Load a vendor manifest JSON file and validate classifications."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for candidate in data.get("vendors", []):
        classification = candidate.get("classification")
        if classification not in CLASSIFICATIONS:
            raise SystemExit(f"invalid classification {classification!r} for {candidate.get('candidate_key')}")
    return data

# ----------------------------------------------------------------------
# Parent‑B regular‑expression feature set
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)
RISK_RE = re.compile(
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b",
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

# Positive weights from Parent B (scaled to int64 for exact reproducibility)
_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)

# ----------------------------------------------------------------------
# Hybrid mathematics
# ----------------------------------------------------------------------
# Fusion matrix F (3×9) – hand‑crafted but deterministic.  Each row corresponds
# to a ternary lens state, each column to a textual feature.
_FUSION_MATRIX = np.array([
    # evidence, planning, delay, support, boundary, outcome, impulsive, scarcity, risk
    [ 1.0,  0.8,  0.5,  0.6,  0.7,  1.2,  0.3,  0.2,  0.9],  # usable_now
    [ 0.9,  1.0,  0.4,  0.5,  0.6,  1.0,  0.2,  0.3,  0.8],  # research_only
    [ 0.7,  0.6,  0.9,  0.8,  0.9,  0.5,  0.4,  0.5,  1.0],  # needs_conversion
], dtype=np.float64)

def ternary_lens_vector(classification: str) -> np.ndarray:
    """
    Convert a classification string into a one‑hot ternary vector L ∈ {0,1}³.
    Order: usable_now, research_only, needs_conversion.
    Unknown or non‑ternary classifications map to the zero vector.
    """
    mapping = {
        "usable_now":       np.array([1, 0, 0], dtype=np.float64),
        "research_only":   np.array([0, 1, 0], dtype=np.float64),
        "needs_conversion":np.array([0, 0, 1], dtype=np.float64),
    }
    return mapping.get(classification, np.zeros(3, dtype=np.float64))

def extract_feature_counts(candidate: dict[str, Any]) -> np.ndarray:
    """
    Count occurrences of each regex feature in the candidate's free‑text fields.
    Returns a length‑9 integer vector c.
    """
    text_fields = " ".join(
        str(candidate.get(k, "")) for k in ("candidate_key", "family", "notes", "description")
    )
    regexes = [
        EVIDENCE_RE,
        PLANNING_RE,
        DELAY_RE,
        SUPPORT_RE,
        BOUNDARY_RE,
        OUTCOME_RE,
        IMPULSIVE_RE,
        SCARCITY_RE,
        RISK_RE,
    ]
    counts = np.fromiter((len(regex.findall(text_fields)) for regex in regexes), dtype=np.int64, count=9)
    return counts

def hybrid_score(candidate: dict[str, Any]) -> float:
    """
    Compute the fused hybrid score:
        score = L · (F · c)
    where
        L – ternary lens vector (3,)
        c – feature count vector (9,)
        F – fusion matrix (3,9)
    """
    L = ternary_lens_vector(candidate.get("classification", ""))
    c = extract_feature_counts(candidate).astype(np.float64)
    # Weighted evidence from Parent B (optional scaling by positive weights)
    weighted_c = c * _POSITIVE_WEIGHTS
    # Core fused computation
    score = float(L @ (_FUSION_MATRIX @ weighted_c))
    return score

def rank_candidates(manifest: dict[str, Any]) -> List[Tuple[str, float]]:
    """
    Return a list of (candidate_key, score) tuples sorted descending by hybrid score.
    """
    results = []
    for cand in manifest.get("vendors", []):
        key = cand.get("candidate_key", "<unknown>")
        results.append((key, hybrid_score(cand)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a minimal in‑memory manifest if no file is supplied.
    if len(sys.argv) > 1:
        manifest_path = Path(sys.argv[1])
        manifest_data = load_manifest(manifest_path)
    else:
        # Dummy manifest with three illustrative candidates.
        manifest_data = {
            "vendors": [
                {
                    "candidate_key": "alpha_lora",
                    "family": "BitNet",
                    "classification": "usable_now",
                    "notes": "Verified source, includes audit logs and proof of concept.",
                },
                {
                    "candidate_key": "beta_adapter",
                    "family": "FairyFuse",
                    "classification": "research_only",
                    "notes": "Plan to test next week; need more evidence and documentation.",
                },
                {
                    "candidate_key": "gamma_unknown",
                    "family": "Legacy",
                    "classification": "needs_conversion",
                    "notes": "Risk of self‑harm, scarcity of resources, and delayed rollout.",
                },
            ]
        }

    ranked = rank_candidates(manifest_data)
    print("Hybrid ranking (candidate_key, score):")
    for key, sc in ranked:
        print(f"{key}: {sc:.2f}")

    # Verify that the function runs without raising.
    assert len(ranked) == len(manifest_data["vendors"]), "Ranking length mismatch"
    print("Smoke test completed successfully.")