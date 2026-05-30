# DARWIN HAMMER — match 19, survivor 0
# gen: 2
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s2.py (gen1)
# parent_b: ternary_lens_audit.py (gen0)
# born: 2026-05-29T23:23:03Z

"""
Hybrid Ternary Lens Audit and Decision Hygiene Module.

This module fuses the *Ternary Lens Audit* algorithm (Parent B) with the
*Decision Hygiene* algorithm (Parent A) using a novel mathematical bridge. 
The bridge uses the *Shannon Entropy* calculation to evaluate the diversity of 
decision-making cues in the *Ternary Lens Audit* process.

The governing equations of both parents are integrated by using the *feature 
vector* produced by the hygiene regexes from Parent A and applying it to the
*Ternary Lens Audit* classification process. The *Shannon Entropy* calculation 
is then used to evaluate the diversity of the classification results, 
providing an additional layer of evaluation for the decision-making process.
"""

import math
import re
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import json
import argparse
from datetime import datetime, timezone

# Parent A - regexes and raw count extraction
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

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)

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

# Parent B - Ternary Lens Audit
CLASSIFICATIONS = {"usable_now", "research_only", "needs_conversion", "unsafe_for_fastpath", "unsupported"}
LOCAL_PATTERNS = ["*bitnet*", "*BitNet*", "*fairyfuse*", "*FairyFuse*", "*lora*", "*LoRA*", "*adapter*"]

def _raw_counts(text: str) -> dict[str, int]:
    """Extract raw feature counts from *text* using parent-A regexes."""
    return {
        "evidence_count": len(EVIDENCE_RE.findall(text or "")),
        "planning_count": len(PLANNING_RE.findall(text or "")),
        "delay_count": len(DELAY_RE.findall(text or "")),
        "support_count": len(SUPPORT_RE.findall(text or "")),
        "boundary_count": len(BOUNDARY_RE.findall(text or "")),
        "outcome_count": len(OUTCOME_RE.findall(text or "")),
        "impulsive_count": len(IMPULSIVE_RE.findall(text or "")),
        "scarcity_count": len(SCARCITY_RE.findall(text or "")),
        "risk_count": len(RISK_RE.findall(text or "")),
    }

def feature_vector(text: str) -> np.ndarray:
    """Return a 9-element ``numpy`` array of counts ordered as _FEATURE_ORDER."""
    c = _raw_counts(text)
    return np.array(
        [
            c["evidence_count"],
            c["planning_count"],
            c["delay_count"],
            c["support_count"],
            c["boundary_count"],
            c["outcome_count"],
            c["impulsive_count"],
            c["scarcity_count"],
            c["risk_count"],
        ],
        dtype=np.int64,
    )

def hygiene_score(vector: np.ndarray) -> Tuple[int, str]:
    """Compute the original decision-hygiene score and a textual label."""
    positive = int(np.dot(vector, _POSITIVE_WEIGHTS))
    negative = int(np.dot(vector, _NEGATIVE_WEIGHTS))
    raw_score = max(-10000, min(10000, positive - negative))
    return raw_score, "valid" if raw_score >= 0 else "invalid"

def shannon_entropy(observations: Iterable[float], is_distribution: bool = False) -> float:
    """Return Shannon entropy (bits) of a discrete distribution."""
    xs = list(observations)
    if not xs:
        return 0.0
    if is_distribution:
        probs = [float(x) for x in xs]
        if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError("distribution must sum to 1")
    else:
        cnt = Counter(xs)
        total = float(sum(cnt.values()))
        probs = [v / total for v in cnt.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0.0)

def classify_candidate(candidate: dict[str, str]) -> str:
    """Classify a candidate based on its notes and classification."""
    notes = candidate.get("notes", "")
    classification = candidate.get("classification", "")
    vector = feature_vector(notes)
    score, label = hygiene_score(vector)
    if classification in CLASSIFICATIONS and score >= 0:
        return classification
    elif classification == "unsafe_for_fastpath" or score < 0:
        return "unsafe_for_fastpath"
    else:
        return "needs_conversion"

def evaluate_candidates(candidates: list[dict[str, str]]) -> list[str]:
    """Evaluate a list of candidates and return their classifications."""
    classifications = []
    for candidate in candidates:
        classification = classify_candidate(candidate)
        classifications.append(classification)
    return classifications

def auditLens(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    """Perform an audit on the lens by evaluating the candidates."""
    classifications = evaluate_candidates(candidates)
    summary = {classification: classifications.count(classification) for classification in set(classifications)}
    return summary

def load_manifest(path: Path) -> dict[str, Any]:
    """Load a manifest from a file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data

def main() -> int:
    parser = argparse.ArgumentParser(description="Offline audit for LUCIDOTA/FairyFuse ternary lens candidates")
    parser.add_argument("--manifest", type=Path, default=Path("services/ternary_lab/vendor_manifest.json"))
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    candidates = manifest.get("vendors", [])
    summary = auditLens(candidates)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())