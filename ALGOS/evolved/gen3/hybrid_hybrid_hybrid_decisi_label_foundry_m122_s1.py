# DARWIN HAMMER — match 122, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s4.py (gen2)
# parent_b: label_foundry.py (gen0)
# born: 2026-05-29T23:26:55Z

"""Hybrid algorithm combining regex‑based feature scoring (Parent A) with weak‑supervision
label aggregation (Parent B).

Mathematical bridge:
- Parent A produces a feature count vector **c** ∈ ℕ⁹ for a document, one entry per
  regex pattern.
- Parent B treats each weak labeling function as a binary vote and aggregates votes
  with equal weight.

In the hybrid we interpret the positive weight vector **w** (from Parent A) as a
weighting of the binary votes.  The aggregated confidence is the weighted average

    confidence = (w·ℓ) / Σ w

where ℓ ∈ {0,1}⁹ are the binary outputs of the labeling functions derived from the
features.  This unifies the two topologies: regex detection supplies the weak
labels, and the weighted vote supplies a probabilistic label identical in spirit
to Parent B’s `aggregate_labels`."""
from __future__ import annotations

import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Regex feature set – from Parent A
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

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)

# ----------------------------------------------------------------------
# Weak‑supervision primitives – from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  # 0 or 1


@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int          # 0 or 1
    confidence: float   # in [0,1]


@dataclass(frozen=True)
class LabelError:
    doc_id: str
    given_label: int
    suggested_label: int
    error_probability: float


def labeling_function(name: str | None = None):
    """Decorator that annotates a function as a weak labeling function."""
    def deco(fn: Callable[[dict], int]) -> Callable[[dict], int]:
        fn.lf_name = name or fn.__name__
        return fn
    return deco


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def extract_feature_counts(text: str) -> Dict[str, int]:
    """Return a dict mapping each feature name to the number of regex matches."""
    counts = {}
    regexes = {
        "evidence": EVIDENCE_RE,
        "planning": PLANNING_RE,
        "delay": DELAY_RE,
        "support": SUPPORT_RE,
        "boundary": BOUNDARY_RE,
        "outcome": OUTCOME_RE,
        "impulsive": IMPULSIVE_RE,
        "scarcity": SCARCITY_RE,
        "risk": RISK_RE,
    }
    for name in _FEATURE_ORDER:
        counts[name] = len(regexes[name].findall(text))
    return counts


def _make_lf_from_feature(feature: str) -> Callable[[dict], int]:
    """Factory that creates a binary labeling function for a given feature."""
    @labeling_function(name=f"lf_{feature}")
    def lf(doc: dict) -> int:
        # doc must contain a pre‑computed 'features' dict
        return 1 if doc["features"].get(feature, 0) > 0 else 0
    return lf


# Dynamically create labeling functions for each feature
_LABELING_FUNCTIONS: List[Callable[[dict], int]] = [
    _make_lf_from_feature(f) for f in _FEATURE_ORDER
]


def apply_labeling_functions(doc_id: str, text: str) -> List[LabelingFunctionResult]:
    """
    Run all feature‑based labeling functions on *text* and return the raw results.
    """
    features = extract_feature_counts(text)
    doc = {"id": doc_id, "features": features}
    results: List[LabelingFunctionResult] = []
    for lf in _LABELING_FUNCTIONS:
        label = lf(doc)
        results.append(LabelingFunctionResult(lf_name=lf.lf_name, doc_id=doc_id, label=label))
    return results


def aggregate_weighted_labels(batch: List[LabelingFunctionResult]) -> ProbabilisticLabel:
    """
    Weighted aggregation analogous to Parent B's `aggregate_labels`, but using the
    positive‑weight vector from Parent A.

    confidence = (w·ℓ) / Σ w   where ℓ are binary votes.
    """
    # Map feature name → weight index
    weight_index = {f"lf_{name}": i for i, name in enumerate(_FEATURE_ORDER)}
    total_weight = float(_POSITIVE_WEIGHTS.sum())
    if total_weight == 0:
        raise ValueError("Sum of positive weights must be non‑zero.")

    weighted_sum = 0.0
    for r in batch:
        idx = weight_index.get(r.lf_name)
        if idx is not None:
            weighted_sum += _POSITIVE_WEIGHTS[idx] * r.label

    confidence = weighted_sum / total_weight
    label = 1 if confidence >= 0.5 else 0
    doc_id = batch[0].doc_id if batch else "unknown"
    return ProbabilisticLabel(doc_id=doc_id, label=label, confidence=confidence)


def find_label_errors(
    docs: List[dict],
    given: List[int],
    probs: List[float],
    threshold: float = 0.65,
) -> List[LabelError]:
    """
    Identical to Parent B's implementation – retained for compatibility.
    """
    if not (len(docs) == len(given) == len(probs)):
        raise ValueError("length mismatch")
    errs: List[LabelError] = []
    for doc, g, p in zip(docs, given, probs):
        errp = p if g == 0 else 1.0 - p
        if errp >= threshold:
            errs.append(
                LabelError(
                    doc_id=str(doc.get("id", len(errs))),
                    given_label=g,
                    suggested_label=1 - g,
                    error_probability=errp,
                )
            )
    return sorted(errs, key=lambda e: -e.error_probability)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "I have evidence that the plan is solid and we can ship it tomorrow.",
        "I'm scared, I might kill myself. Please help, I need support now.",
        "No money left, broke, can't afford rent. Need a loan.",
        "Let's do it right now, no delay, just burn it down!",
    ]

    all_results: List[ProbabilisticLabel] = []
    for i, txt in enumerate(sample_texts):
        doc_id = f"doc_{i}"
        lf_results = apply_labeling_functions(doc_id, txt)
        agg = aggregate_weighted_labels(lf_results)
        print(f"{doc_id}: confidence={agg.confidence:.3f}, label={agg.label}")
        all_results.append(agg)

    # Demonstrate error detection with artificial given labels
    given_labels = [1, 0, 0, 1]
    probs = [pl.confidence for pl in all_results]
    errors = find_label_errors(
        docs=[{"id": pl.doc_id} for pl in all_results],
        given=given_labels,
        probs=probs,
        threshold=0.6,
    )
    print("\nPotential label errors:")
    for e in errors:
        print(f"- {e.doc_id}: given={e.given_label}, suggested={e.suggested_label}, prob={e.error_probability:.2f}")