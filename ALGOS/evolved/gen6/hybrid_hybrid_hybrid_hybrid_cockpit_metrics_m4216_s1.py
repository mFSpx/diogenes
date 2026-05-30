# DARWIN HAMMER — match 4216, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2739_s1.py (gen5)
# parent_b: cockpit_metrics.py (gen0)
# born: 2026-05-29T23:54:10Z

"""
Hybrid Algorithm: evidence_label_fusion.py

This module fuses the core topologies of two parent algorithms:

* **Parent A** – a text‑analysis system that extracts categorical evidence using
  compiled regular expressions (EVIDENCE_RE, PLANNING_RE, …).  Its output can be
  interpreted as a vector **c** of claim counts per category.

* **Parent B** – a lightweight “cockpit” metric suite that evaluates honesty and
  coverage through simple rational functions:
    - `anti_slop_ratio(claims_with_evidence, total_claims_emitted)`
    - `cockpit_honesty(displayed_ok, unknown_displayed_as_ok)`
    - `audit_debt(exports_missing_audit_step)`

**Mathematical Bridge**  
We treat each regex match from Parent A as a *claim*.  The subset of matches that
fall under the `EVIDENCE_RE` pattern are the *evidence‑supported* claims.
Thus we obtain the two scalars required by Parent B:


claims_with_evidence = count(EVIDENCE_RE matches)
total_claims_emitted = Σ count(all regex categories)


The bridge consists of feeding these scalars into the cockpit metrics, then
using the resulting “trust weight” to scale the categorical probabilities
derived from Parent A.  The hybrid score for a category *i* is therefore


score_i = trust_weight * (count_i / total_claims_emitted)


where `trust_weight = anti_slop_ratio(claims_with_evidence, total_claims_emitted)`.

The following implementation provides three public functions that embody this
fusion and a small smoke test.
"""

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regex based claim extraction
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
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b|"
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted|depressed|stress|anxiety|overworked|jobless)\b",
    re.I,
)

# Mapping of pattern names to compiled regex objects
PATTERN_MAP = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
    "scarcity": SCARCITY_RE,
}

# ----------------------------------------------------------------------
# Parent B – cockpit metric functions
# ----------------------------------------------------------------------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Return a trust weight in [0,1] based on evidence coverage."""
    if total_claims_emitted <= 0:
        return 1.0
    ratio = claims_with_evidence / total_claims_emitted
    return max(0.0, min(1.0, ratio))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Honesty metric for displayed vs. unknown claims."""
    total = displayed_ok + unknown_displayed_as_ok
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, displayed_ok / total))


def audit_debt(exports_missing_audit_step: int) -> float:
    """Simple debt count (non‑negative)."""
    return float(max(0, exports_missing_audit_step))


# ----------------------------------------------------------------------
# Hybrid functionality
# ----------------------------------------------------------------------
def extract_claim_counts(text: str) -> Counter:
    """
    Scan *text* with all Parent‑A regexes and return a Counter mapping
    category → number of matches.
    """
    counts = Counter()
    for name, regex in PATTERN_MAP.items():
        matches = regex.findall(text)
        counts[name] = len(matches)
    return counts


def compute_trust_weight(claim_counts: Counter) -> float:
    """
    Derive the trust weight using Parent‑B's anti_slop_ratio.
    Evidence claims are taken from the 'evidence' bucket.
    """
    evidence = claim_counts.get("evidence", 0)
    total = sum(claim_counts.values())
    return anti_slop_ratio(evidence, total)


def hybrid_category_scores(claim_counts: Counter) -> dict:
    """
    Produce a trust‑scaled probability distribution over categories.
    Each raw frequency is multiplied by the trust weight and normalised.
    Returns a dict ``category → score`` where scores sum to 1 (or 0 if no claims).
    """
    trust = compute_trust_weight(claim_counts)
    total = sum(claim_counts.values())
    if total == 0:
        return {cat: 0.0 for cat in claim_counts}
    raw = np.array([claim_counts[cat] for cat in claim_counts], dtype=float)
    scaled = trust * raw
    normalized = scaled / scaled.sum()
    return dict(zip(claim_counts.keys(), normalized.tolist()))


def hybrid_honesty_report(claim_counts: Counter) -> dict:
    """
    Combine multiple cockpit metrics into a single report.
    - `trust_weight`  : anti_slop_ratio
    - `honesty`      : cockpit_honesty (treat 'evidence' as displayed_ok,
                     all other categories as unknown_displayed_as_ok)
    - `audit_debt`   : number of categories with zero matches (simulated debt)
    """
    trust = compute_trust_weight(claim_counts)
    displayed_ok = claim_counts.get("evidence", 0)
    unknown_ok = sum(v for k, v in claim_counts.items() if k != "evidence")
    honesty = cockpit_honesty(displayed_ok, unknown_ok)
    debt = audit_debt(len([c for c in claim_counts.values() if c == 0]))
    return {
        "trust_weight": trust,
        "honesty": honesty,
        "audit_debt": debt,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = """
    I have verified the source and attached a screenshot as proof.
    Our next steps include a detailed plan and timeline.
    However, we might need to pause the rollout until tomorrow.
    Please ask the support team for any help.
    It's critical we respect privacy and set clear boundaries.
    The deployment was completed and verified as green.
    I feel stressed and can't afford any more delays.
    """
    counts = extract_claim_counts(sample_text)
    print("Claim counts per category:")
    print(json.dumps(counts, indent=2))

    scores = hybrid_category_scores(counts)
    print("\nTrust‑scaled category scores:")
    print(json.dumps(scores, indent=2))

    report = hybrid_honesty_report(counts)
    print("\nHybrid honesty report:")
    print(json.dumps(report, indent=2))