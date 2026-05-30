# DARWIN HAMMER — match 12, survivor 1
# gen: 1
# parent_a: decision_hygiene.py (gen0)
# parent_b: shannon_entropy.py (gen0)
# born: 2026-05-29T23:18:35Z

"""
This module integrates the decision_hygiene and shannon_entropy algorithms into a single hybrid system.
The bridge between the two structures is the concept of information entropy, which can be applied to the decision hygiene scoring system.
By calculating the Shannon entropy of the decision hygiene feature counts, we can gain insights into the complexity and uncertainty of the decision-making process.
"""

import re
import statistics
from typing import Any
import math
from collections import Counter
import numpy as np
import random
import sys
import pathlib

EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b", re.I)
SUPPORT_RE = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I)
OUTCOME_RE = re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I)
IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b", re.I)
SCARCITY_RE = re.compile(r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b", re.I)
RISK_RE = re.compile(r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b", re.I)


def counts(text: str) -> dict[str, int]:
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


def shannon_entropy(observations: list[int | float], is_distribution: bool = False) -> float:
    xs = list(observations)
    if not xs:
        return 0.0
    if is_distribution:
        probs = [float(x) for x in xs]
        if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError("distribution must sum to 1")
    else:
        c = Counter(xs)
        total = sum(c.values())
        probs = [v / total for v in c.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def score_features(c: dict[str, int]) -> tuple[int, str]:
    positive = (
        c["evidence_count"] * 1600
        + c["planning_count"] * 1200
        + c["delay_count"] * 1400
        + c["support_count"] * 1000
        + c["boundary_count"] * 1200
        + c["outcome_count"] * 800
    )
    negative = c["impulsive_count"] * 1500 + c["scarcity_count"] * 700 + c["risk_count"] * 1200
    score = max(-10000, min(10000, positive - negative))
    if c["risk_count"] and score < 2500:
        label = "critical_risk_or_pain_signal"
    elif score >= 7000:
        label = "high_decision_hygiene"
    elif score >= 3000:
        label = "improving_decision_hygiene"
    elif score <= -2500:
        label = "strained_decision_context"
    else:
        label = "neutral_or_unclear"
    return score, label


def hybrid_score(text: str) -> tuple[float, int, str]:
    feature_counts = counts(text)
    decision_hygiene_score, label = score_features(feature_counts)
    shannon_entropy_score = shannon_entropy(list(feature_counts.values()))
    return shannon_entropy_score, decision_hygiene_score, label


def monthly(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[int]] = {}
    for r in rows:
        ts = r.get("occurred_at")
        key = str(ts)[:7] if ts else "unknown"
        buckets.setdefault(key, []).append(int(r["score"]))
    return [{"month": k, "n": len(v), "avg_score": round(sum(v) / len(v), 2), "median_score": statistics.median(v)} for k, v in sorted(buckets.items())]


def compare_halves(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dated = [r for r in rows if r.get("occurred_at") is not None]
    dated.sort(key=lambda r: r["occurred_at"])
    if len(dated) < 2:
        return {"available": False, "reason": "need at least two dated decision signals"}
    half = len(dated) // 2
    early = dated[:half]
    late = dated[half:]

    def avg(rs: list[dict[str, Any]]) -> float:
        return sum(int(r["score"]) for r in rs) / len(rs) if rs else 0.0

    early_avg = avg(early)
    late_avg = avg(late)
    delta = late_avg - early_avg
    return {
        "available": True,
        "early_n": len(early),
        "late_n": len(late),
        "early_avg_score": round(early_avg, 2),
        "late_avg_score": round(late_avg, 2),
        "delta": round(delta, 2),
        "interpretation": "better_decision_hygiene_signal" if delta > 500 else "roughly_flat" if abs(delta) <= 500 else "more_strained_late_signal",
    }


if __name__ == "__main__":
    text = "This is a test text with some decision hygiene features."
    shannon_entropy_score, decision_hygiene_score, label = hybrid_score(text)
    print(f"Shannon Entropy Score: {shannon_entropy_score}")
    print(f"Decision Hygiene Score: {decision_hygiene_score}")
    print(f"Label: {label}")