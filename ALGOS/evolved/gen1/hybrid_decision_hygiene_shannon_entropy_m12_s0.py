# DARWIN HAMMER — match 12, survivor 0
# gen: 1
# parent_a: decision_hygiene.py (gen0)
# parent_b: shannon_entropy.py (gen0)
# born: 2026-05-29T23:18:35Z

#!/usr/bin/env python3
"""Hybrid algorithm combining decision hygiene scoring from decision_hygiene.py and Shannon entropy calculation from shannon_entropy.py.

The mathematical bridge between the two parent algorithms lies in using the Shannon entropy calculation to analyze the distribution of decision hygiene scores. This allows for a more detailed understanding of the decision-making process, incorporating both the scoring system and the information-theoretic properties of the scores.
"""

import numpy as np
import re
import math
from collections import Counter
from typing import Any

EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|first|after|review)\b", re.I)
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


def shannon_entropy(observations: list[int], is_distribution: bool = False) -> float:
    if not observations: return 0.0
    if is_distribution:
        probs = [float(x) for x in observations]
        if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError("distribution must sum to 1")
    else:
        c = Counter(observations)
        total = sum(c.values())
        probs = [v / total for v in c.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def analyze_scores(scores: list[int]) -> tuple[float, float]:
    """Analyze the distribution of decision hygiene scores using Shannon entropy calculation."""
    entropy = shannon_entropy(scores)
    avg_score = np.mean(scores)
    return entropy, avg_score


def compare_decision_hygiene(text1: str, text2: str) -> tuple[tuple[int, str], tuple[int, str]]:
    """Compare the decision hygiene scores of two texts."""
    c1 = counts(text1)
    c2 = counts(text2)
    score1, label1 = score_features(c1)
    score2, label2 = score_features(c2)
    return (score1, label1), (score2, label2)


def hybrid_analysis(texts: list[str]) -> tuple[float, list[tuple[int, str]]]:
    """Perform a hybrid analysis of a list of texts, calculating both decision hygiene scores and Shannon entropy."""
    scores = []
    labels = []
    for text in texts:
        c = counts(text)
        score, label = score_features(c)
        scores.append(score)
        labels.append(label)
    entropy, avg_score = analyze_scores(scores)
    return entropy, list(zip(scores, labels))


if __name__ == "__main__":
    text1 = "I have a plan to achieve my goals, and I will review it regularly."
    text2 = "I'm feeling overwhelmed and don't know what to do."
    score1, label1 = score_features(counts(text1))
    score2, label2 = score_features(counts(text2))
    print(f"Score 1: {score1}, Label 1: {label1}")
    print(f"Score 2: {score2}, Label 2: {label2}")
    entropy, scores = hybrid_analysis([text1, text2])
    print(f"Entropy: {entropy}")
    print(f"Scores: {scores}")