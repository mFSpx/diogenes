#!/usr/bin/env python3
"""Decision hygiene scoring algorithms.

Reusable deterministic text-feature counts and scoring. Scripts own DB scans/writes.
No LLM calls.
"""
from __future__ import annotations

import re
import statistics
from typing import Any

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


def excerpt(text: str, n: int = 260) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:n]


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
