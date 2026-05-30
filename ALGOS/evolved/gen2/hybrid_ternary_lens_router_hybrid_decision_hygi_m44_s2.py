# DARWIN HAMMER — match 44, survivor 2
# gen: 2
# parent_a: ternary_lens_router.py (gen0)
# parent_b: hybrid_decision_hygiene_shannon_entropy_m12_s0.py (gen1)
# born: 2026-05-29T23:23:38Z

"""Hybrid Ternary‑Decision Hygiene Analyzer.

Parent A: ternary_lens_router.py – provides cryptographic payload hashing,
stable primitive IDs, ternary vectors and confidence basis‑points.
Parent B: hybrid_decision_hygiene_shannon_entropy_m12_s0.py – supplies a
decision‑hygiene scoring system and a Shannon‑entropy evaluator.

Mathematical bridge:
Both parents produce discrete integer sequences that can be interpreted as
probability‑mass samples.  The ternary vector (values ∈ {-1,0,1}) and the
decision‑hygiene scores (continuous integers) are each mapped to a common
ternary alphabet (−1, 0, +1) and concatenated into a single hybrid vector.
Shannon entropy is then computed over the empirical distribution of this
combined vector, yielding a single information‑theoretic measure that
reflects both low‑level payload characteristics and high‑level decision
quality.  Additional hybrid metrics (e.g. confidence‑adjusted scores) are
derived by linearly mixing the original confidence basis‑points with the
average decision‑hygiene score.

The module implements the full pipeline while remaining self‑contained and
executable with only the Python standard library and NumPy.
"""

import argparse
import collections
import hashlib
import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A utilities (trimmed to essentials)
# ----------------------------------------------------------------------
TERNARY_DIMS = 12


def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601 without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def payload_hash(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> str:
    """Deterministic SHA‑256 of the command envelope."""
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ternary_vector(
    raw_command: str, normalized_intent: str, context: dict[str, Any], dims: int = TERNARY_DIMS
) -> List[int]:
    """Map a payload to a fixed‑size ternary vector (values ∈ {-1,0,1})."""
    digest = hashlib.sha256(
        (raw_command + "\0" + normalized_intent + "\0" + json.dumps(context, sort_keys=True)).encode()
    ).digest()
    return [((digest[i] % 3) - 1) for i in range(dims)]


def confidence_bps(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> int:
    """Heuristic confidence expressed in basis points (0 – 9900)."""
    base = 4500
    if normalized_intent.strip():
        base += 1800
    if raw_command.strip():
        base += 1200
    if context:
        base += 800
    return max(0, min(9900, base))


# ----------------------------------------------------------------------
# Parent‑B utilities (trimmed to essentials)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|first|after|review)\b", re.I)
SUPPORT_RE = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I)
OUTCOME_RE = re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I)
IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b", re.I)
SCARCITY_RE = re.compile(r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b", re.I)
RISK_RE = re.compile(r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b", re.I)


def _counts(text: str) -> dict[str, int]:
    """Count occurrences of each lexical category."""
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


def decision_hygiene_score(counts: dict[str, int]) -> Tuple[int, str]:
    """Compute a bounded hygiene score and a categorical label."""
    positive = (
        counts["evidence_count"] * 1600
        + counts["planning_count"] * 1200
        + counts["delay_count"] * 1400
        + counts["support_count"] * 1000
        + counts["boundary_count"] * 1200
        + counts["outcome_count"] * 800
    )
    negative = (
        counts["impulsive_count"] * 1500
        + counts["scarcity_count"] * 700
        + counts["risk_count"] * 1200
    )
    score = max(-10000, min(10000, positive - negative))

    if counts["risk_count"] and score < 2500:
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


def shannon_entropy(observations: List[int]) -> float:
    """Classic Shannon entropy over the empirical distribution of integer symbols."""
    if not observations:
        return 0.0
    counter = collections.Counter(observations)
    total = sum(counter.values())
    probs = [cnt / total for cnt in counter.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0.0)


# ----------------------------------------------------------------------
# Hybrid layer – mathematical fusion of both parents
# ----------------------------------------------------------------------
def _score_to_ternary(score: int) -> int:
    """Map a decision‑hygiene score to the ternary alphabet."""
    if score >= 7000:
        return 1
    if score <= -2500:
        return -1
    return 0


def hybrid_vector(
    raw_command: str,
    normalized_intent: str,
    context: dict[str, Any],
    decision_texts: List[str],
) -> List[int]:
    """
    Produce a unified vector:

    * First TERNARY_DIMS entries come from the cryptographic ternary_vector().
    * Subsequent entries are ternary‑mapped decision‑hygiene scores for each
      supplied text (one entry per text).

    The resulting integer sequence lives in the same algebraic space for both
    parents, enabling joint statistical analysis.
    """
    base_vec = ternary_vector(raw_command, normalized_intent, context, dims=TERNARY_DIMS)
    decision_vec = [_score_to_ternary(decision_hygiene_score(_counts(t))[0]) for t in decision_texts]
    return base_vec + decision_vec


def hybrid_entropy(
    raw_command: str,
    normalized_intent: str,
    context: dict[str, Any],
    decision_texts: List[str],
) -> float:
    """
    Compute Shannon entropy of the hybrid vector defined by `hybrid_vector`.
    The entropy quantifies the informational richness of the combined payload
    and decision‑hygiene signals.
    """
    vec = hybrid_vector(raw_command, normalized_intent, context, decision_texts)
    return shannon_entropy(vec)


def hybrid_confidence(
    raw_command: str,
    normalized_intent: str,
    context: dict[str, Any],
    decision_texts: List[str],
) -> float:
    """
    Blend the original confidence basis‑points with the average decision‑hygiene
    score (scaled to the 0‑1 interval).  The formula is:

        hybrid = 0.6 * (confidence_bps / 9900) + 0.4 * ((avg_score + 10000) / 20000)

    Result lies in [0, 1] where higher values indicate both cryptographic
    confidence and sound decision context.
    """
    conf = confidence_bps(raw_command, normalized_intent, context) / 9900.0
    scores = [decision_hygiene_score(_counts(t))[0] for t in decision_texts]
    if scores:
        avg_score = np.mean(scores)
        norm_score = (avg_score + 10000) / 20000.0  # map [-10000,10000] → [0,1]
    else:
        norm_score = 0.5  # neutral default
    return 0.6 * conf + 0.4 * norm_score


# ----------------------------------------------------------------------
# Command‑line interface (dry‑run friendly)
# ----------------------------------------------------------------------
def _parse_context(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        val = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(val, dict):
        raise SystemExit("context must be a JSON object")
    return val


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid ternary / decision‑hygiene analyzer")
    parser.add_argument("--raw-command", required=True, help="Original command string")
    parser.add_argument("--normalized-intent", required=True, help="Normalized intent")
    parser.add_argument("--context", default="{}", help="JSON object with auxiliary context")
    parser.add_argument("--texts", nargs="*", default=[], help="One or more decision‑hygiene texts")
    args = parser.parse_args()

    context = _parse_context(args.context)

    # Core hybrid metrics
    hv = hybrid_vector(args.raw_command, args.normalized_intent, context, args.texts)
    he = hybrid_entropy(args.raw_command, args.normalized_intent, context, args.texts)
    hc = hybrid_confidence(args.raw_command, args.normalized_intent, context, args.texts)

    output = {
        "timestamp": utc_now(),
        "payload_sha256": payload_hash(args.raw_command, args.normalized_intent, context),
        "hybrid_vector": hv,
        "hybrid_entropy": he,
        "hybrid_confidence": hc,
    }

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    # Smoke test – runs with deterministic stub data if no CLI arguments are given.
    if len(sys.argv) == 1:
        test_args = [
            "--raw-command", "run diagnostics",
            "--normalized-intent", "execute",
            "--context", '{"user":"alice","env":"test"}',
            "--texts",
            "I have a plan and will verify each step.",
            "I'm stuck and need help now."
        ]
        sys.argv.extend(test_args)
    sys.exit(main())