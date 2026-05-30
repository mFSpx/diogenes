# DARWIN HAMMER — match 19, survivor 5
# gen: 2
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s2.py (gen1)
# parent_b: ternary_lens_audit.py (gen0)
# born: 2026-05-29T23:23:03Z

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Regex feature set – identical to parent A
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
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)

# ----------------------------------------------------------------------
# Parent‑B constants
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

# ----------------------------------------------------------------------
# Utility: Shannon entropy (parent B)
# ----------------------------------------------------------------------
def shannon_entropy(probs: Iterable[float]) -> float:
    """Return Shannon entropy (base‑2) of a probability distribution."""
    p = np.asarray(list(probs), dtype=float)
    if p.size == 0:
        return 0.0
    if np.any(p < 0):
        raise ValueError("probabilities must be non‑negative")
    total = p.sum()
    if not np.isclose(total, 1.0):
        raise ValueError("probabilities must sum to 1")
    # Guard against log2(0) – zero entries contribute nothing.
    nonzero = p[p > 0]
    return -float(np.sum(nonzero * np.log2(nonzero)))


# ----------------------------------------------------------------------
# Feature extraction (parent A core)
# ----------------------------------------------------------------------
def _raw_counts(text: str) -> dict[str, int]:
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


def _raw_hygiene_score(vector: np.ndarray) -> int:
    """Unclipped hygiene score (positive – negative)."""
    positive = int(np.dot(vector, _POSITIVE_WEIGHTS))
    negative = int(np.dot(vector, _NEGATIVE_WEIGHTS))
    return positive - negative


def hygiene_score(vector: np.ndarray) -> Tuple[int, str]:
    """Clip raw score to [-10 000, 10 000] and assign a qualitative label."""
    raw = max(-10_000, min(10_000, _raw_hygiene_score(vector)))
    if raw > 6_000:
        label = "EXCELLENT"
    elif raw > 2_000:
        label = "GOOD"
    elif raw > -2_000:
        label = "NEUTRAL"
    elif raw > -6_000:
        label = "POOR"
    else:
        label = "DISASTROUS"
    return raw, label


# ----------------------------------------------------------------------
# Deeper mathematical fusion
# ----------------------------------------------------------------------
_H_MAX = math.log2(len(_FEATURE_ORDER))  # log₂ 9 ≈ 3.1699


def _entropy_factor(vector: np.ndarray) -> float:
    """Return (1 + H / H_max) safely, handling zero‑vector cases."""
    total = int(vector.sum())
    if total == 0:
        return 1.0  # No information → neutral multiplier
    probs = vector.astype(float) / total
    H = shannon_entropy(probs)
    return 1.0 + H / _H_MAX


def _normalize_score(score: int) -> float:
    """Map raw hygiene score from [-10 000, 10 000] to [0, 1]."""
    return (score + 10_000) / 20_000.0


def hybrid_score(
    vector: np.ndarray,
    classification: str | None = None,
    alpha: float = 0.7,
) -> float:
    """
    Compute a deeper hybrid metric.

    The metric is a convex combination of a normalized hygiene component
    and an entropy‑richness component, optionally biased by the vendor
    classification.

    Parameters
    ----------
    vector:
        Feature count vector for the candidate.
    classification:
        Optional classification string; influences the final score.
    alpha:
        Weight of the hygiene component (0 ≤ α ≤ 1). The entropy component
        receives weight (1‑α). Default 0.7 favours hygiene while still
        rewarding information richness.

    Returns
    -------
    float
        Hybrid score in the range [0, 1].
    """
    raw_hygiene, _ = hygiene_score(vector)
    h_norm = _normalize_score(raw_hygiene)  # ∈ [0,1]

    # Entropy factor already lies in [1, 2] because H ≤ H_max.
    entropy_factor = _entropy_factor(vector)
    # Rescale to [0,1] for convex combination.
    e_norm = (entropy_factor - 1.0) / 1.0  # (H/H_max) ∈ [0,1]

    # Base convex combination.
    combined = alpha * h_norm + (1.0 - alpha) * e_norm

    # Classification‑based bias: penalise unsafe categories.
    if classification == "unsafe_for_fastpath":
        combined *= 0.85  # modest penalty
    elif classification == "unsupported":
        combined *= 0.70
    elif classification == "research_only":
        combined *= 0.90

    # Clamp to [0,1] to guard against floating‑point drift.
    return max(0.0, min(1.0, combined))


# ----------------------------------------------------------------------
# Parent‑B helpers (trimmed to essentials)
# ----------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for cand in data.get("vendors", []):
        classification = cand.get("classification")
        if classification not in CLASSIFICATIONS:
            raise SystemExit(
                f"invalid classification {classification!r} for {cand.get('candidate_key')}"
            )
    return data


def enforce_fast_path_rule(candidate: dict[str, Any]) -> List[str]:
    """
    Validate fast‑path compatibility rules and return a list of textual findings.
    """
    findings: List[str] = []
    key = candidate.get("candidate_key", "")
    family = candidate.get("family", "")
    notes = candidate.get("notes", "")

    # Rule 1 – standard LoRA / PEFT must be marked unsafe unless benchmarked.
    if re.search(r"standard.*lora|peft|qlora", f"{key} {family}", re.I):
        if candidate.get("classification") != "unsafe_for_fastpath" or candidate.get(
            "fast_path_compatible"
        ):
            findings.append(
                "STANDARD_LORA_RULE_VIOLATION: normal PEFT/QLoRA must be "
                "'unsafe_for_fastpath' unless benchmark proves hot‑path safety"
            )

    # Rule 2 – mixed precision hints conflict with explicit fast‑path flag.
    if re.search(r"fp16|fp32", notes, re.I) and candidate.get("fast_path_compatible"):
        findings.append(
            "FP_HOTPATH_CONFLICT: presence of fp16/fp32 in notes contradicts "
            "'fast_path_compatible' flag"
        )

    return findings


# ----------------------------------------------------------------------
# Main orchestration (unchanged interface)
# ----------------------------------------------------------------------
def process_candidates(manifest: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": utc_now(),
        "candidates": [],
        "summary": {"total": 0, "processed": 0, "errors": 0},
    }

    for cand in manifest.get("vendors", []):
        report["summary"]["total"] += 1
        try:
            text_fields = " ".join(
                filter(
                    None,
                    [
                        cand.get("candidate_key", ""),
                        cand.get("display_name", ""),
                        cand.get("family", ""),
                        cand.get("notes", ""),
                    ],
                )
            )
            vec = feature_vector(text_fields)
            raw_score, label = hygiene_score(vec)
            hybrid = hybrid_score(vec, cand.get("classification"))
            findings = enforce_fast_path_rule(cand)

            report["candidates"].append(
                {
                    "candidate_key": cand.get("candidate_key"),
                    "hygiene": {"raw": raw_score, "label": label},
                    "hybrid_score": hybrid,
                    "entropy_factor": _entropy_factor(vec),
                    "fast_path_findings": findings,
                    "classification": cand.get("classification"),
                }
            )
            report["summary"]["processed"] += 1
        except Exception as exc:  # pragma: no cover – defensive
            report["candidates"].append(
                {"candidate_key": cand.get("candidate_key"), "error": str(exc)}
            )
            report["summary"]["errors"] += 1

    return report


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Hybrid Ternary Lens Audit & Decision‑Hygiene")
    parser.add_argument("manifest", type=Path, help="Path to vendor manifest JSON")
    parser.add_argument("-o", "--output", type=Path, default=Path("hybrid_report.json"))
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    report = process_candidates(manifest)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()