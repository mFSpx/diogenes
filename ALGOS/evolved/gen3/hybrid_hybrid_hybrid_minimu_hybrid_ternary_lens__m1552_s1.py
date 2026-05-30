# DARWIN HAMMER — match 1552, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s5.py (gen2)
# parent_b: hybrid_ternary_lens_router_hybrid_decision_hygi_m44_s0.py (gen2)
# born: 2026-05-29T23:37:22Z

"""Hybrid Epistemic‑Ternary Entropy Analyzer

Parents:
- hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s5.py
- hybrid_ternary_lens_router_hybrid_decision_hygi_m44_s0.py

Mathematical bridge:
The epistemic certainty flag supplies a confidence weight `w = confidence_bps/10_000`.
The ternary lens router converts a textual observation into a 12‑dimensional ternary
vector `v ∈ {‑1,0,1}¹².  Shannon entropy `H(v)` is computed from the empirical
distribution of the three symbol frequencies in `v`.  The hybrid algorithm
produces a *weighted entropy* `E = w·H(v)`, thus fusing the Bayesian‑style
certainty measure with the information‑theoretic analysis of the ternary
representation.  Matrix‑level operations (vectorisation with NumPy, probability
vectors, and outer products) are used throughout the implementation.
"""

import math
import random
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np

# ----------------------------------------------------------------------
# Epistemic certainty (Parent A)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")
        if not self.generated_at:
            object.__setattr__(
                self,
                "generated_at",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )

    def as_dict(self) -> Dict[str, Union[str, int, Tuple[str, ...]]]:
        return {
            "label": self.label,
            "confidence_bps": self.confidence_bps,
            "authority_class": self.authority_class,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "generated_at": self.generated_at,
        }


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    """Factory for :class:`CertaintyFlag`."""
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


# ----------------------------------------------------------------------
# Ternary lens router & Shannon entropy (Parent B)
# ----------------------------------------------------------------------
TERNARY_DIMS = 12

# Regular‑expression categories – each maps to one ternary dimension.
_CATEGORY_REGEX: List[Tuple[str, re.Pattern]] = [
    ("evidence", re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)),
    ("planning", re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)),
    ("delay", re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|first|after|review)\b", re.I)),
    ("support", re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)),
    ("boundary", re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I)),
    ("outcome", re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I)),
    ("impulsive", re.compile(r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b", re.I)),
    ("scarcity", re.compile(r"\b(?:shortage|limited|rare|few|only|exclusive|deadline|last chance|urgent|pressure)\b", re.I)),
    ("security", re.compile(r"\b(?:encrypt|secure|authentication|auth|token|key|password|vuln|exploit|attack|breach|firewall)\b", re.I)),
    ("performance", re.compile(r"\b(?:speed|latency|throughput|optimize|fast|slow|lag|benchmark)\b", re.I)),
    ("cost", re.compile(r"\b(?:price|cost|budget|expensive|cheap|free|charge|fee)\b", re.I)),
    ("legal", re.compile(r"\b(?:license|compliance|law|regulation|gdpr|ccpa|policy|terms)\b", re.I)),
]


def _ternary_score_for_match(match: re.Match) -> int:
    """Return a ternary symbol for a regex match.
    Positive presence → +1, explicit negation (e.g. 'no evidence') → -1,
    otherwise 0.  For simplicity we only detect the positive case here.
    """
    # Very naive negation detection – look for a preceding 'no' or "n't".
    start = match.start()
    preceding = match.string[max(0, start - 4) : start].lower()
    if "no " in preceding or "n't" in preceding:
        return -1
    return 1


def generate_ternary_vector(text: str) -> np.ndarray:
    """Convert *text* into a 12‑dimensional ternary vector.

    Each dimension corresponds to one of the categories defined in
    ``_CATEGORY_REGEX``.  The entry is -1, 0, or +1 according to the presence of
    a matching token (negative if a simple negation pattern is detected).
    """
    vector = np.zeros(TERNARY_DIMS, dtype=int)
    for idx, (_, pattern) in enumerate(_CATEGORY_REGEX):
        match = pattern.search(text)
        if match:
            vector[idx] = _ternary_score_for_match(match)
    return vector


def shannon_entropy_from_counts(counts: np.ndarray) -> float:
    """Compute Shannon entropy (base‑2) from raw symbol counts."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    # Avoid log2(0) by masking zero probabilities.
    mask = probs > 0
    return -float(np.sum(probs[mask] * np.log2(probs[mask])))


def ternary_entropy(vector: np.ndarray) -> float:
    """Shannon entropy of the empirical distribution of -1, 0, +1 in *vector*."""
    # Map symbols to indices 0,1,2 → -1,0,+1
    shifted = vector + 1  # now in {0,1,2}
    counts = np.bincount(shifted, minlength=3)
    return shannon_entropy_from_counts(counts)


def certainty_weight(flag: CertaintyFlag) -> float:
    """Convert ``confidence_bps`` into a probability weight in [0,1]."""
    return flag.confidence_bps / 10_000.0


def weighted_entropy(vector: np.ndarray, weight: float) -> float:
    """Return w·H(v) where H is the ternary Shannon entropy."""
    return weight * ternary_entropy(vector)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_analyze(text: str, flag: CertaintyFlag) -> Dict[str, Any]:
    """Perform the full hybrid analysis for a single observation.

    Returns a dictionary containing:
    - the raw ternary vector,
    - the plain entropy,
    - the epistemic weight,
    - the weighted entropy,
    - the original certainty flag as a plain dict.
    """
    vec = generate_ternary_vector(text)
    plain_H = ternary_entropy(vec)
    w = certainty_weight(flag)
    weighted_H = weighted_entropy(vec, w)
    return {
        "ternary_vector": vec.tolist(),
        "plain_entropy": plain_H,
        "certainty_weight": w,
        "weighted_entropy": weighted_H,
        "certainty_flag": flag.as_dict(),
    }


def batch_hybrid_matrix(texts: List[str], flags: List[CertaintyFlag]) -> Tuple[np.ndarray, np.ndarray]:
    """Create a matrix of ternary vectors and a vector of weighted entropies.

    *texts* and *flags* must be of the same length.
    Returns ``(V, E)`` where ``V`` is shape (n,12) and ``E`` is shape (n,).
    """
    if len(texts) != len(flags):
        raise ValueError("texts and flags length mismatch")
    n = len(texts)
    V = np.zeros((n, TERNARY_DIMS), dtype=int)
    E = np.zeros(n, dtype=float)
    for i, (txt, flg) in enumerate(zip(texts, flags)):
        vec = generate_ternary_vector(txt)
        V[i] = vec
        E[i] = weighted_entropy(vec, certainty_weight(flg))
    return V, E


def aggregate_weighted_entropy(matrix: np.ndarray, weights: np.ndarray) -> float:
    """Aggregate weighted entropies across a batch.

    Computes the mean of the individual weighted entropies, optionally weighted
    by an external *weights* vector (e.g. confidence).  If *weights* is None,
    a simple arithmetic mean is returned.
    """
    if weights is None:
        return float(np.mean(matrix))
    # Normalise external weights to sum to 1.
    norm = weights / weights.sum()
    return float(np.sum(matrix * norm))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "We have verified the sha256 hash, the plan is ready, but we must wait until "
        "tomorrow because the budget is limited and the security audit is pending."
    )
    sample_flag = certainty(
        "PROBABLE",
        confidence_bps=7_500,
        authority_class="internal_audit",
        rationale="Partial evidence from logs and pending security review",
        evidence_refs=["sha256:deadbeef", "log:/var/log/app.log"],
    )
    result = hybrid_analyze(sample_text, sample_flag)
    print("Hybrid analysis result:")
    for k, v in result.items():
        print(f"{k}: {v}")

    # Batch demonstration
    texts = [
        "Evidence collected, plan approved, cost is cheap.",
        "No evidence, urgent deadline, security breach detected.",
        "Documentation ready, performance is good, legal compliance met.",
    ]
    flags = [
        certainty("FACT", confidence_bps=10_000, authority_class="ops", rationale="Full evidence", evidence_refs=[]),
        certainty("PROBABLE", confidence_bps=6_000, authority_class="security", rationale="Partial breach info", evidence_refs=[]),
        certainty("POSSIBLE", confidence_bps=4_000, authority_class="legal", rationale="Docs pending", evidence_refs=[]),
    ]
    V, E = batch_hybrid_matrix(texts, flags)
    print("\nBatch ternary matrix:\n", V)
    print("\nBatch weighted entropies:\n", E)
    overall = aggregate_weighted_entropy(E, np.array([certainty_weight(f) for f in flags]))
    print("\nOverall aggregated weighted entropy:", overall)