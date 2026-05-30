# DARWIN HAMMER — match 25, survivor 2
# gen: 2
# parent_a: hybrid_geometric_product_voronoi_partition_m4_s2.py (gen1)
# parent_b: decision_hygiene.py (gen0)
# born: 2026-05-29T23:23:00Z

"""Hybrid decision‑hygiene & geometric‑algebra module.

Parents:
- **geometric_product.py / voronoi_partition.py** – provides a Clifford algebra
  implementation (Cl(n,0)), inner‑product based Euclidean distance and a Voronoi
  region assignment.
- **decision_hygiene.py** – extracts nine deterministic textual feature counts
  (evidence, planning, …, risk) and maps them to a bounded hygiene score.

Mathematical bridge:
Each decision text is mapped to a 9‑dimensional grade‑1 multivector  


v = c₁·e₁ + c₂·e₂ + … + c₉·e₉


where the coefficients *cᵢ* are the feature counts.  
The Euclidean squared distance between two decisions *a* and *b* is the scalar
part of the inner product ⟨a‑b , a‑b⟩, exactly the metric used by the Voronoi
partition.  Consequently we can assign a decision to the nearest “hygiene
prototype” (high, medium, low) and also rotate a decision vector toward a
desired prototype, then re‑score it with the original decision‑hygiene logic.

The module therefore fuses:
1. Feature extraction → multivector encoding.
2. Geometric‑product distance → Voronoi region selection.
3. Linear‑interpolation rotor → guided improvement of the decision signal.
"""

from __future__ import annotations

import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 1️⃣  Decision‑hygiene feature extraction (parent B)
# ---------------------------------------------------------------------------

EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b", re.I)
SUPPORT_RE = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I)
OUTCOME_RE = re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I)
IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b", re.I)
SCARCITY_RE = re.compile(r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b", re.I)
RISK_RE = re.compile(r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b", re.I)


def counts(text: str) -> Dict[str, int]:
    """Return the nine deterministic feature counts for *text*."""
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


def score_features(c: Dict[str, int]) -> Tuple[int, str]:
    """Map raw counts to a bounded hygiene score and a qualitative label."""
    positive = (
        c["evidence_count"] * 1600
        + c["planning_count"] * 1200
        + c["delay_count"] * 1400
        + c["support_count"] * 1000
        + c["boundary_count"] * 1200
        + c["outcome_count"] * 800
    )
    negative = (
        c["impulsive_count"] * 1500
        + c["scarcity_count"] * 700
        + c["risk_count"] * 1200
    )
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


# ---------------------------------------------------------------------------
# 2️⃣  Clifford algebra core (parent A, trimmed)
# ---------------------------------------------------------------------------

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Sort *indices* by bubble‑sort, counting sign flips; cancel equal pairs."""
    lst = list(indices)
    sign = 1
    i = 0
    while i < len(lst):
        j = 0
        while j < len(lst) - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # e_i * e_i = 1 → cancel
                del lst[j : j + 2]
                sign *= 1
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> Tuple[frozenset[int], int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Grade‑mixed element of Cl(n,0)."""

    def __init__(self, components: Dict[frozenset[int], float], n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    # -------------------------------------------------------------------
    # Arithmetic
    # -------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        neg = Multivector({k: -v for k, v in other.components.items()}, other.n)
        return self + neg

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Multivector({k: v * other for k, v in self.components.items()}, self.n)
        return geometric_product(self, other)

    __rmul__ = __mul__

    def __neg__(self) -> "Multivector":
        return Multivector({k: -v for k, v in self.components.items()}, self.n)

    # -------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------
    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            label = "1" if not blade else "e" + "".join(str(i) for i in sorted(blade))
            terms.append(f"{coef:+.3g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    result: Dict[frozenset[int], float] = {}
    for blade_a, coef_a in a.components.items():
        for blade_b, coef_b in b.components.items():
            out_blade, sign = _multiply_blades(blade_a, blade_b)
            result[out_blade] = result.get(out_blade, 0.0) + sign * coef_a * coef_b
    n = max(a.n, b.n)
    return Multivector(result, n)


def inner_product(a: Multivector, b: Multivector) -> Multivector:
    """Symmetric inner product (ab + ba)/2."""
    return (geometric_product(a, b) + geometric_product(b, a)) * 0.5


def mv_distance(a: Multivector, b: Multivector) -> float:
    """Euclidean distance via scalar part of ⟨a‑b , a‑b⟩."""
    diff = a - b
    scalar = inner_product(diff, diff).scalar_part()
    return math.sqrt(abs(scalar))


# ---------------------------------------------------------------------------
# 3️⃣  Hybrid operations
# ---------------------------------------------------------------------------

# Map feature keys to basis indices (1‑based for readability)
_FEATURE_ORDER = [
    "evidence_count",
    "planning_count",
    "delay_count",
    "support_count",
    "boundary_count",
    "outcome_count",
    "impulsive_count",
    "scarcity_count",
    "risk_count",
]


def text_to_mv(text: str) -> Multivector:
    """Encode *text* as a grade‑1 multivector in ℝ⁹."""
    c = counts(text)
    comps = {
        frozenset({i + 1}): float(c[key]) for i, key in enumerate(_FEATURE_ORDER)
    }
    return Multivector(comps, n=9)


def prototype_mv(label: str) -> Multivector:
    """
    Return a handcrafted prototype vector for a hygiene *label*.
    The numbers are chosen to be representative of the semantic meaning.
    """
    if label == "high":
        # Many positive signals, few negatives
        vals = [3, 2, 2, 2, 2, 3, 0, 0, 0]
    elif label == "medium":
        vals = [1, 1, 1, 1, 1, 1, 1, 1, 0]
    elif label == "low":
        # Many negatives, few positives
        vals = [0, 0, 0, 0, 0, 0, 3, 2, 2]
    else:
        raise ValueError(f"unknown prototype label {label!r}")
    comps = {frozenset({i + 1}): float(v) for i, v in enumerate(vals)}
    return Multivector(comps, n=9)


def voronoi_partition_decisions(
    texts: List[str],
    prototypes: Dict[str, Multivector],
) -> Dict[str, List[str]]:
    """
    Assign each decision *text* to the nearest prototype using Clifford‑product distance.
    Returns a mapping ``label -> list of texts``.
    """
    assignment: Dict[str, List[str]] = {label: [] for label in prototypes}
    for txt in texts:
        mv = text_to_mv(txt)
        best_label = None
        best_dist = float("inf")
        for label, proto in prototypes.items():
            d = mv_distance(mv, proto)
            if d < best_dist:
                best_dist = d
                best_label = label
        assignment[best_label].append(txt)  # type: ignore[arg-type]
    return assignment


def rotate_toward(text: str, target_label: str, alpha: float = 0.5) -> Tuple[str, int, str]:
    """
    Perform a simple “rotor” by linear interpolation towards the *target_label* prototype.
    Returns the transformed text (re‑generated from the interpolated multivector),
    the new hygiene score and its label.
    The interpolation factor *alpha* ∈ [0,1] controls how aggressively we move.
    """
    src_mv = text_to_mv(text)
    tgt_mv = prototype_mv(target_label)

    # Linear interpolation in the vector space (acts as a rotor for grade‑1)
    new_mv = src_mv + (tgt_mv - src_mv) * alpha

    # Project back to integer counts (rounding, never negative)
    projected_counts = {
        key: max(0, int(round(new_mv.components.get(frozenset({i + 1}), 0.0))))
        for i, key in enumerate(_FEATURE_ORDER)
    }

    # Re‑compose a dummy text that reflects the new counts (for demonstration)
    dummy_parts = []
    for key, cnt in projected_counts.items():
        dummy_parts.append(f"{key}*{cnt}")
    transformed_text = " ".join(dummy_parts)

    # Re‑score using the original decision‑hygiene logic
    new_score, new_label = score_features(projected_counts)
    return transformed_text, new_score, new_label


def batch_score(texts: List[str]) -> List[Tuple[int, str]]:
    """Convenient wrapper returning (score, label) for each *text*."""
    return [score_features(counts(t)) for t in texts]


# ---------------------------------------------------------------------------
# 4️⃣  Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_texts = [
        "I have verified the source and documented the proof. The plan is ready and we can ship tomorrow.",
        "I'm scared, I might die. Can't afford any more delays. Need to act right now!",
        "We have a checklist, the budget is approved, and the team is ready. No risk detected.",
        "I don't know what to do, maybe wait until later. No clear evidence yet.",
    ]

    # 1️⃣ Partition decisions into hygiene zones
    prototypes = {
        "high": prototype_mv("high"),
        "medium": prototype_mv("medium"),
        "low": prototype_mv("low"),
    }
    partitions = voronoi_partition_decisions(sample_texts, prototypes)
    print("Voronoi partition:")
    for label, group in partitions.items():
        print(f"  {label}: {len(group)} decision(s)")

    # 2️⃣ Show original scores
    print("\nOriginal scores:")
    for txt, (sc, lbl) in zip(sample_texts, batch_score(sample_texts)):
        print(f"  Score={sc:5d} → {lbl}")

    # 3️⃣ Attempt to improve a low‑hygiene decision
    low_text = partitions["low"][0] if partitions["low"] else sample_texts[1]
    transformed, new_sc, new_lbl = rotate_toward(low_text, target_label="high", alpha=0.7)
    print("\nImprovement attempt:")
    print(f"  Original text: {low_text}")
    print(f"  Transformed MV‑derived text: {transformed}")
    print(f"  New score={new_sc} → {new_lbl}")

    sys.exit(0)