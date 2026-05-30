# DARWIN HAMMER — match 25, survivor 1
# gen: 2
# parent_a: hybrid_geometric_product_voronoi_partition_m4_s2.py (gen1)
# parent_b: decision_hygiene.py (gen0)
# born: 2026-05-29T23:23:00Z

"""
Hybrid module combining geometric algebra (from hybrid_geometric_product_voronoi_partition_m4_s2.py) 
and decision hygiene scoring (from decision_hygiene.py). 

Mathematical bridge: 
- Geometric algebra's multivector representation is used to encode decision hygiene features 
  as points in a high-dimensional space, enabling Voronoi partitioning of decisions based 
  on their hygiene features.
- The decision hygiene feature extraction and scoring algorithms are used to compute 
  the coordinates of these points in the high-dimensional space.
"""

import math
import random
import sys
import pathlib
import numpy as np
from typing import Tuple, List, Dict

# Geometric algebra core
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble-sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j:j + 2]
                n -= 2
                sign *= 1
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> Tuple[frozenset[int], int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components: Dict[frozenset[int], float], n: int):
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if abs(v) > 1e-15}, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        neg = Multivector({k: -v for k, v in other.components.items()}, other.n)
        return self.__add__(neg)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Multivector({k: v * other for k, v in self.components.items()}, self.n)
        return geometric_product(self, other)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """Full Clifford product a*b."""
    result: Dict[frozenset[int], float] = {}
    for blade_a, coef_a in a.components.items():
        for blade_b, coef_b in b.components.items():
            blade_out, sign = _multiply_blades(blade_a, blade_b)
            contrib = sign * coef_a * coef_b
            result[blade_out] = result.get(blade_out, 0.0) + contrib
    n = max(a.n, b.n)
    return Multivector({k: v for k, v in result.items() if abs(v) > 1e-15}, n)


def inner_product(a: Multivector, b: Multivector) -> Multivector:
    """Symmetric inner product (ab + ba)/2."""
    ab = geometric_product(a, b)
    ba = geometric_product(b, a)
    return (ab + ba) * 0.5


# Decision hygiene core
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


# Hybrid functions
def text_to_multivector(text: str) -> Multivector:
    c = counts(text)
    blades = {
        frozenset([1]): c["evidence_count"],
        frozenset([2]): c["planning_count"],
        frozenset([3]): c["delay_count"],
        frozenset([4]): c["support_count"],
        frozenset([5]): c["boundary_count"],
        frozenset([6]): c["outcome_count"],
        frozenset([7]): c["impulsive_count"],
        frozenset([8]): c["scarcity_count"],
        frozenset([9]): c["risk_count"],
    }
    return Multivector(blades, 9)


def multivector_distance(a: Multivector, b: Multivector) -> float:
    diff = a - b
    return diff.scalar_part()


def cluster_texts(texts: list[str]) -> list[list[str]]:
    multivectors = [text_to_multivector(text) for text in texts]
    clusters = []
    for multivector in multivectors:
        assigned = False
        for cluster in clusters:
            centroid = sum([text_to_multivector(text) for text in cluster], Multivector({}, 9)) / len(cluster)
            if multivector_distance(multivector, centroid) < 10:
                cluster.append(texts[multivectors.index(multivector)])
                assigned = True
                break
        if not assigned:
            clusters.append([texts[multivectors.index(multivector)]])
    return clusters


if __name__ == "__main__":
    texts = [
        "I have a plan to finish this project on time.",
        "I need to verify the sources before I proceed.",
        "I'm feeling overwhelmed and just want to quit.",
        "I've been working hard and I'm proud of my progress.",
        "I need to take a break and come back to this later.",
        "I'm not sure if I can afford to take on this new project.",
    ]
    clusters = cluster_texts(texts)
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {cluster}")