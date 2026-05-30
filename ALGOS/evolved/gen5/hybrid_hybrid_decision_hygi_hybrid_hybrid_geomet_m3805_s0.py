# DARWIN HAMMER — match 3805, survivor 0
# gen: 5
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s4.py (gen1)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s2.py (gen4)
# born: 2026-05-29T23:51:44Z

"""
This module fuses the hybrid decision hygiene algorithm from 
hybrid_decision_hygiene_shannon_entropy_m12_s4.py with the hybrid geometric product 
from hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s2.py. The mathematical bridge 
between these two structures is formed by representing the evidence 
counts from the decision hygiene algorithm as a multivector in the 
Clifford algebra (Cl(n,0)), and then using the geometric product to 
compute the distances and orientations between these evidence counts and 
the points in the ternary route graph.

The governing equations of both algorithms are used to compute the 
geometric product of multivectors, which are then used to represent 
points and vectors in the ternary route graph. The hybrid decision 
hygiene algorithm is used to assign weights to the evidence counts, 
and the geometric product is used to compute the distances and 
orientations between these points and nodes.

This module provides functions to compute the geometric product of 
multivectors, assign points to their nearest route nodes using the 
hybrid decision hygiene algorithm, and visualize the resulting assignments.
"""

import math
import numpy as np
import random
import sys
import pathlib
from typing import Any, Dict, List, Tuple
from collections import Counter
from datetime import datetime

# ----------------------------------------------------------------------
# Parent A – regexes and raw count extraction
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
# Core blade arithmetic helpers
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list.

    Each transposition of adjacent indices that are out of order flips the
    sign (anti-commutativity).  Duplicate indices cancel (e_i^2 = 1 → they
    annihilate and contribute +1 to the sign, but the index disappears).
    """
    lst = list(indices)
    sign = 1
    # Bubble sort; track swaps
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # Duplicate: e_i * e_i = 1, remove both
                lst.pop(j)
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices).

    Returns (result_blade_frozenset, sign).
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


# Multivector
class Multivector:
    def __init__(self, basis_blades):
        self.basis_blades = basis_blades

    def multiply(self, other):
        result = np.zeros(len(self.basis_blades))
        for i, blade in enumerate(self.basis_blades):
            for j, blade2 in enumerate(other.basis_blades):
                mult_blade, sign = _multiply_blades(blade, blade2)
                result[i] += sign
        return Multivector(result)


# ----------------------------------------------------------------------
# Hybrid decision hygiene and geometric product
# ----------------------------------------------------------------------
def hybrid_decision_geometric_product(text):
    # Extract evidence counts from text
    evidence_counts = {}
    for word in text.split():
        for pattern in [EVIDENCE_RE, PLANNING_RE, DELAY_RE, SUPPORT_RE, BOUNDARY_RE, OUTCOME_RE, IMPULSIVE_RE, SCARCITY_RE, RISK_RE]:
            match = pattern.search(word)
            if match:
                feature = match.group().lower()
                evidence_counts[feature] = evidence_counts.get(feature, 0) + 1

    # Assign weights to evidence counts
    weights = np.zeros(len(_FEATURE_ORDER))
    for i, feature in enumerate(_FEATURE_ORDER):
        weights[i] = _POSITIVE_WEIGHTS[i] * evidence_counts.get(feature, 0) - _NEGATIVE_WEIGHTS[i] * (evidence_counts.get(feature, 0) - 1)

    # Create multivector from evidence counts
    basis_blades = [frozenset([i]) for i in range(len(_FEATURE_ORDER))]
    multivector = Multivector(basis_blades)

    # Compute geometric product
    for i, weight in enumerate(weights):
        multivector.multiply(Multivector([frozenset([i])]))

    return multivector


def hybrid_decision_geometric_product_distance(text, reference_multivector):
    # Compute geometric product
    multivector = hybrid_decision_geometric_product(text)

    # Compute distance between multivectors
    result = np.sum(np.abs(np.array(multivector.basis_blades) - np.array(reference_multivector.basis_blades)))

    return result


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    text = "I need evidence to confirm the outcome."
    reference_multivector = hybrid_decision_geometric_product(text)
    distance = hybrid_decision_geometric_product_distance(text, reference_multivector)
    print(distance)