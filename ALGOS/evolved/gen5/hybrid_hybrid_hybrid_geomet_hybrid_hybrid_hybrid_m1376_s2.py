# DARWIN HAMMER — match 1376, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_decision_hygiene_m25_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_endpoint_circ_m1035_s0.py (gen4)
# born: 2026-05-29T23:37:15Z

"""Hybrid Algorithm integrating Decision Hygiene Multivectors (Parent A) with
Posterior‑Edge‑Belief Recovery Priority (Parent B).

Mathematical bridge
-------------------
* Parent A encodes a decision‑hygiene score as a **multivector** 𝑀 in a
  Clifford algebra 𝔾(ℝⁿ).  The scalar part 𝑀₀ and the grade‑1 part
  (coefficients of the basis vectors 𝑒ᵢ) form a real vector **w** ∈ ℝⁿ.

* Parent B builds an expected feature‑count vector **f** (evidence / planning
  token counts) and a posterior edge‑belief distribution **b** that weights a
  recovery‑priority score.

The hybrid replaces the deterministic weight vector of Parent B by the
vector **w** extracted from the multivector.  All subsequent linear
operations (expected‑feature weighting, inner‑product recovery priority and
distance weighting in a tree cost) are performed with **w**, while the
geometric product of 𝑀 with a “audit” multivector 𝐴 injects higher‑grade
information into the final scalar score.

Thus the core topology of both parents is fused through the linear map

    w_i = coeff(𝑀, e_i)          (grade‑1 coefficients of 𝑀)

and the scalar augmentation

    score = w·f + ⟨𝑀·A⟩₀

where ⟨·⟩₀ extracts the scalar part of the geometric product.
"""

import math
import random
import sys
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Multivector implementation (geometric algebra)
# ----------------------------------------------------------------------
class Multivector:
    """Simple Euclidean Clifford algebra multivector."""

    def __init__(self, components: Dict[frozenset, float], n: int):
        # components: blade (frozenset of basis indices) -> coefficient
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    # ------------------------------------------------------------------
    # Basic grade / scalar helpers
    # ------------------------------------------------------------------
    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if abs(v) > 1e-15}, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        neg = Multivector({k: -v for k, v in other.components.items()}, other.n)
        return self.__add__(neg)

    def __neg__(self) -> "Multivector":
        return Multivector({k: -v for k, v in self.components.items()}, self.n)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Multivector({k: v * other for k, v in self.components.items()}, self.n)
        if isinstance(other, Multivector):
            return geometric_product(self, other)
        raise TypeError("Unsupported multiplication")

    __rmul__ = __mul__

# ----------------------------------------------------------------------
# Geometric product (Euclidean metric, anti‑commuting basis)
# ----------------------------------------------------------------------
def blade_mul(b1: frozenset, b2: frozenset) -> Tuple[frozenset, int]:
    """Multiply two blades returning (result_blade, sign)."""
    # Convert to ordered list for sign counting
    result = list(b1) + list(b2)
    sign = 1
    # Perform pairwise swaps to bring to canonical order and cancel duplicates
    i = 0
    while i < len(result):
        j = i + 1
        while j < len(result):
            if result[i] == result[j]:
                # e_i * e_i = 1  -> cancel both
                del result[j]
                del result[i]
                i -= 1
                break
            elif result[i] > result[j]:
                # swap to enforce ordering, flip sign
                result[i], result[j] = result[j], result[i]
                sign = -sign
                j += 1
            else:
                j += 1
        i += 1
    return frozenset(result), sign

def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """Full geometric product a*b."""
    result: Dict[frozenset, float] = {}
    for blade_a, coeff_a in a.components.items():
        for blade_b, coeff_b in b.components.items():
            blade_res, sign = blade_mul(blade_a, blade_b)
            result[blade_res] = result.get(blade_res, 0.0) + sign * coeff_a * coeff_b
    return Multivector(result, a.n)

# ----------------------------------------------------------------------
# Helper: extract grade‑1 vector (the "weight" vector) from a multivector
# ----------------------------------------------------------------------
def multivector_to_weight_vector(mv: Multivector) -> np.ndarray:
    """Return a dense numpy array of the grade‑1 coefficients (size = mv.n)."""
    w = np.zeros(mv.n, dtype=float)
    for blade, coef in mv.components.items():
        if len(blade) == 1:
            idx = next(iter(blade))  # basis index (0‑based)
            w[idx] = coef
    return w

# ----------------------------------------------------------------------
# Parent B – regex based evidence/planning counting and posterior edge beliefs
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)

def count_tokens(text: str) -> Tuple[int, int]:
    """Return (evidence_hits, planning_hits) in a piece of text."""
    ev = len(EVIDENCE_RE.findall(text))
    pl = len(PLANNING_RE.findall(text))
    return ev, pl

def posterior_edge_beliefs(edges: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
    """Assign a random belief (probability) to each edge, normalised to sum to 1."""
    raw = np.random.rand(len(edges))
    probs = raw / raw.sum()
    return {e: float(p) for e, p in zip(edges, probs)}

# ----------------------------------------------------------------------
# Hybrid functions (the required three+ demonstrations)
# ----------------------------------------------------------------------
def hybrid_lsm_vector(texts: List[str], mv: Multivector) -> np.ndarray:
    """
    Expected feature‑count vector for a collection of texts, weighted by the
    grade‑1 part of the multivector (the decision‑hygiene weight vector).

    Returns a 2‑element array: [weighted_evidence, weighted_planning].
    """
    # raw counts
    ev_total = 0
    pl_total = 0
    for t in texts:
        ev, pl = count_tokens(t)
        ev_total += ev
        pl_total += pl

    raw_vec = np.array([ev_total, pl_total], dtype=float)

    # weight vector w (size must be at least 2)
    w = multivector_to_weight_vector(mv)
    if w.size < 2:
        raise ValueError("Multivector must have at least two basis vectors for weighting.")
    weight_slice = w[:2]

    weighted = raw_vec * weight_slice
    return weighted

def hybrid_recovery_priority(feature_vec: np.ndarray, mv: Multivector) -> float:
    """
    Compute a recovery priority score that blends:
      * linear inner product between the feature vector and the multivector's weight vector,
      * scalar part of the geometric product between the multivector and an
        audit‑report multivector (grade‑0/1 components encode a ternary lens audit).
    """
    # Linear part
    w = multivector_to_weight_vector(mv)[: feature_vec.size]
    linear = float(np.dot(w, feature_vec))

    # Build a simple audit multivector A:
    #   scalar = 0.5  (baseline audit weight)
    #   grade‑1 e0 = 0.2 , e1 = -0.1  (ternary lens adjustments)
    audit_components = {
        frozenset(): 0.5,
        frozenset({0}): 0.2,
        frozenset({1}): -0.1,
    }
    A = Multivector(audit_components, mv.n)

    # Geometric product and scalar extraction
    scalar_part = (mv * A).scalar_part()

    return linear + scalar_part

def hybrid_tree_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    mv: Multivector,
) -> float:
    """
    Compute a tree‑cost where each edge distance is weighted by the
    corresponding grade‑1 coefficient from the multivector.
    The weight for edge i is w_i where w = multivector_to_weight_vector(mv).
    If there are more edges than basis vectors, excess edges receive weight 1.
    """
    w = multivector_to_weight_vector(mv)
    total = 0.0
    for idx, (u, v) in enumerate(edges):
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        dist = math.hypot(x2 - x1, y2 - y1)
        weight = w[idx] if idx < w.size else 1.0
        total += weight * dist
    return total

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Construct a decision‑hygiene multivector with 3 basis vectors
    # e0 = evidence weight, e1 = planning weight, e2 = auxiliary (unused here)
    components = {
        frozenset(): 0.0,               # scalar part
        frozenset({0}): 1.2,            # evidence coefficient
        frozenset({1}): 0.8,            # planning coefficient
        frozenset({2}): 0.3,            # extra dimension
    }
    mv = Multivector(components, n=3)

    # Sample texts
    sample_texts = [
        "The evidence was verified and the plan was approved.",
        "We need to document the source and create a checklist.",
        "Audit log shows proof of concept and a timeline for rollout.",
    ]

    # Hybrid LSM vector
    lsm = hybrid_lsm_vector(sample_texts, mv)
    print("Hybrid LSM vector (weighted counts):", lsm)

    # Recovery priority
    priority = hybrid_recovery_priority(lsm, mv)
    print("Hybrid recovery priority score:", priority)

    # Simple graph for tree cost
    nodes = {
        "A": (0.0, 0.0),
        "B": (3.0, 4.0),
        "C": (6.0, 0.0),
    }
    edges = [("A", "B"), ("B", "C")]
    cost = hybrid_tree_cost(nodes, edges, mv)
    print("Hybrid tree cost:", cost)