# DARWIN HAMMER — match 3805, survivor 3
# gen: 5
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s4.py (gen1)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s2.py (gen4)
# born: 2026-05-29T23:51:44Z

"""hybrid_decision_hygiene_shannon_entropy_geometric_ternary.py
Hybrid algorithm merging:
- Parent A: regex‑based feature extraction with positive/negative weight scoring.
- Parent B: Clifford geometric product and ternary‑route style nearest‑node assignment.

Mathematical bridge:
Each extracted textual feature is mapped to a basis vector of a Euclidean Clifford algebra
Cl(n,0) (n = number of features).  The feature count vector becomes a multivector
by assigning the count as the coefficient of the corresponding 1‑blade.  A
“target” multivector encodes the positive/negative weight scheme of Parent A.
The geometric product between the text multivector and the target multivector
produces a new multivector whose scalar (grade‑0) part is mathematically
equivalent to the weighted dot‑product used in Parent A, while higher‑grade
components capture interactions between features.  Those higher‑grade parts are
then used as coordinates in a ternary‑route graph: each route node is also a
multivector, and a point (the text multivector) is assigned to the nearest node
using the norm induced by the geometric product.  This yields a unified decision
score together with a topological routing decision.
"""

import re
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regexes and raw count extraction (kept unchanged)
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
# Parent B – Clifford geometric product helpers (trimmed & completed)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return sorted indices and the sign resulting from anti‑commutative swaps.

    Duplicate indices cancel because e_i*e_i = 1.
    """
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            if lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i]
                sign *= -1
            elif lst[i] == lst[j]:
                # cancel the pair
                del lst[j]
                del lst[i]
                n -= 2
                i -= 1  # step back to re‑evaluate at new i
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    """Geometric product of two basis blades."""
    combined = list(blade_a) + list(blade_b)
    sorted_idxs, sign = _blade_sign(combined)
    return frozenset(sorted_idxs), sign


class Multivector:
    """Simple multivector for Cl(n,0) with dictionary representation.

    The key is a frozenset of basis indices (grade = len(key)).
    The value is the scalar coefficient.
    """

    def __init__(self, components: Dict[frozenset, float] | None = None):
        self.components: Dict[frozenset, float] = {}
        if components:
            # prune zeroes
            for k, v in components.items():
                if abs(v) > 1e-12:
                    self.components[frozenset(k)] = float(v)

    @staticmethod
    def basis_vector(idx: int, coeff: float = 1.0) -> "Multivector":
        return Multivector({frozenset([idx]): coeff})

    def __add__(self, other: "Multivector") -> "Multivector":
        result = Multivector(self.components.copy())
        for blade, coeff in other.components.items():
            result.components[blade] = result.components.get(blade, 0.0) + coeff
            if abs(result.components[blade]) < 1e-12:
                del result.components[blade]
        return result

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self.components.items()})

    def geometric_product(self, other: "Multivector") -> "Multivector":
        """Geometric product (Clifford multiplication)."""
        result: Dict[frozenset, float] = {}
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                res_blade, sign = _multiply_blades(blade_a, blade_b)
                res_coeff = coeff_a * coeff_b * sign
                result[res_blade] = result.get(res_blade, 0.0) + res_coeff
        # prune tiny entries
        result = {b: c for b, c in result.items() if abs(c) > 1e-12}
        return Multivector(result)

    __mul__ = geometric_product  # alias for * operator

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) component, 0 if absent."""
        return self.components.get(frozenset(), 0.0)

    def norm(self) -> float:
        """Euclidean norm induced by the scalar product."""
        return math.sqrt(sum(c * c for c in self.components.values()))

    def distance(self, other: "Multivector") -> float:
        """Metric induced by the norm of the difference."""
        return (self - other).norm()

    def __repr__(self) -> str:
        if not self.components:
            return "0"
        terms = []
        for blade in sorted(self.components, key=lambda b: (len(b), sorted(b))):
            coeff = self.components[blade]
            if blade:
                basis = "e" + "^".join(str(i) for i in sorted(blade))
                terms.append(f"{coeff:.3g}*{basis}")
            else:
                terms.append(f"{coeff:.3g}")
        return " + ".join(terms)


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def extract_feature_counts(text: str) -> Dict[str, int]:
    """Return a dictionary mapping each feature name to its regex hit count."""
    regex_map = {
        "evidence": EVIDENCE_RE,
        "planning": PLANNING_RE,
        "delay": DELAY_RE,
        "support": SUPPORT_RE,
        "boundary": BOUNDARY_RE,
        "outcome": OUTCOME_RE,
        "impulsive": IMPULSIVE_RE,
        "scarcity": SCARCITY_RE,
        "risk": RISK_RE,
    }
    counts = {}
    for name, pattern in regex_map.items():
        matches = pattern.findall(text)
        counts[name] = len(matches)
    return counts


def multivector_from_counts(counts: Dict[str, int]) -> Multivector:
    """Map feature counts onto a multivector in Cl(n,0).

    Feature i → basis vector e_i (i starts at 0).  The coefficient is the raw count.
    """
    components: Dict[frozenset, float] = {}
    for idx, feature in enumerate(_FEATURE_ORDER):
        cnt = counts.get(feature, 0)
        if cnt != 0:
            components[frozenset([idx])] = float(cnt)
    return Multivector(components)


def target_multivector() -> Multivector:
    """Construct the target multivector that encodes the A‑algorithm weights.

    Positive weights are placed on the corresponding 1‑blades, negative weights
    on the same blades (subtracted).  The scalar part is zero.
    """
    comps: Dict[frozenset, float] = {}
    for idx in range(len(_FEATURE_ORDER)):
        pos = _POSITIVE_WEIGHTS[idx]
        neg = _NEGATIVE_WEIGHTS[idx]
        net = float(pos - neg)  # can be negative
        if net != 0.0:
            comps[frozenset([idx])] = net
    return Multivector(comps)


def hybrid_score(text: str) -> float:
    """Compute a decision score that merges Parent A's weighted sum with
    Parent B's geometric product.

    The scalar part of (text_mv * target_mv) equals the classic weighted dot product.
    """
    counts = extract_feature_counts(text)
    text_mv = multivector_from_counts(counts)
    tgt_mv = target_multivector()
    product = text_mv * tgt_mv
    return product.scalar_part()


def generate_random_route_nodes(num_nodes: int, dim: int) -> List[Multivector]:
    """Create a list of random multivectors (one‑blade each) to serve as route nodes."""
    nodes = []
    for _ in range(num_nodes):
        idx = random.randint(0, dim - 1)
        coeff = random.uniform(-5.0, 5.0)
        nodes.append(Multivector.basis_vector(idx, coeff))
    return nodes


def assign_to_nearest_node(text: str, nodes: List[Multivector]) -> Tuple[int, float]:
    """Assign the text multivector to the nearest route node using the geometric norm.

    Returns (node_index, distance).
    """
    counts = extract_feature_counts(text)
    text_mv = multivector_from_counts(counts)
    distances = [text_mv.distance(node) for node in nodes]
    nearest_idx = int(np.argmin(distances))
    return nearest_idx, distances[nearest_idx]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample = """
    I have evidence that the plan was delayed because we ran out of money.
    The team needs support and a clear checklist before we can finish.
    """
    print("Feature counts:", extract_feature_counts(sample))
    print("Hybrid score:", hybrid_score(sample))

    # Build a tiny ternary‑style route graph (3 nodes, 3‑dimensional space)
    route_nodes = generate_random_route_nodes(num_nodes=3, dim=len(_FEATURE_ORDER))
    print("Route nodes:")
    for i, node in enumerate(route_nodes):
        print(f"  Node {i}: {node}")

    idx, dist = assign_to_nearest_node(sample, route_nodes)
    print(f"Text assigned to node {idx} (distance = {dist:.4f})")