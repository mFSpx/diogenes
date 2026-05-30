# DARWIN HAMMER — match 1281, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s6.py (gen3)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s3.py (gen1)
# born: 2026-05-29T23:34:58Z

"""Hybrid algorithm combining geometric algebra (Parent A) with feature extraction (Parent B).

Bridge:
- Each feature extracted by *Parent B* is mapped to a distinct basis vector (grade‑1 blade) of a
  Clifford algebra `Cl(n,0)` defined in *Parent A*.
- A feature dictionary becomes a multivector `M = Σ_i f_i e_i`.
- The geometric product `M₁ * M₂` simultaneously yields:
    * the scalar (grade‑0) part → dot‑product similarity of the two feature sets,
    * higher‑grade parts (bivectors, trivectors, …) → interaction terms useful for
      curvature‑like metrics.
- This single algebraic operation fuses the two parent topologies: the statistical
  feature space is embedded in a geometric‑algebraic space, allowing both linear
  (inner‑product) and multilinear (outer‑product) analyses in one unified framework.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Helper functions (Parent A)
# ---------------------------------------------------------------------------

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting an index list.
    Repeated indices cancel (Grassmann algebra rule)."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign *= -1
            i = max(i - 1, 0)
        elif lst[i] == lst[i + 1]:
            # cancel duplicate index
            lst.pop(i)
            lst.pop(i)  # second element now occupies position i
            n -= 2
            i = max(i - 1, 0)
        else:
            i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    """Geometric product of two basis blades (as frozensets of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


# ---------------------------------------------------------------------------
# Multivector class (Parent A, extended with geometric product)
# ---------------------------------------------------------------------------

class Multivector:
    """Element of Cl(n,0) represented as a sparse sum of basis blades."""

    def __init__(self, components: Dict[frozenset, float], n: int):
        self.n = int(n)
        # prune zero coefficients
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    # -----------------------------------------------------------------------
    # Algebraic operations
    # -----------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self.components.items()}, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product (distributive over addition)."""
        result: Dict[frozenset, float] = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                prod_blade, sign = _multiply_blades(blade_a, blade_b)
                prod_coef = sign * coef_a * coef_b
                result[prod_blade] = result.get(prod_blade, 0.0) + prod_coef
        return Multivector(result, self.n)

    # -----------------------------------------------------------------------
    # Grade extraction & utilities
    # -----------------------------------------------------------------------
    def grade(self, k: int) -> "Multivector":
        """Return a new Multivector keeping only grade‑k blades."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def norm(self) -> float:
        """Euclidean norm of the multivector (using scalar product with reverse)."""
        rev = self.reversion()
        prod = (self * rev).scalar_part()
        return math.sqrt(abs(prod))

    def reversion(self) -> "Multivector":
        """Reverse (grade‑dependent sign change)."""
        rev_components = {}
        for blade, coef in self.components.items():
            grade = len(blade)
            sign = (-1) ** (grade * (grade - 1) // 2)
            rev_components[blade] = sign * coef
        return Multivector(rev_components, self.n)

    # -----------------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------------
    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), tuple(sorted(x[0])))):
            if blade:
                basis = "e" + "".join(str(i) for i in sorted(blade))
            else:
                basis = "1"
            terms.append(f"{_pct(coef)}*{basis}")
        return " + ".join(terms)


# ---------------------------------------------------------------------------
# Feature extraction (Parent B)
# ---------------------------------------------------------------------------

def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑random feature vector from a string."""
    rnd = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "operator_ledger_density",
        "operator_recursion_score", "operator_directive_ratio",
        "operator_target_density", "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy", "psyche_dissociative_index",
        "psyche_wrath_velocity", "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric", "resilience_swarm_orchestration_density",
        "resilience_logic_crucifixion_index", "resilience_conspiracy_grounding_ratio",
        "resilience_chaotic_good_tax", "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density", "rainmaker_asset_structuring_weight",
        "rainmaker_pitch_formatting_ratio", "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline", "telemetry_manic_velocity",
    ]
    return {k: rnd.random() * 10 for k in keys}


def extract_master_vector(text: str) -> Dict[str, float]:
    """Select a subset of the full features that will be embedded in the algebra."""
    if not text.strip():
        return {}
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "ledger_density": f.get("operator_ledger_density", 0.0),
        "recursion_score": f.get("operator_recursion_score", 0.0),
        "directive_ratio": f.get("operator_directive_ratio", 0.0),
        "target_density": f.get("operator_target_density", 0.0),
        "forensic_shield_ratio": f.get("psyche_forensic_shield_ratio", 0.0),
        "poetic_entropy": f.get("psyche_poetic_entropy", 0.0),
        "dissociative_index": f.get("psyche_dissociative_index", 0.0),
        "wrath_velocity": f.get("psyche_wrath_velocity", 0.0),
        "bureaucratic_weaponization_index": f.get(
            "resilience_bureaucratic_weaponization_index", 0.0
        ),
        "resource_exhaustion_metric": f.get(
            "resilience_resource_exhaustion_metric", 0.0
        ),
        "swarm_orchestration_density": f.get(
            "resilience_swarm_orchestration_density", 0.0
        ),
        "logic_crucifixion_index": f.get(
            "resilience_logic_crucifixion_index", 0.0
        ),
        "conspiracy_grounding_ratio": f.get(
            "resilience_conspiracy_grounding_ratio", 0.0
        ),
        "chaotic_good_tax": f.get("resilience_chaotic_good_tax", 0.0),
        "corporate_grit_tension": f.get("rainmaker_corporate_grit_tension", 0.0),
        "countdown_density": f.get("rainmaker_countdown_density", 0.0),
        "asset_structuring_weight": f.get("rainmaker_asset_structuring_weight", 0.0),
        "pitch_formatting_ratio": f.get("rainmaker_pitch_formatting_ratio", 0.0),
        "agent_symmetry_ratio": f.get("telemetry_agent_symmetry_ratio", 0.0),
        "protocol_discipline": f.get("telemetry_protocol_discipline", 0.0),
        "manic_velocity": f.get("telemetry_manic_velocity", 0.0),
    }


# ---------------------------------------------------------------------------
# Fusion utilities
# ---------------------------------------------------------------------------

# Fixed ordering of features → basis indices
_FEATURE_ORDER = sorted(extract_master_vector("dummy").keys())
_FEATURE_TO_IDX = {name: i for i, name in enumerate(_FEATURE_ORDER)}
_ALGEBRA_DIM = len(_FEATURE_ORDER)  # n for Cl(n,0)


def features_to_multivector(features: Dict[str, float]) -> Multivector:
    """Embed a feature dict into a grade‑1 multivector.
    Each feature becomes coefficient of a distinct basis vector e_i."""
    comps: Dict[frozenset, float] = {}
    for name, value in features.items():
        if name in _FEATURE_TO_IDX:
            idx = _FEATURE_TO_IDX[name]
            comps[frozenset({idx})] = float(value)
    return Multivector(comps, _ALGEBRA_DIM)


def multivector_similarity(mv1: Multivector, mv2: Multivector) -> float:
    """Scalar part of the geometric product = dot‑product similarity."""
    prod = mv1 * mv2
    return prod.scalar_part()


def combine_texts(text_a: str, text_b: str) -> float:
    """High‑level bridge: extract features, embed, and return similarity."""
    vec_a = features_to_multivector(extract_master_vector(text_a))
    vec_b = features_to_multivector(extract_master_vector(text_b))
    return multivector_similarity(vec_a, vec_b)


def ricci_like_curvature(texts: List[str]) -> float:
    """A curvature‑inspired metric.
    For each unordered pair we compute the squared norm of the bivector (grade‑2)
    part of the geometric product; the average of those norms is returned."""
    if len(texts) < 2:
        return 0.0
    multivectors = [features_to_multivector(extract_master_vector(t)) for t in texts]
    bivector_norms = []
    for i in range(len(multivectors)):
        for j in range(i + 1, len(multivectors)):
            prod = multivectors[i] * multivectors[j]
            biv = prod.grade(2)
            biv_norm = biv.norm()
            bivector_norms.append(biv_norm ** 2)
    return sum(bivector_norms) / len(bivector_norms)


def aggregate_multivector(texts: List[str]) -> Multivector:
    """Sum of all embedded feature multivectors (a collective representation)."""
    agg = Multivector({}, _ALGEBRA_DIM)
    for t in texts:
        agg += features_to_multivector(extract_master_vector(t))
    return agg


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    a = "The quick brown fox jumps over the lazy dog."
    b = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    c = "Artificial intelligence blends mathematics and language."

    # Pairwise similarity
    sim_ab = combine_texts(a, b)
    sim_ac = combine_texts(a, c)
    sim_bc = combine_texts(b, c)

    print(f"Similarity AB: {_pct(sim_ab)}")
    print(f"Similarity AC: {_pct(sim_ac)}")
    print(f"Similarity BC: {_pct(sim_bc)}")

    # Curvature‑like metric on a triple
    curvature = ricci_like_curvature([a, b, c])
    print(f"Ricci‑like curvature (triple): {_pct(curvature)}")

    # Aggregate multivector and its norm
    agg_mv = aggregate_multivector([a, b, c])
    print(f"Aggregate multivector: {agg_mv}")
    print(f"Aggregate norm: {_pct(agg_mv.norm())}")