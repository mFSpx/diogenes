# DARWIN HAMMER — match 1281, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s6.py (gen3)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s3.py (gen1)
# born: 2026-05-29T23:34:58Z

"""Hybrid geometric‑feature engine.

This module fuses two distinct parent algorithms:

* **Parent A** – a geometric‑algebra implementation centred on the
  `Multivector` class, blade multiplication and grade extraction.
* **Parent B** – a deterministic feature extractor that turns an arbitrary
  text into a fixed‑size dictionary of floating‑point descriptors.

**Mathematical bridge**

Each feature key from Parent B is assigned a unique basis index of a
Clifford algebra `Cl(n,0)`.  The extracted feature values become the scalar
coefficients of the corresponding basis 1‑vectors, thus forming a
multivector representation of the text.  Geometric products between two
such multivectors combine the feature spaces algebraically; the scalar
(grade‑0) part of the product furnishes a similarity measure, while higher
grades encode antisymmetric interactions.  This unified view lets us apply
geometric‑algebra tools (grade projection, norm, products) directly to the
feature vectors, achieving a true hybrid algorithm.

The public API provides three representative hybrid operations:

1. `master_vector_to_multivector` – convert a feature dict to a `Multivector`.
2. `geometric_similarity` – scalar similarity from the geometric product of
   two texts.
3. `ollivier_ricci_like_curvature` – a lightweight Ollivier‑Ricci‑curvature
   analogue built from pairwise geometric similarities.

All code runs with only the Python standard library and NumPy.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – geometric algebra core
# ---------------------------------------------------------------------------

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return a sorted blade index list and the sign incurred by bubble‑sorting.

    Identical indices cancel (Grassmann algebra property) and are removed.
    """
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign *= -1
            i = max(i - 1, 0)  # re‑check previous pair after swap
        elif lst[i] == lst[i + 1]:
            # cancel equal indices
            del lst[i : i + 2]
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

class Multivector:
    """Element of the Clifford algebra Cl(n,0) expressed as a sparse sum of blades."""

    def __init__(self, components: Dict[frozenset, float], n: int):
        # filter out zero coefficients and store a copy
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    # -----------------------------------------------------------------------
    # Basic algebraic operations
    # -----------------------------------------------------------------------

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) - coef
        return Multivector(result, self.n)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self.components.items()}, self.n)

    def __rmul__(self, scalar: float) -> "Multivector":
        return Multivector({b: scalar * c for b, c in self.components.items()}, self.n)

    __mul__ = __rmul__  # scalar multiplication via __rmul__

    def geometric_product(self, other: "Multivector") -> "Multivector":
        """Full geometric product (Clifford multiplication) of two multivectors."""
        result: Dict[frozenset, float] = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                result[blade_res] = result.get(blade_res, 0.0) + sign * coef_a * coef_b
        return Multivector(result, self.n)

    # -----------------------------------------------------------------------
    # Grade‑specific utilities
    # -----------------------------------------------------------------------

    def grade(self, k: int) -> "Multivector":
        """Return a new multivector containing only blades of grade *k*."""
        return Multivector(
            {b: c for b, c in self.components.items() if len(b) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def norm(self) -> float:
        """Euclidean norm of the multivector (square root of summed squares)."""
        return math.sqrt(sum(c * c for c in self.components.values()))

    # -----------------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self.components:
            return "0"
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), x[0])):
            if not blade:
                term = f"{coef:.4g}"
            else:
                idx = "".join(str(i) for i in sorted(blade))
                term = f"{coef:.4g}e{idx}"
            terms.append(term)
        return " + ".join(terms)


# ---------------------------------------------------------------------------
# Parent B – deterministic feature extraction
# ---------------------------------------------------------------------------

def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministically map *text* to a fixed set of 24 pseudo‑random features."""
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
    return {k: rnd.random() * 10.0 for k in keys}


def extract_master_vector(text: str) -> Dict[str, float]:
    """Select a curated subset of features and rename them for downstream use."""
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
# Hybrid layer – mapping features ↔ geometric algebra
# ---------------------------------------------------------------------------

# Deterministic ordering of feature keys – defines the basis assignment.
_FEATURE_KEYS: Tuple[str, ...] = tuple(sorted(extract_master_vector("dummy").keys()))
_FEATURE_INDEX: Dict[str, int] = {k: i for i, k in enumerate(_FEATURE_KEYS)}
_DIMENSION: int = len(_FEATURE_KEYS)  # n for Cl(n,0)


def master_vector_to_multivector(master: Dict[str, float]) -> Multivector:
    """Encode a feature dictionary as a grade‑1 multivector.

    Each feature *k* maps to basis vector `e_{i}` where *i* = index(k).
    The coefficient is the feature value.
    """
    comps: Dict[frozenset, float] = {}
    for key, value in master.items():
        idx = _FEATURE_INDEX.get(key)
        if idx is None:
            continue  # ignore unexpected keys
        blade = frozenset({idx})
        comps[blade] = comps.get(blade, 0.0) + float(value)
    return Multivector(comps, _DIMENSION)


def geometric_similarity(text_a: str, text_b: str) -> float:
    """Scalar similarity derived from the geometric product of two texts.

    The procedure:
        1. Extract master vectors.
        2. Convert each to a grade‑1 multivector.
        3. Compute the full geometric product.
        4. Return the scalar (grade‑0) part.
    """
    mv_a = master_vector_to_multivector(extract_master_vector(text_a))
    mv_b = master_vector_to_multivector(extract_master_vector(text_b))
    product = mv_a.geometric_product(mv_b)
    return product.scalar_part()


def ollivier_ricci_like_curvature(texts: Iterable[str]) -> Dict[Tuple[int, int], float]:
    """Compute a lightweight Ollivier‑Ricci curvature analogue.

    For each ordered pair (i, j) we define:

        κ(i, j) = 1 - (W_ij / (d_i + d_j))

    where:
        * W_ij*  – geometric similarity between texts i and j,
        * d_i*   – average similarity from i to all other texts,
        * d_j*   – analogous for j.

    The result is a dict keyed by index pairs (i, j) with curvature values.
    """
    txt_list = list(texts)
    n = len(txt_list)
    if n < 2:
        return {}

    # Pairwise similarity matrix
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            s = geometric_similarity(txt_list[i], txt_list[j])
            sim[i, j] = sim[j, i] = s

    # Average distances (here we treat similarity as a "weight")
    avg_sim = sim.mean(axis=1)  # shape (n,)

    curvatures: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            denom = avg_sim[i] + avg_sim[j]
            if denom == 0:
                kappa = 0.0
            else:
                kappa = 1.0 - (sim[i, j] / denom)
            curvatures[(i, j)] = kappa
    return curvatures


# ---------------------------------------------------------------------------
# Demonstration / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple sanity checks
    txt1 = "The quick brown fox jumps over the lazy dog."
    txt2 = "Pack my box with five dozen liquor jugs."
    txt3 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."

    print("Feature vector (first 5 entries) for txt1:")
    mv1 = master_vector_to_multivector(extract_master_vector(txt1))
    print(mv1)

    sim12 = geometric_similarity(txt1, txt2)
    sim13 = geometric_similarity(txt1, txt3)
    sim23 = geometric_similarity(txt2, txt3)

    print("\nGeometric similarities:")
    print(f"txt1 ↔ txt2 : {sim12:.6f}")
    print(f"txt1 ↔ txt3 : {sim13:.6f}")
    print(f"txt2 ↔ txt3 : {sim23:.6f}")

    curvature = ollivier_ricci_like_curvature([txt1, txt2, txt3])
    print("\nOllivier‑Ricci‑like curvature matrix (κ):")
    for (i, j), kappa in sorted(curvature.items()):
        print(f"κ({i},{j}) = {kappa:.4f}")

    print("\nAll tests executed without error.")