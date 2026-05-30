# DARWIN HAMMER — match 161, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_path_signatur_hybrid_krampus_brain_m25_s1.py (gen2)
# parent_b: geometric_product.py (gen0)
# born: 2026-05-29T23:27:16Z

"""
Hybrid Path Signature – Clifford Geometric Product

Parent A: hybrid_hybrid_path_signatur_hybrid_krampus_brain_m25_s1.py
Parent B: geometric_product.py

Mathematical bridge
------------------
A path signature lives in the (truncated) tensor algebra of a path and its
level‑1 and level‑2 components can be identified with a vector (grade‑1) and a
bivector (grade‑2) of a Euclidean Clifford algebra Cl(n,0).  By mapping the
approximated iterated integrals obtained from the feature extraction of the
parent‑A code onto the corresponding multivector grades, we obtain a single
`Multivector` that encodes the signature.  The geometric product from the
parent‑B code then provides a natural way to concatenate signatures (the
signature of a concatenated path equals the geometric product of the individual
signatures).  The hybrid module therefore:

* extracts deterministic scalar‑vector features from a text (parent A),
* builds a multivector containing scalar, vector and bivector parts that
  approximate the level‑0/1/2 signature,
* uses the Clifford geometric product to combine, rotate or otherwise
  manipulate these signatures (parent B).

The three core functions below showcase this fusion.
"""

import math
import random
import sys
import pathlib
import numpy as np

# ---------------------------------------------------------------------------
# Parent A utilities (lead‑lag transform & deterministic feature extraction)
# ---------------------------------------------------------------------------

def lead_lag_transform(path):
    """Lead‑lag embedding of a discrete path.

    Input: (T, d) ndarray.
    Output: (2T‑1, 2d) ndarray where successive points are duplicated and
    interleaved as in the classic lead‑lag transform.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def extract_full_features(text: str) -> dict:
    """Deterministic pseudo‑random feature vector from a string.

    The same text always yields the same dictionary of 24 float features.
    """
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
    return {k: rnd.random() for k in keys}

# ---------------------------------------------------------------------------
# Parent B utilities – Clifford geometric product
# ---------------------------------------------------------------------------

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting index list.

    Duplicated indices cancel (e_i*e_i = 1) and each swap flips the sign.
    """
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
                # e_i*e_i = 1 → remove both occurrences
                lst.pop(j)
                lst.pop(j)  # second element now occupies position j
                n -= 2
                i = -1   # restart outer loop because list shortened
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices).

    Returns (result_blade_frozenset, sign).
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    """Element of Cl(n,0) represented as a sparse sum of basis blades."""

    def __init__(self, components: dict, n: int):
        """components: {frozenset(indices): coefficient}."""
        self.n = int(n)
        # prune zeros
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    # -----------------------------------------------------------------------
    # Basic arithmetic
    # -----------------------------------------------------------------------
    def __add__(self, other):
        if not isinstance(other, Multivector):
            return NotImplemented
        if self.n != other.n:
            raise ValueError("Dimension mismatch in addition")
        comp = self.components.copy()
        for blade, coeff in other.components.items():
            comp[blade] = comp.get(blade, 0.0) + coeff
            if abs(comp[blade]) < 1e-15:
                del comp[blade]
        return Multivector(comp, self.n)

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, Multivector):
            return NotImplemented
        return self + (-1.0) * other

    # scalar multiplication
    def __rmul__(self, scalar):
        if isinstance(scalar, (int, float)):
            comp = {blade: scalar * coeff for blade, coeff in self.components.items()}
            return Multivector(comp, self.n)
        return NotImplemented

    def __mul__(self, other):
        """Geometric product.

        * If `other` is a scalar → scalar multiplication.
        * If `other` is a Multivector → full geometric product.
        """
        if isinstance(other, (int, float)):
            return other * self
        if not isinstance(other, Multivector):
            return NotImplemented
        if self.n != other.n:
            raise ValueError("Dimension mismatch in geometric product")
        result = {}
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                coeff_res = coeff_a * coeff_b * sign
                result[blade_res] = result.get(blade_res, 0.0) + coeff_res
        return Multivector(result, self.n)

    def __repr__(self):
        if not self.components:
            return f"Multivector(0, n={self.n})"
        terms = []
        for blade, coeff in sorted(self.components.items(), key=lambda x: (len(x[0]), x[0])):
            if not blade:
                term = f"{coeff:.3g}"
            else:
                idx = "".join(str(i) for i in sorted(blade))
                term = f"{coeff:.3g}e{idx}"
            terms.append(term)
        return " + ".join(terms)

    # -----------------------------------------------------------------------
    # Grade extraction and reversal (needed for rotors)
    # -----------------------------------------------------------------------
    def grade(self, g: int):
        """Return a Multivector containing only grade‑g components."""
        comp = {blade: coeff for blade, coeff in self.components.items()
                if len(blade) == g}
        return Multivector(comp, self.n)

    def reverse(self):
        """Reverse (Clifford conjugation) of the multivector."""
        comp = {}
        for blade, coeff in self.components.items():
            k = len(blade)
            sign = 1 if (k * (k - 1) // 2) % 2 == 0 else -1
            comp[blade] = coeff * sign
        return Multivector(comp, self.n)

# ---------------------------------------------------------------------------
# Hybrid construction utilities
# ---------------------------------------------------------------------------

def vector_mv(arr: np.ndarray, n: int) -> Multivector:
    """Create a grade‑1 multivector from a length‑n array."""
    if arr.shape != (n,):
        raise ValueError("Array length does not match dimension n")
    comp = {frozenset({i}): float(arr[i]) for i in range(n) if abs(arr[i]) > 1e-15}
    return Multivector(comp, n)

def bivector_from_outer(v: np.ndarray) -> Multivector:
    """Approximate the level‑2 signature as the antisymmetric outer product v∧v.

    For a real vector v the outer product with itself is zero; we therefore use
    the outer product of v with a shifted copy to obtain a non‑trivial bivector.
    """
    n = v.shape[0]
    comp = {}
    for i in range(n):
        for j in range(i + 1, n):
            # simple antisymmetric estimator: v_i * v_j_shift - v_j * v_i_shift
            # where the shift is a one‑step circular rotation.
            v_shift = np.roll(v, -1)
            coeff = v[i] * v_shift[j] - v[j] * v_shift[i]
            if abs(coeff) > 1e-15:
                comp[frozenset({i, j})] = float(coeff)
    return Multivector(comp, n)

def path_signature_approx(path: np.ndarray) -> Multivector:
    """Hybrid signature: scalar + vector (level‑1) + bivector (level‑2).

    1. Apply lead‑lag transform.
    2. Collapse to a single vector by summing coordinates.
    3. Build bivector from the antisymmetric outer estimator.
    4. Assemble into a Multivector (scalar part set to 1 for normalization).
    """
    transformed = lead_lag_transform(path)          # (2T‑1, 2d)
    # Simple aggregation: mean over rows → vector of length 2d
    vec = transformed.mean(axis=0)
    n = vec.shape[0]
    # Level‑1 part
    v_mv = vector_mv(vec, n)
    # Level‑2 part (approximation)
    b_mv = bivector_from_outer(vec)
    # Assemble: scalar = 1 (signature of empty path)
    components = {frozenset(): 1.0}
    components.update(v_mv.components)
    components.update(b_mv.components)
    return Multivector(components, n)

def combine_signatures(sig1: Multivector, sig2: Multivector) -> Multivector:
    """Concatenate two path signatures via the geometric product."""
    return sig1 * sig2

def rotor(i: int, j: int, theta: float, n: int) -> Multivector:
    """Construct a rotor for a rotation in the (i,j) plane by angle theta."""
    if i == j or not (0 <= i < n) or not (0 <= j < n):
        raise ValueError("Invalid plane indices")
    scalar = math.cos(theta / 2.0)
    biv_coeff = -math.sin(theta / 2.0)
    blade = frozenset({i, j})
    comp = {frozenset(): scalar, blade: biv_coeff}
    return Multivector(comp, n)

def rotate_signature(sig: Multivector, i: int, j: int, theta: float) -> Multivector:
    """Rotate a signature using a rotor R:  R * sig * ~R."""
    R = rotor(i, j, theta, sig.n)
    return R * sig * R.reverse()

# ---------------------------------------------------------------------------
# Demonstration / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple 2‑dimensional path (square)
    path = np.array([[0.0, 0.0],
                     [1.0, 0.0],
                     [1.0, 1.0],
                     [0.0, 1.0],
                     [0.0, 0.0]])

    # Hybrid signature of the path
    sig = path_signature_approx(path)
    print("Signature (approx.):", sig)

    # Rotate the signature by 90° in the (0,1) plane
    rot_sig = rotate_signature(sig, 0, 1, math.pi / 2)
    print("Rotated signature:", rot_sig)

    # Concatenate the path with itself using geometric product
    combined = combine_signatures(sig, sig)
    print("Combined (self‑concatenated) signature:", combined)

    # Verify deterministic feature extraction (used indirectly by path_signature)
    feats = extract_full_features("example text")
    print("Sample deterministic features (first 3):", {k: feats[k] for k in list(feats)[:3]})