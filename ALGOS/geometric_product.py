#!/usr/bin/env python3
"""Clifford geometric product over Cl(n,0) Euclidean algebra.

ab = a·b + a∧b  (inner + outer product unified).
Basis blades satisfy: e_i * e_i = 1, e_i * e_j = -e_j * e_i for i != j.
Multivectors unify scalars (grade-0), vectors (grade-1), bivectors (grade-2),
trivectors (grade-3), … into a single graded algebra.  Coordinate-free
rotations via rotors: R = cos(θ/2) − sin(θ/2)·(e_i ∧ e_j).
"""
from __future__ import annotations
import math
import numpy as np


# ---------------------------------------------------------------------------
# Core blade arithmetic helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Multivector
# ---------------------------------------------------------------------------

class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades.

    components: dict mapping frozenset(basis_indices) -> float coefficient.
                frozenset() is the scalar (grade-0) blade.
    n: dimension of the base vector space.
    """

    def __init__(self, components, n):
        # Drop zero coefficients to keep repr clean
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    # ------------------------------------------------------------------
    # Grade projection
    # ------------------------------------------------------------------

    def grade(self, k):
        """Return a new Multivector keeping only grade-k blades."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items()
             if len(blade) == k},
            self.n,
        )

    def scalar_part(self):
        """Return the scalar (grade-0) coefficient."""
        return self.components.get(frozenset(), 0.0)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self):
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
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other):
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if v != 0.0}, self.n)

    def __sub__(self, other):
        neg = Multivector({k: -v for k, v in other.components.items()}, other.n)
        return self.__add__(neg)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Multivector({k: v * other for k, v in self.components.items()}, self.n)
        return geometric_product(self, other)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __neg__(self):
        return Multivector({k: -v for k, v in self.components.items()}, self.n)


# ---------------------------------------------------------------------------
# Product functions
# ---------------------------------------------------------------------------

def geometric_product(a, b):
    """Full Clifford product ab in Cl(n,0)."""
    result = {}
    for blade_a, coef_a in a.components.items():
        for blade_b, coef_b in b.components.items():
            blade_out, sign = _multiply_blades(blade_a, blade_b)
            contrib = sign * coef_a * coef_b
            result[blade_out] = result.get(blade_out, 0.0) + contrib
    n = max(a.n, b.n)
    return Multivector({k: v for k, v in result.items() if v != 0.0}, n)


def inner_product(a, b):
    """Symmetric inner product (ab + ba) / 2."""
    ab = geometric_product(a, b)
    ba = geometric_product(b, a)
    return (ab + ba) * 0.5


def outer_product(a, b):
    """Antisymmetric wedge product (ab − ba) / 2."""
    ab = geometric_product(a, b)
    ba = geometric_product(b, a)
    return (ab - ba) * 0.5


def reverse(a):
    """Reverse of a multivector: flip sign on blades of grade k where k%4 in {2,3}."""
    result = {}
    for blade, coef in a.components.items():
        k = len(blade)
        # Number of swaps needed to reverse a k-blade is k*(k-1)/2
        sign = -1 if (k % 4) in (2, 3) else 1
        result[blade] = coef * sign
    return Multivector(result, a.n)


# ---------------------------------------------------------------------------
# Rotor construction and application
# ---------------------------------------------------------------------------

def rotor(angle, plane_i, plane_j, n):
    """Build rotation rotor R = cos(θ/2) − sin(θ/2)·(e_i ∧ e_j).

    Rotating by `angle` radians in the e_i/e_j plane (0-indexed).
    """
    half = angle / 2.0
    # Scalar part
    components = {frozenset(): math.cos(half)}
    # Bivector part: e_i ∧ e_j  (i < j by convention for positive orientation)
    i, j = (plane_i, plane_j) if plane_i < plane_j else (plane_j, plane_i)
    sign = 1 if plane_i < plane_j else -1
    blade = frozenset({i, j})
    components[blade] = -sign * math.sin(half)
    return Multivector(components, n)


def rotate_vector(v, angle, plane_i=0, plane_j=1):
    """Sandwich product R v R† where R is the rotor for `angle` in plane (i,j).

    v: Multivector (grade-1) or numpy array.
    Returns a Multivector.
    """
    if isinstance(v, np.ndarray):
        v = vec_to_multivector(v)
    n = v.n
    R = rotor(angle, plane_i, plane_j, n)
    R_rev = reverse(R)
    return geometric_product(geometric_product(R, v), R_rev)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def vec_to_multivector(v):
    """Convert a numpy 1-D array to a grade-1 Multivector."""
    v = np.asarray(v, dtype=float)
    components = {}
    for i, val in enumerate(v):
        if val != 0.0:
            components[frozenset({i})] = float(val)
    return Multivector(components, len(v))


def multivector_to_vec(m, n):
    """Extract grade-1 components from Multivector into a numpy array of length n."""
    out = np.zeros(n)
    for blade, coef in m.components.items():
        if len(blade) == 1:
            (i,) = blade
            if i < n:
                out[i] = coef
    return out


# ---------------------------------------------------------------------------
# Main: 2-D rotation sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    angle = math.pi / 3  # 60 degrees
    n = 2

    # Input vector [1, 0]
    v_np = np.array([1.0, 0.0])
    v_mv = vec_to_multivector(v_np)

    # Clifford rotor rotation
    rotated_mv = rotate_vector(v_mv, angle, plane_i=0, plane_j=1)
    clifford_result = multivector_to_vec(rotated_mv, n)

    # Standard numpy rotation matrix
    c, s = math.cos(angle), math.sin(angle)
    rot_mat = np.array([[c, -s], [s, c]])
    numpy_result = rot_mat @ v_np

    print(f"Input vector        : {v_np}")
    print(f"Rotation angle      : {math.degrees(angle):.1f} degrees")
    print(f"Clifford rotor result: {clifford_result}")
    print(f"NumPy rot-matrix     : {numpy_result}")

    diff = np.linalg.norm(clifford_result - numpy_result)
    print(f"L2 difference       : {diff:.2e}")

    if diff < 1e-12:
        print("PASS — rotor and rotation matrix agree to machine precision.")
    else:
        print("FAIL — results diverge.", file=sys.stderr)
        sys.exit(1)

    # Additional: show grade decomposition
    print()
    print("Rotor R:", rotor(angle, 0, 1, n))
    print("Rotated multivector (full):", rotated_mv)
    print("Grade-0 part of rotated v:", rotated_mv.grade(0))
    print("Grade-1 part of rotated v:", rotated_mv.grade(1))
    print("Grade-2 part of rotated v:", rotated_mv.grade(2))
