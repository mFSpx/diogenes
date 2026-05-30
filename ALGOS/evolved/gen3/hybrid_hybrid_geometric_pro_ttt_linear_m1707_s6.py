# DARWIN HAMMER — match 1707, survivor 6
# gen: 3
# parent_a: hybrid_geometric_product_hybrid_model_vram_sc_m22_s1.py (gen2)
# parent_b: ttt_linear.py (gen0)
# born: 2026-05-29T23:38:23Z

"""
Hybrid Geometric‑Product Test‑Time Training (Hybrid‑GP‑TTT)

Parents
-------
* **geometric_product.py** – defines a Clifford algebra multivector with a
  geometric product.
* **ttt_linear.py** – implements Test‑Time Training (TTT) where the weight
  matrix **W** is updated online by a gradient‑descent step on the
  reconstruction loss  ‖W x − x‖².

Mathematical Bridge
-------------------
The bridge is the *representation* of the weight matrix **W** as a
multivector **Ŵ** in the Clifford algebra **Cl(n,0)**.  Each matrix entry
`W[i, j]` is stored as the scalar coefficient of a distinct basis blade
`e_i ∧ e_{n_out + j}` (i.e. the wedge of an “output” basis vector and an
“input” basis vector).  With this encoding the ordinary matrix‑vector
product `W x` becomes a *geometric product* between **Ŵ** and the vector
multivector **x̂**.  Consequently the TTT gradient update

    W ← W − η ∇_W L

can be performed entirely in the multivector domain using the geometric
product for the linear map and the Clifford‑algebra addition for the
parameter update.

The module below implements this hybrid system:
* conversion utilities between matrices and multivectors,
* a full geometric‑product implementation,
* TTT‑style online update that operates on the multivector representation,
* a forward pass that extracts the conventional hidden state from the
  geometric product.

All operations rely only on NumPy and the standard library.
"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import Dict, FrozenSet, Tuple

# ----------------------------------------------------------------------
# Clifford algebra utilities (parent A)
# ----------------------------------------------------------------------


def _blade_sign(indices: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """Return a sorted tuple of indices and the sign incurred by sorting.

    Identical indices cancel (e ∧ e = 0).  The algorithm is a bubble sort that
    counts swaps; when a duplicate is found the pair is removed and the sign
    is unchanged.
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
                # cancel the duplicate basis vector
                lst.pop(j)
                lst.pop(j)  # second element now occupies position j
                n -= 2
                continue
            j += 1
        i += 1
    return tuple(lst), sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Geometric product of two basis blades.

    Returns the resulting blade (as a frozenset) and the sign (+1 / -1).
    """
    combined = tuple(list(blade_a) + list(blade_b))
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sparse sum of basis blades."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # store only non‑zero coefficients
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    # ------------------------------------------------------------------
    # Basic algebraic operations
    # ------------------------------------------------------------------

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    __radd__ = __add__

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self.components.items()}, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __rmul__(self, scalar: float) -> "Multivector":
        return Multivector({b: scalar * c for b, c in self.components.items()}, self.n)

    __mul__ = None  # placeholder – will be defined after the class

    # ------------------------------------------------------------------
    # Extraction utilities
    # ------------------------------------------------------------------

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def grade(self, k: int) -> "Multivector":
        """Return a new multivector containing only blades of grade k."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    # ------------------------------------------------------------------
    # Pretty printing (useful for debugging)
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                basis = "^".join(str(i) for i in sorted(blade))
                terms.append(f"{coef:.3g}*e{{{basis}}}")
            else:
                terms.append(f"{coef:.3g}")
        return " + ".join(terms) if terms else "0"


def _geometric_product(mv_a: Multivector, mv_b: Multivector) -> Multivector:
    """Full geometric product (Clifford) between two multivectors."""
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in mv_a.components.items():
        for blade_b, coeff_b in mv_b.components.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            result[blade_res] = result.get(blade_res, 0.0) + sign * coeff_a * coeff_b
    return Multivector(result, mv_a.n)


# bind the operator
Multivector.__mul__ = _geometric_product  # type: ignore


# ----------------------------------------------------------------------
# Conversion between matrices / vectors and multivectors
# ----------------------------------------------------------------------


def matrix_to_multivector(W: np.ndarray, n_out: int, n_in: int) -> Multivector:
    """Encode a (n_out, n_in) matrix as a multivector.

    Each entry W[i, j] becomes the scalar of the blade
    e_i ∧ e_{n_out + j}.  The total algebra dimension is n = n_out + n_in.
    """
    n = n_out + n_in
    comps: Dict[FrozenSet[int], float] = {}
    for i in range(n_out):
        for j in range(n_in):
            val = float(W[i, j])
            if abs(val) > 1e-15:
                blade = frozenset({i, n_out + j})
                comps[blade] = val
    return Multivector(comps, n)


def multivector_to_matrix(MV: Multivector, n_out: int, n_in: int) -> np.ndarray:
    """Decode a multivector back to a (n_out, n_in) matrix."""
    W = np.zeros((n_out, n_in), dtype=float)
    for blade, coef in MV.components.items():
        if len(blade) == 2:
            i, j = sorted(blade)
            if i < n_out and j >= n_out:
                W[i, j - n_out] = coef
    return W


def vector_to_multivector(v: np.ndarray, offset: int = 0) -> Multivector:
    """Encode a vector as a grade‑1 multivector (sum of basis vectors).

    The optional offset allows embedding the vector into a larger algebra.
    """
    n = len(v) + offset
    comps = {frozenset({idx + offset}): float(val) for idx, val in enumerate(v)}
    return Multivector(comps, n)


def multivector_to_vector(MV: Multivector, dim: int, offset: int = 0) -> np.ndarray:
    """Extract the grade‑1 part (vector) from a multivector."""
    vec = np.zeros(dim, dtype=float)
    for blade, coef in MV.components.items():
        if len(blade) == 1:
            idx = next(iter(blade)) - offset
            if 0 <= idx < dim:
                vec[idx] = coef
    return vec


# ----------------------------------------------------------------------
# Hybrid TTT operations (core of the fusion)
# ----------------------------------------------------------------------


def init_hybrid(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> Tuple[Multivector, int, int]:
    """Initialize a random weight matrix and return its multivector encoding.

    Returns
    -------
    W_mv : Multivector
        Clifford encoding of the weight matrix.
    n_out, n_in : int
        Output and input dimensions (needed for later decoding).
    """
    if d_out is None:
        d_out = d_in
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d_out, d_in)) * scale
    W_mv = matrix_to_multivector(W, d_out, d_in)
    return W_mv, d_out, d_in


def hybrid_loss(W_mv: Multivector, x: np.ndarray, n_out: int, n_in: int) -> float:
    """Reconstruction loss ‖W x − x‖² using the geometric product.

    The matrix product `W x` is obtained via `W_mv * x̂` and then projected
    onto the input‑space vector part.
    """
    x_mv = vector_to_multivector(x, offset=n_out)  # embed input after output basis
    prod_mv = W_mv * x_mv
    y = multivector_to_vector(prod_mv, n_out, offset=0)  # output lives in first n_out slots
    # For reconstruction we compare the *first* n_in components of y with x.
    # When n_out != n_in we simply truncate/pad.
    y_trunc = y[:n_in] if n_out >= n_in else np.pad(y, (0, n_in - n_out))
    diff = y_trunc - x
    return float(np.dot(diff, diff))


def hybrid_step(W_mv: Multivector, x: np.ndarray, eta: float, n_out: int, n_in: int) -> Multivector:
    """One TTT gradient‑descent step performed in the Clifford domain.

    The gradient of the reconstruction loss w.r.t. the matrix is
        G = 2 * (W x - x) xᵀ
    We encode G as a multivector and update via Clifford addition.
    """
    # ----- ordinary matrix view for gradient computation -----
    W = multivector_to_matrix(W_mv, n_out, n_in)
    y = W @ x
    grad_mat = 2.0 * np.outer(y - x, x)  # shape (n_out, n_in)

    # ----- encode gradient as multivector -----
    G_mv = matrix_to_multivector(grad_mat, n_out, n_in)

    # ----- update using Clifford addition (scalar multiplication works) -----
    W_new_mv = W_mv + (-eta) * G_mv
    return W_new_mv


def hybrid_forward(W_mv: Multivector, x: np.ndarray, n_out: int, n_in: int) -> np.ndarray:
    """Compute the hidden state h_t = W_t x using the geometric product.

    Returns a NumPy array of shape (n_out,).
    """
    x_mv = vector_to_multivector(x, offset=n_out)
    prod_mv = W_mv * x_mv
    h = multivector_to_vector(prod_mv, n_out, offset=0)
    return h


# ----------------------------------------------------------------------
# Convenience wrappers for full sequences (mirrors parent B)
# ----------------------------------------------------------------------


def hybrid_sequence(xs: np.ndarray, eta: float, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Run hybrid TTT over a sequence of input vectors.

    Parameters
    ----------
    xs : np.ndarray, shape (T, d_in)
        Input sequence.
    eta : float
        Learning rate.
    scale, seed : passed to the initializer.

    Returns
    -------
    H : np.ndarray, shape (T, d_out)
        Hidden states after each step.
    """
    T, d_in = xs.shape
    W_mv, n_out, n_in = init_hybrid(d_in, scale=scale, seed=seed)
    H = np.zeros((T, n_out), dtype=float)

    for t in range(T):
        x = xs[t]
        H[t] = hybrid_forward(W_mv, x, n_out, n_in)
        W_mv = hybrid_step(W_mv, x, eta, n_out, n_in)

    return H


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple sanity check
    d = 4
    rng = np.random.default_rng(42)
    xs = rng.standard_normal((5, d))

    eta = 0.05
    H = hybrid_sequence(xs, eta)

    # Verify shapes and that the code runs without exception
    assert H.shape == (5, d), "Hidden state shape mismatch"
    # Print first hidden vector for visual confirmation
    print("First hidden state:", H[0])
    # Compute loss before and after a single step to ensure it changes
    W_mv, n_out, n_in = init_hybrid(d, scale=0.01, seed=1)
    loss_before = hybrid_loss(W_mv, xs[0], n_out, n_in)
    W_mv = hybrid_step(W_mv, xs[0], eta, n_out, n_in)
    loss_after = hybrid_loss(W_mv, xs[0], n_out, n_in)
    print(f"Loss before: {loss_before:.6f}, after: {loss_after:.6f}")
    assert loss_after != loss_before, "Loss should change after an update"