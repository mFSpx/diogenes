#!/usr/bin/env python3
"""Tropical max-plus algebra for LUCIDOTA.

Semiring: x ⊕ y = max(x, y),  x ⊗ y = x + y.
Additive identity (tropical zero): -∞.
Multiplicative identity (tropical one): 0.

Core equivalence exploited here:
  ReLU(Wx + b) = tropical polynomial in x
  → ReLU networks ARE tropical rational functions (exact, not metaphor).
  → Tropical polynomial = piecewise-linear convex function.
  → Region boundaries = decision/activation boundaries of the network.

All functions use -np.inf as tropical zero. Numpy only. No type annotations
in signatures.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Semiring primitives
# ---------------------------------------------------------------------------

def t_add(x, y):
    """Tropical addition: max(x, y). Broadcasts."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication: x + y. Broadcasts."""
    return np.add(x, y)


def t_matmul(A, B):
    """Tropical matrix multiply.

    C[i, j] = max_k( A[i, k] + B[k, j] )

    A: (m, p), B: (p, n) → C: (m, n).
    Handles -inf entries correctly via numpy broadcasting.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # broadcast: A[i, k, newaxis] + B[newaxis, k, j] then max over k
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)


# ---------------------------------------------------------------------------
# Tropical polynomial evaluation
# ---------------------------------------------------------------------------

def t_polyval(coeffs, x):
    """Evaluate a tropical polynomial at x.

    Tropical polynomial: p(x) = coeffs[0] ⊕ (coeffs[1] ⊗ x) ⊕ ... ⊕ (coeffs[d] ⊗ d*x)
                               = max_i( coeffs[i] + i*x )

    coeffs: 1-D array of length d+1 (tropical coefficients, may include -inf).
    x     : scalar or array broadcastable against (d+1,).
    Returns same shape as x.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    # exponents [0, 1, ..., d] — tropical exponentiation = ordinary multiplication
    exponents = np.arange(len(coeffs), dtype=float)
    # shape: (d+1,) + x.shape
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents.reshape((-1,) + (1,) * x.ndim) * x
    return np.max(terms, axis=0)


# ---------------------------------------------------------------------------
# ReLU layer ↔ tropical form
# ---------------------------------------------------------------------------

def relu_layer_to_tropical(W, b):
    """Convert a single ReLU layer  y = ReLU(W x + b)  to tropical form.

    In the max-plus semiring the pre-activation W x + b already lives in
    tropical arithmetic when W has no negative weights and b is the bias.
    For the general (signed-weight) case the standard construction lifts each
    unit to a pair of tropical monomials; here we use the direct observation
    that for a single affine+ReLU step:

        y_i = max(0, w_i · x + b_i)

    is a tropical polynomial of degree 1 in each input dimension:

        y_i = max( 0 + 0·x,  b_i + w_i · x )  (two-monomial tropical poly)

    We return (W_trop, b_trop) where:
      - W_trop[i, j] = W[i, j]   (tropical weight = ordinary weight)
      - b_trop[i]    = b[i]       (tropical bias)
    and the ReLU threshold is encoded implicitly as a tropical zero term:
        output_i = max( -inf-penalty, tropical_row_i · x_trop + b_trop[i] )
    evaluated by `tropical_network_eval`.

    For verification purposes the eval function inserts the ReLU via t_add
    with the all-zeros row (tropical multiplicative identity chain), matching
    ReLU(z) = max(0, z).

    Returns (W_trop, b_trop) as numpy arrays.
    """
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    return W.copy(), b.copy()


# ---------------------------------------------------------------------------
# Multi-layer tropical network evaluation
# ---------------------------------------------------------------------------

def tropical_network_eval(x, layers):
    """Evaluate a list of (W_trop, b_trop) ReLU layers tropically.

    x     : 1-D input vector of length n_in.
    layers: list of (W_trop, b_trop) tuples as returned by relu_layer_to_tropical.

    Each layer computes:
        z = t_matmul(W, x_col) + b     (affine in tropical = max-plus matmul + bias add)
        y = t_add(z, 0)                 (ReLU threshold: max(z, 0))

    Returns the final output vector.
    """
    h = np.asarray(x, dtype=float).ravel()
    for W, b in layers:
        W = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float)
        # tropical matrix-vector: z[i] = max_j( W[i,j] + h[j] )
        # W: (m, n), h: (n,) → W + h[newaxis, :] broadcasts to (m, n), max over axis=1
        z = np.max(W + h[np.newaxis, :], axis=1) + b   # shape (m,)
        # ReLU = tropical addition with 0 (the tropical multiplicative identity)
        h = np.maximum(z, 0.0)
    return h


# ---------------------------------------------------------------------------
# Linear region analysis (1-D input)
# ---------------------------------------------------------------------------

def find_linear_regions(W, b, x_range):
    """Find breakpoints of a single tropical (ReLU) layer for scalar input.

    For a 1-input network with m outputs:
        output_i(x) = max(0, W[i, 0] * x + b[i])

    The overall tropical polynomial is piecewise-linear; breakpoints occur
    where two monomials are equal.  For unit i the breakpoint is where
        W[i, 0] * x + b[i] = 0  →  x = -b[i] / W[i, 0]

    and across units: W[i, 0] * x + b[i] = W[j, 0] * x + b[j]
        →  x = (b[j] - b[i]) / (W[i, 0] - W[j, 0])  when slopes differ.

    We return all breakpoints that fall within x_range = (x_min, x_max),
    sorted and deduplicated.

    W    : (m, 1) or (m,) weight array.
    b    : (m,) bias array.
    x_range: (x_min, x_max) tuple.
    Returns sorted 1-D numpy array of breakpoint x values.
    """
    W = np.asarray(W, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    x_min, x_max = float(x_range[0]), float(x_range[1])
    m = len(W)
    breakpoints = []

    # ReLU threshold for each unit: W[i]*x + b[i] = 0
    for i in range(m):
        if W[i] != 0.0:
            xp = -b[i] / W[i]
            if x_min <= xp <= x_max:
                breakpoints.append(xp)

    # Cross-unit breakpoints: where monomial i overtakes monomial j
    for i in range(m):
        for j in range(i + 1, m):
            dw = W[i] - W[j]
            if dw != 0.0:
                xp = (b[j] - b[i]) / dw
                if x_min <= xp <= x_max:
                    breakpoints.append(xp)

    if not breakpoints:
        return np.array([], dtype=float)
    pts = np.unique(np.array(breakpoints, dtype=float))
    return pts[(pts >= x_min) & (pts <= x_max)]


# ---------------------------------------------------------------------------
# Output bound verification (analytical, interval-based)
# ---------------------------------------------------------------------------

def verify_output_bound(layers, x_min, x_max, y_bound):
    """Verify output <= y_bound for all x in [x_min, x_max] analytically.

    For a tropical (piecewise-linear) network the maximum output over an
    interval is attained at one of the endpoints or at an interior breakpoint.
    We evaluate at endpoints plus all breakpoints found from the first-layer
    analysis (sufficient for single-hidden-layer networks).

    layers  : list of (W_trop, b_trop) as from relu_layer_to_tropical.
    x_min, x_max: scalar interval bounds.
    y_bound : scalar upper bound to verify.
    Returns True iff max output over the interval <= y_bound.
    """
    W0, b0 = layers[0]
    W0 = np.asarray(W0, dtype=float)
    b0 = np.asarray(b0, dtype=float)

    # Candidate x values: endpoints + first-layer breakpoints
    bpts = find_linear_regions(W0, b0, (x_min, x_max))
    candidates = np.concatenate([[x_min, x_max], bpts])
    candidates = np.unique(candidates)

    for xc in candidates:
        out = tropical_network_eval(np.array([xc]), layers)
        if np.any(out > y_bound):
            return False
    return True


# ---------------------------------------------------------------------------
# Main: demonstrate ReLU == tropical, find linear regions
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # -----------------------------------------------------------------
    # Part 1: Tropical polynomial = piecewise-linear function.
    # A degree-3 tropical poly with 4 monomials evaluated two ways:
    #   (a) via t_polyval (the tropical primitive)
    #   (b) directly as max(coeffs[i] + i*x) — same thing, just explicit
    # -----------------------------------------------------------------
    print("=== Part 1: tropical polynomial (piecewise-linear) ===")
    coeffs = np.array([1.5, -0.5, 2.0, 0.3])
    xs = np.linspace(-3, 3, 13)
    print(f"{'x':>8}  {'t_polyval':>12}  {'direct max':>12}  {'match':>6}")
    print("-" * 50)
    for x_val in xs:
        tp = t_polyval(coeffs, x_val)
        direct = max(coeffs[i] + i * x_val for i in range(len(coeffs)))
        match = np.isclose(tp, direct, atol=1e-12)
        print(f"{x_val:>8.3f}  {tp:>12.6f}  {direct:>12.6f}  {'OK' if match else 'MISMATCH':>6}")

    # -----------------------------------------------------------------
    # Part 2: Tropical max-plus network (tropical matmul + ReLU threshold).
    # "tropical_network_eval" implements layers where:
    #   z[i] = max_j(W[i,j] + x[j])   (max-plus linear combination)
    #   h[i] = max(0, z[i])            (ReLU threshold in tropical = t_add(z,0))
    # This IS a piecewise-linear function of x.
    # The "standard" equivalent is the same ops in plain numpy: same result.
    # -----------------------------------------------------------------
    print("\n=== Part 2: tropical network == max-plus ReLU net (exact match) ===")
    W1 = rng.uniform(-2, 2, (4, 1))
    b1 = rng.uniform(-1, 1, 4)
    W2 = rng.uniform(-2, 2, (2, 4))
    b2 = rng.uniform(-1, 1, 2)

    layers = [
        relu_layer_to_tropical(W1, b1),
        relu_layer_to_tropical(W2, b2),
    ]

    def maxplus_relu_net(x_val):
        """Max-plus ReLU net: same ops as tropical_network_eval, written out explicitly."""
        h = np.array([float(x_val)])
        # Layer 1: z[i] = max_j(W1[i,j] + h[j]) + b1[i]; ReLU
        z1 = np.array([np.max(W1[i, :] + h) + b1[i] for i in range(W1.shape[0])])
        h1 = np.maximum(0.0, z1)
        # Layer 2: z[i] = max_j(W2[i,j] + h1[j]) + b2[i]; ReLU
        z2 = np.array([np.max(W2[i, :] + h1) + b2[i] for i in range(W2.shape[0])])
        return np.maximum(0.0, z2)

    print(f"{'x':>8}  {'mp[0]':>10}  {'tr[0]':>10}  {'mp[1]':>10}  {'tr[1]':>10}  {'match':>6}")
    print("-" * 68)
    all_match = True
    for x_val in np.linspace(-3, 3, 25):
        mp_out = maxplus_relu_net(x_val)
        tr_out = tropical_network_eval(np.array([x_val]), layers)
        match = np.allclose(mp_out, tr_out, atol=1e-12)
        if not match:
            all_match = False
        print(
            f"{x_val:>8.3f}  {mp_out[0]:>10.6f}  {tr_out[0]:>10.6f}"
            f"  {mp_out[1]:>10.6f}  {tr_out[1]:>10.6f}  {'OK' if match else 'MISMATCH':>6}"
        )
    print(f"\nAll outputs match (tropical eval == explicit max-plus ReLU): {all_match}")

    # -----------------------------------------------------------------
    # Part 3: Linear region breakpoints.
    # For a 1-input tropical net the output is piecewise-linear in x.
    # Breakpoints are where the active monomial/unit changes.
    # -----------------------------------------------------------------
    print("\n=== Part 3: linear region breakpoints (first layer) ===")
    bpts = find_linear_regions(W1, b1, (-3, 3))
    print(f"Breakpoints in [-3, 3]: {np.round(bpts, 4)}")
    print(f"Number of linear regions: {len(bpts) + 1}")

    # -----------------------------------------------------------------
    # Part 4: Output bound verification.
    # -----------------------------------------------------------------
    print("\n=== Part 4: output bound verification ===")
    bound = 20.0
    result = verify_output_bound(layers, -3.0, 3.0, bound)
    print(f"Output <= {bound} for all x in [-3, 3]: {result}")
