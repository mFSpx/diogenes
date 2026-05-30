#!/usr/bin/env python3
"""Kolmogorov-Arnold Networks (KAN) — clean numpy reference implementation.

Theory
------
The Kolmogorov-Arnold representation theorem (1957) states that any continuous
multivariate function f: [0,1]^n → R can be written as

    f(x) = Σ_{q=1}^{2n+1} Φ_q( Σ_{p=1}^n ϕ_{q,p}(x_p) )

where Φ_q and ϕ_{q,p} are continuous univariate functions.  KANs (Liu et al.,
2024) operationalise this as a deep network where *every edge* carries a
learnable univariate function (a B-spline) rather than a fixed activation on a
node.

Deep KAN composition:
    KAN(x) = Φ_{L-1} ∘ ... ∘ Φ_1 ∘ Φ_0 ∘ x

Each Φ_l maps R^{n_in} → R^{n_out} by computing, for each output unit q and
input unit p, a B-spline ϕ_{q,p}(x_p), then summing over p.

B-spline basis
--------------
Given a knot grid t_0 < t_1 < ... < t_m and order k (degree k-1), the
Cox-de Boor recursion yields n_basis = m - k basis functions.  Here we use
uniform grids extended with k repeated boundary knots so that the basis
spans the full input range cleanly.

Mutation class: read_only (no side effects, pure computation).
"""
from __future__ import annotations

import numpy as np

__all__ = ["bspline_basis", "kan_layer", "kan_forward", "init_kan_layer"]


# ---------------------------------------------------------------------------
# B-spline basis
# ---------------------------------------------------------------------------

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Evaluate B-spline basis functions of order k at positions x.

    Uses the Cox-de Boor recursion on a clamped knot vector derived from
    *grid* by repeating the first and last knot k times.

    Parameters
    ----------
    x:    shape (N,) — evaluation points (should lie within grid range).
    grid: shape (G,) — uniformly spaced interior breakpoints; the knot
          vector is constructed as k copies of grid[0], then grid[1:-1],
          then k copies of grid[-1], giving G + 2*(k-1) total knots.
    k:    spline order (polynomial degree = k - 1).  Default 3 (cubic).

    Returns
    -------
    B: shape (N, G - 1) — one column per basis function.
       Number of basis functions = len(grid) - 1.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Build clamped knot vector: repeat boundary knots (k-1) times so that
    # the polynomial spans cleanly to the boundary.
    # Knot vector length: (k-1) + G + (k-1) = G + 2(k-1)
    # After Cox-de Boor of order k the number of basis functions is:
    #   len(t) - k = G + 2(k-1) - k = G + k - 2
    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2      # number of B-splines
    N = len(x)

    # Order-1 basis (step functions): B_{i,1}(x) = 1 iff t[i] <= x < t[i+1]
    n_knots = len(t)
    # B has shape (N, n_knots - 1) at order 1, shrinks to n_basis at order k
    B = np.zeros((N, n_knots - 1), dtype=np.float64)
    for i in range(n_knots - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    # Handle right endpoint: last basis function includes x == t[-1]
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    # Cox-de Boor recursion up to order k
    for order in range(2, k + 1):
        B_new = np.zeros((N, n_knots - order), dtype=np.float64)
        for i in range(n_knots - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (
                (x - t[i]) / denom_l * B[:, i]
                if denom_l > 0 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 0 else np.zeros(N)
            )
            B_new[:, i] = term_l + term_r
        B = B_new

    # B now has shape (N, n_knots - k) = (N, len(grid) + k - 1 - k) = (N, len(grid) - 1)
    assert B.shape == (N, n_basis), (
        f"basis shape mismatch: got {B.shape}, expected ({N}, {n_basis})"
    )
    return B


# ---------------------------------------------------------------------------
# Single KAN layer
# ---------------------------------------------------------------------------

def kan_layer(
    x: np.ndarray,
    spline_weights: np.ndarray,
    grid: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """Apply one KAN layer.

    For each output unit q and input unit p, the edge function is
        ϕ_{q,p}(x_p) = Σ_b  w_{q,p,b} · B_b(x_p)
    The layer output for unit q is Σ_p ϕ_{q,p}(x_p).

    Parameters
    ----------
    x:              shape (batch, n_in)
    spline_weights: shape (n_out, n_in, n_basis) where n_basis = len(grid) - 1
    grid:           shape (G,) — shared knot breakpoints for all edges
    k:              spline order

    Returns
    -------
    out: shape (batch, n_out)
    """
    x = np.asarray(x, dtype=np.float64)
    spline_weights = np.asarray(spline_weights, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    batch, n_in = x.shape
    n_out, n_in_w, n_basis = spline_weights.shape
    assert n_in == n_in_w, f"n_in mismatch: x has {n_in}, weights expect {n_in_w}"
    expected_n_basis = len(grid) + k - 2
    assert n_basis == expected_n_basis, (
        f"n_basis mismatch: weights have {n_basis}, grid+k gives {expected_n_basis}"
    )

    out = np.zeros((batch, n_out), dtype=np.float64)

    for p in range(n_in):
        # B shape: (batch, n_basis)
        B = bspline_basis(x[:, p], grid, k)
        # spline_weights[:, p, :] shape: (n_out, n_basis)
        # contribution: (batch, n_basis) @ (n_basis, n_out) → (batch, n_out)
        out += B @ spline_weights[:, p, :].T

    return out


# ---------------------------------------------------------------------------
# Deep KAN forward pass
# ---------------------------------------------------------------------------

def kan_forward(x: np.ndarray, layers: list[dict]) -> np.ndarray:
    """Compose L KAN layers: KAN(x) = Φ_{L-1} ∘ ... ∘ Φ_0 ∘ x.

    Parameters
    ----------
    x:      shape (batch, n_in) — input to the first layer
    layers: list of layer dicts, each with keys:
              "spline_weights": np.ndarray shape (n_out, n_in, n_basis)
              "grid":           np.ndarray shape (G,)
              "k":              int (spline order, default 3)

    Returns
    -------
    out: shape (batch, n_out_final)
    """
    h = np.asarray(x, dtype=np.float64)
    for layer in layers:
        h = kan_layer(
            h,
            spline_weights=layer["spline_weights"],
            grid=layer["grid"],
            k=layer.get("k", 3),
        )
    return h


# ---------------------------------------------------------------------------
# Layer initialiser
# ---------------------------------------------------------------------------

def init_kan_layer(
    n_in: int,
    n_out: int,
    grid_size: int = 5,
    k: int = 3,
    grid_range: tuple = (-1.0, 1.0),
) -> dict:
    """Initialise a KAN layer with zero spline weights and a uniform grid.

    Parameters
    ----------
    n_in:       number of input features
    n_out:      number of output features
    grid_size:  number of interior breakpoints (G); n_basis = G - 1
    k:          spline order
    grid_range: (lo, hi) for the uniform breakpoint grid

    Returns
    -------
    dict with keys "spline_weights", "grid", "k"
    """
    lo, hi = grid_range
    grid = np.linspace(lo, hi, grid_size)
    n_basis = grid_size + k - 2      # matches bspline_basis output width
    spline_weights = np.zeros((n_out, n_in, n_basis), dtype=np.float64)
    return {"spline_weights": spline_weights, "grid": grid, "k": k}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    batch, n0, n1, n2 = 8, 4, 6, 3

    layer0 = init_kan_layer(n0, n1, grid_size=7, k=3, grid_range=(-1.0, 1.0))
    layer1 = init_kan_layer(n1, n2, grid_size=7, k=3, grid_range=(-1.0, 1.0))

    # Non-zero weights so output is non-trivially shaped
    layer0["spline_weights"] = rng.standard_normal(layer0["spline_weights"].shape) * 0.1
    layer1["spline_weights"] = rng.standard_normal(layer1["spline_weights"].shape) * 0.1

    x = rng.uniform(-1.0, 1.0, (batch, n0))

    h0 = kan_layer(x, layer0["spline_weights"], layer0["grid"], layer0["k"])
    print(f"input shape      : {x.shape}")
    print(f"after layer 0    : {h0.shape}")

    out = kan_forward(x, [layer0, layer1])
    print(f"after layer 1    : {out.shape}")
    print(f"output sample[0] : {out[0]}")

    # Verify bspline_basis shapes
    g = np.linspace(-1, 1, 7)
    B = bspline_basis(x[:, 0], g, k=3)
    print(f"bspline_basis    : {B.shape}  (expect ({batch}, {len(g)+3-2}))")

    print("smoke test OK")
