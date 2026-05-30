# DARWIN HAMMER — match 30, survivor 2
# gen: 1
# parent_a: path_signature.py (gen0)
# parent_b: kan.py (gen0)
# born: 2026-05-29T23:23:27Z

import numpy as np
import math
import random
import sys
import pathlib

def lead_lag_transform(path):
    """Lead-lag transform: interleave (lead, lag) channels for causality encoding.

    path: (T, d). Returns (2T-1, 2d) interleaved lead-lag path.

    At even indices 2t   : (X_t,   X_t)    (lead and lag both at t)
    At odd  indices 2t+1 : (X_{t+1}, X_t)  (lead advances, lag holds)
    This matches the standard Chevyrev-Kormilitzin convention.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def signature_level1(path):
    """Level-1 signature: total increment vector.

    path: (T, d). Returns (d,). Equal to path[-1] - path[0].
    """
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def signature_level2(path):
    """Level-2 iterated integral tensor.

    path: (T, d). Returns (d, d).
    S2[i,j] = sum_{s<t} (X_s[i] - X_0[i]) * dX_t[j]
             = sum_{t=1..T-1} (X_{t-1}[i] - X_0[i]) * (X_t[j] - X_{t-1}[j])

    Uses the standard left-point Riemann sum on the increment path.
    """
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    # S2[i,j] = sum_t running[t,i] * increments[t,j]
    return running.T @ increments               # (d, d)


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


def signature(path, depth=3):
    """Signature up to given depth.

    Returns list of k arrays for k=1..depth with shapes (d,), (d,d), ..., (d^k,).
    Uses iterative Chen-like accumulation (not recursive):
      S_k[i1,...,ik] += S_{k-1}[i1,...,i_{k-1}] * dX_t[ik]
    for each time step t.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    increments = np.diff(path, axis=0)          # (T-1, d)

    # Initialize accumulators: level k holds a flat numpy array of size d^k
    # We keep them as flat numpy arrays and rebuild shape on output.
    S = [np.zeros(d ** k) for k in range(1, depth + 1)]

    for t in range(T - 1):
        dx = increments[t]                       # (d,)
        # Update from highest to lowest to avoid overwriting lower levels mid-step
        for k in range(depth - 1, 0, -1):
            # S[k] shape (d^{k+1},): outer product of S[k-1] and dx
            S[k] = S[k] + np.outer(S[k - 1].reshape(-1), dx).ravel()
        # Level 1: just accumulate increments
        S[0] = S[0] + dx

    return [S[k].reshape((d,) * (k + 1)) for k in range(depth)]


def kan_signature(path, grid, spline_weights, k=3):
    """Hybrid KAN signature.

    Uses the KAN to approximate the level-1 and level-2 iterated-integrals.

    Parameters
    ----------
    path: shape (T, d)
    grid: shape (G,) — uniformly spaced interior breakpoints
    spline_weights: shape (n_out, n_in, n_basis) 
    k: spline order

    Returns
    -------
    S: list of signature levels
    """
    T, d = path.shape
    S = [np.zeros(d) for _ in range(2)]

    # Level-1 signature
    S[0] = kan_layer(path[:, 0].reshape(-1, 1), spline_weights[0].reshape(1, 1, -1), grid, k).ravel()

    # Level-2 signature
    increments = np.diff(path, axis=0)
    running = path[:-1] - path[0]
    spline_weights_2 = spline_weights[1].reshape(d, d, -1)
    S[1] = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            S[1][i, j] = np.sum(kan_layer(running[:, i].reshape(-1, 1), spline_weights_2[i, j].reshape(1, 1, -1), grid, k) * increments[:, j])

    # Higher-order signatures
    for level in range(2, 3):
        S.append(np.zeros(tuple([d] * (level + 1))))

    return S


def improved_kan_signature(path, grid, spline_weights, k=3):
    T, d = path.shape
    S = [np.zeros(d) for _ in range(3)]

    # Level-1 signature
    S[0] = kan_layer(path[:, 0].reshape(-1, 1), spline_weights[0].reshape(1, 1, -1), grid, k).ravel()

    # Level-2 signature
    increments = np.diff(path, axis=0)
    running = path[:-1] - path[0]
    spline_weights_2 = spline_weights[1].reshape(d, d, -1)
    S[1] = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            S[1][i, j] = np.sum(kan_layer(running[:, i].reshape(-1, 1), spline_weights_2[i, j].reshape(1, 1, -1), grid, k) * increments[:, j])

    # Level-3 signature
    spline_weights_3 = spline_weights[2].reshape(d**2, d, -1)
    S[2] = np.zeros((d, d, d))
    for i in range(d):
        for j in range(d):
            for k_idx in range(d):
                S[2][i, j, k_idx] = np.sum(kan_layer(increments[:, i].reshape(-1, 1) * running[:, j].reshape(-1, 1), spline_weights_3[i*j, k_idx].reshape(1, 1, -1), grid, k) * increments[:, k_idx])

    return S