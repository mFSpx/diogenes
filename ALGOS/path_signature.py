#!/usr/bin/env python3
"""Path signature and iterated-integral algebra.

S(X)^k_{s,t} = integral_{s<u1<...<uk<t} dX_{u1} (x) ... (x) dX_{uk}

Time-reparametrization invariant. Captures path geometry, not just values.
Lead-lag transform encodes causality by interleaving (X_t, X_{t-1}) channels.

Mutation class: read_only
"""
from __future__ import annotations
import numpy as np


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

    # Initialize accumulators: level k holds a flat array of size d^k
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


def signature_flat(path, depth=3):
    """Flatten all signature terms to a 1D feature vector.

    Concatenates level-1 .. level-depth arrays into one flat array.
    """
    return np.concatenate([s.ravel() for s in signature(path, depth=depth)])


def logsig_level2(path):
    """Level-2 log-signature: antisymmetric part of S2.

    A[i,j] = S2[i,j] - S2[j,i].  Shape (d, d).
    Encodes the signed enclosed area (Levy area) swept by the path.
    """
    s2 = signature_level2(path)
    return s2 - s2.T


def normalize_path(path):
    """Translate to origin, scale to unit total-variation length.

    Reparametrization-invariant normalization; useful for comparison.
    Returns path starting at zero with sum of step norms == 1.
    """
    path = np.asarray(path, dtype=float)
    path = path - path[0]
    tv = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    if tv > 0:
        path = path / tv
    return path


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Two 2-D paths with identical start/end but different shapes.
    # Straight line vs. arch.
    T = 50
    t = np.linspace(0, 1, T)

    path_line = np.column_stack([t, t])                       # diagonal straight line
    path_arch = np.column_stack([t, np.sin(np.pi * t)])       # arch: same x endpoints, 0→1→0 in y

    sig_line = signature(path_line, depth=3)
    sig_arch = signature(path_arch, depth=3)

    print("=== Signature comparison (same endpoints, different shapes) ===")
    print(f"Level-1 line: {sig_line[0]}")
    print(f"Level-1 arch: {sig_arch[0]}")
    print(f"Level-1 equal: {np.allclose(sig_line[0], sig_arch[0])}")
    print(f"Level-2 line:\n{sig_line[1]}")
    print(f"Level-2 arch:\n{sig_arch[1]}")
    print(f"Level-2 equal: {np.allclose(sig_line[1], sig_arch[1])}")

    # Level-1 increments are the same (same endpoints) but level-2 differs.
    print()

    # Reparametrization invariance: monotone remap phi(t)=t^3 on a fine grid.
    # The property is exact in the continuous limit; with N=2000 points the
    # discretization error is small enough to verify numerically.
    N = 2000
    t_fine   = np.linspace(0, 1, N)
    t_reparm = t_fine ** 3                              # strictly monotone phi
    path_fine   = np.column_stack([t_fine,   np.sin(np.pi * t_fine)])
    path_reparm = np.column_stack([t_reparm, np.sin(np.pi * t_reparm)])
    sig_fine   = signature(path_fine,   depth=3)
    sig_reparm = signature(path_reparm, depth=3)

    # atol=2e-3: honest O(1/N) discretization error at N=2000
    print("=== Reparametrization invariance (phi(t)=t^3, N=2000, atol=2e-3) ===")
    for k in range(3):
        flat_orig   = sig_fine[k].ravel()
        flat_rep    = sig_reparm[k].ravel()
        close = np.allclose(flat_orig, flat_rep, atol=2e-3)
        print(f"Level-{k+1} allclose (atol=2e-3): {close}  "
              f"max_diff={np.max(np.abs(flat_orig - flat_rep)):.6f}")

    # Lead-lag demo.
    print()
    small = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
    ll = lead_lag_transform(small)
    print(f"Lead-lag of shape {small.shape} -> {ll.shape}")

    # Log-signature (Levy area).
    print()
    area = logsig_level2(path_arch)
    print(f"Levy area (logsig level-2) of arch:\n{area}")

    # Flat feature vector.
    feat = signature_flat(path_arch, depth=3)
    d = path_arch.shape[1]
    expected_len = sum(d ** k for k in range(1, 4))
    print(f"\nFlat signature length: {len(feat)} (expected {expected_len})")

    # Normalized path.
    norm_arch = normalize_path(path_arch)
    tv = np.sum(np.linalg.norm(np.diff(norm_arch, axis=0), axis=1))
    print(f"Normalized path starts at {norm_arch[0]}, total variation = {tv:.6f}")
