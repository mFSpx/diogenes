# DARWIN HAMMER — match 56, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_workshare_allocator_m60_s0.py (gen2)
# parent_b: hybrid_path_signature_kan_m30_s0.py (gen1)
# born: 2026-05-29T23:25:30Z

import numpy as np
import math
import random
import sys
import pathlib

def lead_lag_transform(path):
    """
    Lead-lag transform: interleave (lead, lag) channels for causality encoding.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(x, grid, k=3):
    """
    Evaluate B-spline basis functions of order k at positions x.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
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

    assert B.shape == (N, n_basis), (
        f"basis shape mismatch: got {B.shape}, expected ({N}, {n_basis})"
    )

    return B

def hybrid_banded_path_signature(x, grid, k=3, banded_width=5):
    """
    Hybrid banded path signature computation using B-spline basis functions.
    """
    banded_x = np.roll(x, -banded_width, axis=0)
    banded_x[banded_x == 0.0] = np.nan

    B = bspline_basis(banded_x, grid, k=k)

    # Weighted sum of path signature terms with B-spline coefficients
    return np.sum(B * banded_x, axis=1)

def store_banded_dance(x, grid, k=3, banded_width=5, store_alpha=0.1):
    """
    Store state dynamics incorporating banded path signature and B-spline basis.
    """
    dance = hybrid_banded_path_signature(x, grid, k=k, banded_width=banded_width)
    store_level, _ = StoreState().update(inflow=[dance], outflow=[1.0])
    return store_level

def bspline_store_dance(delta, grid, k=3, store_alpha=0.1, store_beta=0.1, banded_width=5):
    """
    Store state dynamics incorporating B-spline basis and path signature terms.
    """
    dance = np.sum(bspline_basis(delta, grid, k=k) * delta)
    store_level, _ = StoreState().update(inflow=[dance], outflow=[1.0])
    return store_level

class StoreState:
    """
    Encapsulates the honeybee-style store and its derived control signal.
    """
    def __init__(self, level=0.0, alpha=1.0, beta=1.0, dt=1.0, base=1.0, gain=1.0, limit=10.0):
        self.level = level
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.base = base
        self.gain = gain
        self.limit = limit

    def update(self, inflow, outflow):
        """
        Apply the store equation and recompute the dance duration.
        """
        delta = self.alpha * inflow[0] - self.beta * outflow[0]
        self.level = max(0.0, self.level + self.dt * delta)
        return self.level, delta

    @property
    def dance(self):
        """
        Bounded control signal derived from the last Δ (computed lazily).
        """
        # The most recent Δ is stored temporarily in ``_last_delta`` by ``update``.
        # If ``update`` has not been called, _last_delta is nan.
        # This line would normally be replaced by the actual _last_delta value.
        # For simplicity, we assume that _last_delta is always available and set to the last delta.
        last_delta = self._last_delta
        self._last_delta = delta  # Store the last delta value for future dance calculation
        # Compute the dance signal
        if last_delta is None:
            # Return a default value if last_delta is None
            last_delta = 0.0
        dance_signal = self.base + self.gain * np.clip(last_delta, -self.limit, self.limit)
        return dance_signal

if __name__ == "__main__":
    import numpy as np

    # Create some dummy data to test the hybrid functions
    x = np.random.rand(10, 10)
    grid = np.linspace(0, 10, 100)
    k = 3
    banded_width = 5

    # Test hybrid_banded_path_signature
    dance = hybrid_banded_path_signature(x, grid, k=k, banded_width=banded_width)
    assert np.all(np.isfinite(dance))

    # Test store_banded_dance
    store_level = store_banded_dance(x, grid, k=k, banded_width=banded_width, store_alpha=0.1)
    assert np.all(np.isfinite(store_level))

    # Test bspline_store_dance
    delta = np.random.rand(10)
    store_level = bspline_store_dance(delta, grid, k=k, store_alpha=0.1, store_beta=0.1, banded_width=banded_width)
    assert np.all(np.isfinite(store_level))