# DARWIN HAMMER — match 30, survivor 3
# gen: 1
# parent_a: path_signature.py (gen0)
# parent_b: kan.py (gen0)
# born: 2026-05-29T23:23:27Z

import numpy as np

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def signature_level1(path):
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def signature_level2(path):
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    return running.T @ increments               # (d, d)


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
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

    return B


def kan_layer(
    x: np.ndarray,
    spline_weights: np.ndarray,
    grid: np.ndarray,
    k: int = 3,
) -> np.ndarray:
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
        B = bspline_basis(x[:, p], grid, k)
        out += B @ spline_weights[:, p, :].T

    return out


def signature(path, depth=3):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    increments = np.diff(path, axis=0)          # (T-1, d)

    S = [np.zeros(d ** k) for k in range(1, depth + 1)]

    for t in range(T - 1):
        dx = increments[t]                       
        for k in range(depth - 1, 0, -1):
            S[k] = S[k] + np.outer(S[k - 1].reshape(-1), dx).ravel()
        S[0] = S[0] + dx

    return [S[k].reshape((d,) * (k + 1)) for k in range(depth)]


def kan_signature(path, grid, spline_weights, k=3):
    path = np.asarray(path, dtype=float)
    T, d = path.shape

    # Compute level-1 signature using KAN
    level1 = kan_layer(path, spline_weights[:, :d, :], grid, k)

    # Compute level-2 signature using KAN
    level2 = np.zeros((T, d, d))
    for t in range(T - 1):
        dx = path[t + 1] - path[t]
        level2[t] = np.outer(level1[t], dx)

    return [level1[-1], level2[-1]]


def improved_kan_signature(path, grid, spline_weights, k=3):
    path = np.asarray(path, dtype=float)
    T, d = path.shape

    # Compute level-1 signature using KAN
    level1 = kan_layer(path, spline_weights[:, :d, :], grid, k)

    # Compute level-2 signature using KAN
    level2 = np.zeros((T, d, d))
    for t in range(T - 1):
        dx = path[t + 1] - path[t]
        level2[t] = np.outer(level1[t], dx)

    # Compute higher-level signatures using KAN
    higher_levels = []
    for level in range(3, 4):  # increase depth as needed
        higher_level = np.zeros((T, d ** level))
        for t in range(T - 1):
            dx = path[t + 1] - path[t]
            higher_level[t] = np.outer(higher_level[t - 1], dx).ravel()
        higher_levels.append(higher_level[-1])

    return [level1[-1], level2[-1]] + higher_levels


# Example usage
path = np.random.rand(10, 3)  # random path
grid = np.linspace(0, 1, 10)  # grid for KAN
spline_weights = np.random.rand(3, 3, len(grid) + 3 - 2)  # random spline weights

improved_signature = improved_kan_signature(path, grid, spline_weights)
print(improved_signature)