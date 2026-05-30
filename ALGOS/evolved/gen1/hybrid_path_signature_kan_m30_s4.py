# DARWIN HAMMER — match 30, survivor 4
# gen: 1
# parent_a: path_signature.py (gen0)
# parent_b: kan.py (gen0)
# born: 2026-05-29T23:23:27Z

import numpy as np
from typing import List, Tuple, Callable, Optional


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform of a path.

    Parameters
    ----------
    path : np.ndarray
        Original path of shape (T, d).

    Returns
    -------
    np.ndarray
        Transformed path of shape (2*T-1, 2*d) where even rows contain
        (X_t, X_t) and odd rows contain (X_{t+1}, X_t).
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (T, d)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)

    # even rows
    out[0::2, :d] = path
    out[0::2, d:] = path

    # odd rows (except the last one which does not exist)
    out[1::2, :d] = path[1:]
    out[1::2, d:] = path[:-1]

    return out


def signature_level1(path: np.ndarray) -> np.ndarray:
    """Exact level‑1 signature (total increment)."""
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def signature_level2(path: np.ndarray) -> np.ndarray:
    """Exact level‑2 signature using left‑point Riemann sums."""
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running = path[:-1] - path[0]               # (T‑1, d)
    return running.T @ increments               # (d, d)


def _make_clamped_knots(grid: np.ndarray, order: int) -> np.ndarray:
    """
    Build a clamped knot vector for B‑splines of given order.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError("grid must be 1‑D")
    if order < 1:
        raise ValueError("order must be >= 1")
    # repeat first and last knot (order) times for clamping
    left = np.full(order, grid[0])
    right = np.full(order, grid[-1])
    return np.concatenate([left, grid, right])


def bspline_basis(x: np.ndarray, grid: np.ndarray, order: int = 3) -> np.ndarray:
    """
    Evaluate all B‑spline basis functions of a given order at points x.

    Parameters
    ----------
    x : np.ndarray
        Evaluation points, shape (N,).
    grid : np.ndarray
        Interior breakpoints, shape (G,). Must be sorted.
    order : int, optional
        Spline order (degree = order‑1). Default is 3 (cubic).

    Returns
    -------
    np.ndarray
        Basis matrix B of shape (N, n_basis) where
        n_basis = len(grid) + order - 1.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    grid = np.asarray(grid, dtype=float)
    if np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be strictly increasing")
    knots = _make_clamped_knots(grid, order)
    n_basis = len(grid) + order - 1
    N = x.shape[0]

    # order‑1 (piecewise constant) basis
    B = np.zeros((N, len(knots) - 1), dtype=float)
    for i in range(len(knots) - 1):
        left, right = knots[i], knots[i + 1]
        if left == right:
            continue
        mask = (x >= left) & (x < right)
        B[mask, i] = 1.0
    # include rightmost endpoint
    B[x == knots[-1], -1] = 1.0

    # Cox‑de Boor recursion
    for r in range(2, order + 1):
        n_cols = len(knots) - r
        B_new = np.zeros((N, n_cols), dtype=float)
        for i in range(n_cols):
            denom_l = knots[i + r - 1] - knots[i]
            denom_r = knots[i + r] - knots[i + 1]

            term_l = np.zeros(N, dtype=float)
            term_r = np.zeros(N, dtype=float)

            if denom_l > 0:
                term_l = (x - knots[i]) / denom_l * B[:, i]
            if denom_r > 0:
                term_r = (knots[i + r] - x) / denom_r * B[:, i + 1]

            B_new[:, i] = term_l + term_r
        B = B_new

    if B.shape[1] != n_basis:
        # Numerical edge case when grid has length 1
        B = np.resize(B, (N, n_basis))
    return B


def kan_layer(
    x: np.ndarray,
    spline_weights: np.ndarray,
    grid: np.ndarray,
    order: int = 3,
) -> np.ndarray:
    """
    Single KAN layer with spline‑based edge functions.

    Parameters
    ----------
    x : np.ndarray
        Input batch, shape (batch, n_in).
    spline_weights : np.ndarray
        Edge coefficients, shape (n_out, n_in, n_basis).
    grid : np.ndarray
        Knot positions, shape (G,).
    order : int, optional
        Spline order. Default 3.

    Returns
    -------
    np.ndarray
        Output batch, shape (batch, n_out).
    """
    x = np.asarray(x, dtype=float)
    spline_weights = np.asarray(spline_weights, dtype=float)
    grid = np.asarray(grid, dtype=float)

    batch, n_in = x.shape
    n_out, n_in_w, n_basis = spline_weights.shape
    if n_in != n_in_w:
        raise ValueError(f"Input dimension mismatch: {n_in} vs {n_in_w}")

    expected_basis = len(grid) + order - 1
    if n_basis != expected_basis:
        raise ValueError(
            f"Basis size mismatch: weights have {n_basis}, expected {expected_basis}"
        )

    out = np.zeros((batch, n_out), dtype=float)

    for p in range(n_in):
        B = bspline_basis(x[:, p], grid, order)          # (batch, n_basis)
        # (batch, n_basis) @ (n_basis, n_out) -> (batch, n_out)
        out += B @ spline_weights[:, p, :].T

    return out


def _flatten_tensor(tensor: np.ndarray) -> np.ndarray:
    """Flatten a multi‑dimensional tensor to a 1‑D vector."""
    return tensor.ravel()


def _unflatten_tensor(vec: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reshape a flat vector back to a tensor of given shape."""
    return vec.reshape(shape)


class KanSignature:
    """
    Deep integration of path signatures with Kolmogorov‑Arnold Networks.

    The model approximates each level of the signature by a cascade of
    KAN layers.  Level‑1 and level‑2 are learned directly; higher levels are
    constructed using Chen's identity with the learned lower levels.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        grid: np.ndarray,
        order: int = 3,
        rng: Optional[np.random.Generator] = None,
        init_scale: float = 0.1,
    ):
        """
        Parameters
        ----------
        dim : int
            Dimension of the input path.
        depth : int
            Highest signature level to approximate (>=1).
        grid : np.ndarray
            Knot positions for all spline edges (shared across layers).
        order : int, optional
            Spline order (default 3).
        rng : np.random.Generator, optional
            Random number generator for weight initialization.
        init_scale : float, optional
            Scale of the initial random weights.
        """
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.dim = dim
        self.depth = depth
        self.grid = np.asarray(grid, dtype=float)
        self.order = order
        self.n_basis = len(self.grid) + order - 1

        self.rng = rng if rng is not None else np.random.default_rng()
        self.init_scale = init_scale

        # weights[layer][out, in, basis]
        self.weights: List[np.ndarray] = []
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Create random spline weights for each level."""
        # Level‑1: map d inputs → d outputs (identity‑like)
        w1 = self.rng.normal(
            loc=0.0, scale=self.init_scale, size=(self.dim, self.dim, self.n_basis)
        )
        self.weights.append(w1)

        # Level‑2: map (d + d^2) inputs → d^2 outputs
        if self.depth >= 2:
            in_dim = self.dim + self.dim ** 2
            out_dim = self.dim ** 2
            w2 = self.rng.normal(
                loc=0.0, scale=self.init_scale, size=(out_dim, in_dim, self.n_basis)
            )
            self.weights.append(w2)

        # Higher levels: each level k receives input dim = d^k + d^{k-1}
        # and outputs dim = d^k
        for k in range(3, self.depth + 1):
            in_dim = self.dim ** k + self.dim ** (k - 1)
            out_dim = self.dim ** k
            wk = self.rng.normal(
                loc=0.0, scale=self.init_scale, size=(out_dim, in_dim, self.n_basis)
            )
            self.weights.append(wk)

    def _kan_forward(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """Apply KAN layer `layer_idx` to input batch x."""
        return kan_layer(
            x,
            self.weights[layer_idx],
            self.grid,
            order=self.order,
        )

    def compute(self, path: np.ndarray) -> List[np.ndarray]:
        """
        Compute the hybrid signature of a path.

        Parameters
        ----------
        path : np.ndarray
            Input path of shape (T, dim).

        Returns
        -------
        List[np.ndarray]
            Approximated signature levels 1 … depth.
            Level k has shape (dim,)*k.
        """
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != self.dim:
            raise ValueError(f"path must have shape (T, {self.dim})")

        # Exact level‑1 and level‑2 are used as seeds for the network.
        # They improve stability and give a sensible baseline.
        level_vals: List[np.ndarray] = [
            signature_level1(path),          # (dim,)
            signature_level2(path),          # (dim, dim)
        ]

        # If depth == 1 we stop here.
        if self.depth == 1:
            return [level_vals[0]]

        # Prepare batch for KAN: each level is processed as a single‑sample batch.
        # Level‑2 approximation uses concatenated level‑1 and exact level‑2.
        # Subsequent levels use concatenated previous approximation and exact lower level.
        approximations: List[np.ndarray] = []

        # ---- Level‑2 approximation ----
        inp_lvl2 = np.concatenate([level_vals[0].ravel(), level_vals[1].ravel()])[np.newaxis, :]
        out_lvl2 = self._kan_forward(inp_lvl2, layer_idx=1)          # shape (1, dim^2)
        approx_lvl2 = _unflatten_tensor(out_lvl2.ravel(), (self.dim, self.dim))
        approximations.append(approx_lvl2)

        # ---- Higher levels ----
        for k in range(3, self.depth + 1):
            prev_approx = approximations[-1].ravel()
            lower_exact = level_vals[k - 2].ravel()   # exact level k‑1
            inp = np.concatenate([lower_exact, prev_approx])[np.newaxis, :]
            out = self._kan_forward(inp, layer_idx=k - 1)   # layer index offset by 0‑based
            approx = _unflatten_tensor(out.ravel(), tuple([self.dim] * k))
            approximations.append(approx)

        # Assemble final list: replace exact level‑2 with its approximation,
        # keep level‑1 exact, and use approximations for higher levels.
        result = [level_vals[0]]
        if self.depth >= 2:
            result.append(approximations[0])
        if self.depth >= 3:
            result.extend(approximations[1:])

        return result

    def fit(
        self,
        paths: List[np.ndarray],
        targets: Optional[List[List[np.ndarray]]] = None,
        epochs: int = 200,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> None:
        """
        Simple gradient‑free training using finite‑difference stochastic
        approximation.  This keeps the implementation pure NumPy while still
        allowing the user to improve the spline weights on a dataset.

        Parameters
        ----------
        paths : list of np.ndarray
            Training paths, each of shape (T, dim).
        targets : list of list of np.ndarray, optional
            Exact signatures for each path up to `depth`.  If omitted they are
            computed on the fly.
        epochs : int
            Number of optimisation steps.
        lr : float
            Learning rate for the finite‑difference update.
        verbose : bool
            Print loss every 10 epochs.
        """
        if targets is None:
            targets = [self._exact_signature(p) for p in paths]

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for path, target_levels in zip(paths, targets):
                pred_levels = self.compute(path)
                loss = sum(
                    np.mean((p - t) ** 2)
                    for p, t in zip(pred_levels, target_levels[: self.depth])
                )
                total_loss += loss

                # Finite‑difference gradient estimate for each weight tensor
                for idx, w in enumerate(self.weights):
                    grad = np.zeros_like(w)
                    it = np.nditer(w, flags=["multi_index"], op_flags=["readwrite"])
                    while not it.finished:
                        i = it.multi_index
                        orig = w[i]
                        eps = 1e-6 * max(1.0, abs(orig))
                        w[i] = orig + eps
                        plus = self.compute(path)[idx]
                        w[i] = orig - eps
                        minus = self.compute(path)[idx]
                        w[i] = orig
                        # central difference
                        grad[i] = (np.mean((plus - target_levels[idx]) ** 2) -
                                   np.mean((minus - target_levels[idx]) ** 2)) / (2 * eps)
                        it.iternext()
                    self.weights[idx] -= lr * grad

            if verbose and (epoch % 10 == 0 or epoch == 1):
                avg_loss = total_loss / len(paths)
                print(f"Epoch {epoch:03d} – Avg MSE: {avg_loss:.6e}")

    def _exact_signature(self, path: np.ndarray) -> List[np.ndarray]:
        """Utility to compute exact signatures up to `depth`."""
        sig = [signature_level1(path)]
        if self.depth >= 2:
            sig.append(signature_level2(path))
        if self.depth >= 3:
            # Use the generic Chen‑like accumulation for higher levels
            sig.extend(signature(path, depth=self.depth)[2:])
        return sig


# Public API
__all__ = [
    "lead_lag_transform",
    "signature_level1",
    "signature_level2",
    "bspline_basis",
    "kan_layer",
    "KanSignature",
]