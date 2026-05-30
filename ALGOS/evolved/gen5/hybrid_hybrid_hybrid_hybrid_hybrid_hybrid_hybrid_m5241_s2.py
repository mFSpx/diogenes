# DARWIN HAMMER — match 5241, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_path_s_path_signature_m501_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_path_signatur_m56_s2.py (gen3)
# born: 2026-05-30T00:00:55Z

import numpy as np
from typing import Callable, Tuple


class HybridSystem:
    """
    A corrected and deeper integration of a lead‑lag path transformation,
    B‑spline signature extraction and bandit‑router store dynamics.
    """

    def __init__(self, grid: np.ndarray, k: int = 3):
        """
        Parameters
        ----------
        grid : np.ndarray
            Knot positions for the B‑spline basis (must be sorted).
        k : int, optional
            Degree of the B‑spline (default is 3 → cubic).
        """
        self.grid = np.asarray(grid, dtype=np.float64)
        self.k = int(k)
        self.level = 0.0                     # store level
        self._prepare_knots()

    # --------------------------------------------------------------------- #
    # 1. Lead‑lag transformation
    # --------------------------------------------------------------------- #
    @staticmethod
    def lead_lag_transform(path: np.ndarray) -> np.ndarray:
        """
        Convert a path X(t)∈ℝᵈ into its lead‑lag augmentation
        of shape (2T‑1, 2d).
        """
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2:
            raise ValueError("path must be a 2‑D array (T, d).")
        T, d = path.shape
        out = np.empty((2 * T - 1, 2 * d), dtype=np.float64)
        for t in range(T - 1):
            out[2 * t] = np.concatenate([path[t], path[t]])
            out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
        out[-1] = np.concatenate([path[-1], path[-1]])
        return out

    # --------------------------------------------------------------------- #
    # 2. B‑spline basis (Cox‑de Boor recursion)
    # --------------------------------------------------------------------- #
    def _prepare_knots(self) -> None:
        """Create a clamped knot vector suitable for the chosen degree."""
        k = self.k
        grid = self.grid
        self.t = np.concatenate([
            np.full(k, grid[0]),   # left clamping (k copies)
            grid,
            np.full(k, grid[-1])   # right clamping
        ])
        self.n_basis = len(self.t) - k - 1   # number of basis functions

    def _cox_de_boor(self, x: float, i: int, k: int) -> float:
        """Recursive evaluation of a single B‑spline basis B_{i,k}(x)."""
        t = self.t
        if k == 0:
            return 1.0 if t[i] <= x < t[i + 1] else 0.0
        left = 0.0
        denom_left = t[i + k] - t[i]
        if denom_left > 0:
            left = ((x - t[i]) / denom_left) * self._cox_de_boor(x, i, k - 1)
        right = 0.0
        denom_right = t[i + k + 1] - t[i + 1]
        if denom_right > 0:
            right = ((t[i + k + 1] - x) / denom_right) * self._cox_de_boor(x, i + 1, k - 1)
        return left + right

    def bspline_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the full B‑spline basis matrix at points x.
        Returns a matrix B of shape (len(x), n_basis).
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("x must be a 1‑D array for basis evaluation.")
        B = np.zeros((x.size, self.n_basis), dtype=np.float64)
        for idx, xv in enumerate(x):
            # The basis functions are non‑zero only for indices where
            # t[i] <= xv < t[i+1]; we test a small neighbourhood.
            for i in range(self.n_basis):
                B[idx, i] = self._cox_de_boor(xv, i, self.k)
        return B

    # --------------------------------------------------------------------- #
    # 3. Signature extraction (lead‑lag + B‑spline)
    # --------------------------------------------------------------------- #
    def lead_lag_bspline_signature(self, path: np.ndarray) -> np.ndarray:
        """
        Produce a signature matrix S ∈ ℝ^{(2T‑1) × (n_basis·2d)}.
        Each column corresponds to a B‑spline basis evaluated on one
        coordinate of the lead‑lag path.
        """
        lead_lag = self.lead_lag_transform(path)          # (2T‑1, 2d)
        signatures = []
        for dim in range(lead_lag.shape[1]):
            coord = lead_lag[:, dim]                       # (2T‑1,)
            B = self.bspline_basis(coord)                  # (2T‑1, n_basis)
            signatures.append(B)
        # Concatenate along the feature axis
        return np.hstack(signatures)                       # (2T‑1, n_basis·2d)

    # --------------------------------------------------------------------- #
    # 4. Bandit‑router propensities adjustment
    # --------------------------------------------------------------------- #
    @staticmethod
    def adjust_bandit_propensities(propensities: np.ndarray,
                                   dance: np.ndarray) -> np.ndarray:
        """
        Modulate propensities by a smooth tanh‑scaled dance signal.
        """
        propensities = np.asarray(propensities, dtype=np.float64)
        dance = np.asarray(dance, dtype=np.float64)
        return propensities * np.tanh(dance)

    # --------------------------------------------------------------------- #
    # 5. Store dynamics driven by the signature
    # --------------------------------------------------------------------- #
    def store_update(self,
                     signature: np.ndarray,
                     propensities: np.ndarray,
                     alpha: float,
                     beta: float,
                     dt: float) -> float:
        """
        Update the internal store level using a weighted inflow/outflow
        derived from the signature.

        Parameters
        ----------
        signature : np.ndarray
            Matrix S of shape (N, M) where M = n_basis·2d.
        propensities : np.ndarray
            Vector of length M that weights each basis coefficient.
        alpha, beta, dt : float
            Physical‑style parameters.

        Returns
        -------
        float
            Updated store level (non‑negative).
        """
        if signature.shape[1] != propensities.size:
            raise ValueError("propensities length must match signature feature dimension.")

        # Weighted flow per time step
        weighted = signature @ propensities                     # (N,)

        inflow = np.sum(np.maximum(weighted, 0.0))               # positive contribution
        outflow = np.sum(np.abs(np.minimum(weighted, 0.0)))      # magnitude of negative contribution

        delta = alpha * inflow - beta * outflow
        self.level = max(0.0, self.level + delta * dt)
        return self.level

    # --------------------------------------------------------------------- #
    # 6. End‑to‑end update (signature → dance → propensities → store)
    # --------------------------------------------------------------------- #
    def step(self,
             path: np.ndarray,
             propensities: np.ndarray,
             alpha: float = 1.0,
             beta: float = 1.0,
             dt: float = 1.0) -> Tuple[float, np.ndarray]:
        """
        Perform a full iteration:
        1. Compute the B‑spline signature of the lead‑lag path.
        2. Derive a dance signal from the signature.
        3. Adjust the bandit propensities.
        4. Update the store level.

        Returns
        -------
        level : float
            Updated store level.
        adjusted_propensities : np.ndarray
            Propensities after modulation.
        """
        signature = self.lead_lag_bspline_signature(path)

        # Dance signal: a non‑linear combination of row‑wise and column‑wise sums
        row_sum = np.sum(signature, axis=1)          # (N,)
        col_sum = np.sum(signature, axis=0)          # (M,)
        dance = alpha * (beta * col_sum - row_sum.mean())

        adjusted = self.adjust_bandit_propensities(propensities, dance)
        level = self.store_update(signature, adjusted, alpha, beta, dt)
        return level, adjusted

    # --------------------------------------------------------------------- #
    # 7. Smoke test
    # --------------------------------------------------------------------- #
    def smoke_test(self) -> None:
        T, d = 12, 3
        path = np.random.rand(T, d)
        grid = np.linspace(0.0, 1.0, 15)          # knot grid for each coordinate
        system = HybridSystem(grid, k=3)

        # Initialise random propensities matching the signature dimension
        dummy_sig = system.lead_lag_bspline_signature(path)
        propensities = np.random.rand(dummy_sig.shape[1])

        level, adjusted = system.step(
            path,
            propensities,
            alpha=0.8,
            beta=0.5,
            dt=0.1
        )
        print(f"Smoke test completed – store level: {level:.4f}")



if __name__ == "__main__":
    HybridSystem(np.linspace(0, 1, 10)).smoke_test()