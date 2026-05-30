# DARWIN HAMMER — match 1201, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_dense_associa_m1179_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m503_s3.py (gen5)
# born: 2026-05-29T23:34:26Z

"""Hybrid algorithm combining dense associative KAN transformations (Parent A) with Fisher‑information‑driven bandit decision logic (Parent B).

Mathematical bridge:
- Parent A maps an input matrix M into a transformed space M̂ using B‑spline bases (bspline_basis) and a Kolmogorov‑Arnold Network (kan_transform).
- Parent B evaluates the information content of scalar observations via the Fisher score of a Gaussian beam (fisher_score) and uses this information to adapt action propensities in a contextual bandit.

The hybrid unifies these by:
1. Applying the KAN/B‑spline transform to a pheromone‑like matrix.
2. Interpreting each column (or feature) of the transformed matrix as a “measurement” θ.
3. Computing a Fisher‑information‑derived weight for each feature.
4. Feeding these weights into a bandit‑action update, where higher Fisher weight ⇒ higher propensity (entropy‑aware exploration).

Thus the entropy‑like Fisher information of the transformed pheromone patterns directly modulates the bandit policy, yielding a single coherent system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

# ----------------------------------------------------------------------
# Parent A core: B‑spline basis and KAN transform
# ----------------------------------------------------------------------
def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Cox‑de Boor recursion for uniform clamped B‑splines.

    Parameters
    ----------
    x : ndarray shape (N,)
        Evaluation points (must lie inside the grid range).
    grid : ndarray shape (G,)
        Interior breakpoints (uniform spacing assumed).
    k : int, default 3
        Spline order (degree = k‑1).  k=3 yields cubic splines.

    Returns
    -------
    B : ndarray shape (N, G + k)
    """
    n = len(x)
    g = len(grid)
    # Pad grid for clamping
    extended = np.concatenate((
        np.full(k, grid[0] - (grid[1] - grid[0]) * k),
        grid,
        np.full(k, grid[-1] + (grid[-1] - grid[-2]) * k)
    ))
    B = np.zeros((n, g + k))
    # Zeroth‑order basis (piecewise constant)
    for i in range(n):
        for j in range(g + k):
            if extended[j] <= x[i] < extended[j + 1]:
                B[i, j] = 1.0
    # Recursion for higher orders
    for d in range(1, k + 1):
        for i in range(n):
            for j in range(g + k - d):
                left_den = extended[j + d] - extended[j]
                right_den = extended[j + d + 1] - extended[j + 1]
                left = 0.0 if left_den == 0 else ((x[i] - extended[j]) / left_den) * B[i, j]
                right = 0.0 if right_den == 0 else ((extended[j + d + 1] - x[i]) / right_den) * B[i, j + 1]
                B[i, j] = left + right
    return B[:, :g + k - k]  # trim to original size


def kan_transform(M: np.ndarray, grids: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Apply a KAN‑style edgewise transform using B‑spline bases.

    Parameters
    ----------
    M : ndarray shape (N, N)
        Input matrix (e.g., pheromone concentrations).
    grids : ndarray shape (G,)
        Grid of knot locations for the B‑splines.
    coeffs : ndarray shape (G + k, N)
        Learned spline coefficients for each output dimension.

    Returns
    -------
    M̂ : ndarray shape (N, N)
        Transformed matrix.
    """
    N = M.shape[0]
    k = 3  # cubic splines
    # Flatten rows to treat each row as a set of evaluation points
    M_hat = np.empty_like(M, dtype=float)
    for i in range(N):
        row = M[i, :]
        B = bspline_basis(row, grids, k=k)          # (N, G+k)
        # Trim coeffs to match B columns
        C = coeffs[:B.shape[1], :]                  # (G+k, N)
        M_hat[i, :] = B @ C
    return M_hat

# ----------------------------------------------------------------------
# Parent B core: Fisher information and bandit structures
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian kernel evaluated at theta."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information of a Gaussian beam w.r.t. its mean."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = field(default="HybridKANBandit")


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def compute_hybrid_matrix(M: np.ndarray, grids: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Full hybrid forward pass: KAN transform followed by Fisher‑weight extraction."""
    # 1. KAN/B‑spline transform (entropy‑like representation)
    M_hat = kan_transform(M, grids, coeffs)

    # 2. For each column treat the transformed values as measurements θ_i
    #    Compute a Fisher weight using a Gaussian beam centred at the column mean.
    N = M_hat.shape[1]
    fisher_weights = np.empty(N, dtype=float)
    for j in range(N):
        col = M_hat[:, j]
        center = float(col.mean())
        width = max(col.std(), 1e-3)  # avoid zero width
        # Aggregate Fisher score across the column
        scores = [fisher_score(float(v), center, width) for v in col]
        fisher_weights[j] = float(np.mean(scores))
    # Normalise to obtain a probability‑like vector
    if fisher_weights.sum() == 0:
        normalized = np.full_like(fisher_weights, 1.0 / N)
    else:
        normalized = fisher_weights / fisher_weights.sum()
    # Return both the transformed matrix and the normalized Fisher vector
    return M_hat, normalized


def update_bandit_actions(actions: list[BanditAction], fisher_vector: np.ndarray) -> list[BanditAction]:
    """Adjust action propensities according to Fisher‑information weights.

    The number of actions must match the length of the Fisher vector.
    Propensity_i ← propensity_i * (1 + α * fisher_i)   (α is a learning rate)
    """
    if len(actions) != len(fisher_vector):
        raise ValueError("Length mismatch between actions and Fisher vector")
    alpha = 0.5  # moderate adaptation strength
    updated = []
    for act, w in zip(actions, fisher_vector):
        new_prop = act.propensity * (1.0 + alpha * w)
        # Re‑normalise later; keep other fields unchanged
        updated.append(BanditAction(
            action_id=act.action_id,
            propensity=new_prop,
            expected_reward=act.expected_reward,
            confidence_bound=act.confidence_bound,
            algorithm=act.algorithm
        ))
    # Normalise propensities to sum to 1
    total = sum(a.propensity for a in updated)
    if total == 0:
        normed = [BanditAction(a.action_id, 1.0 / len(updated), a.expected_reward,
                               a.confidence_bound, a.algorithm) for a in updated]
    else:
        normed = [BanditAction(a.action_id, a.propensity / total,
                               a.expected_reward, a.confidence_bound, a.algorithm) for a in updated]
    return normed


def select_action(actions: list[BanditAction]) -> BanditAction:
    """Sample an action according to its propensity distribution."""
    probs = [a.propensity for a in actions]
    cum = np.cumsum(probs)
    r = random.random()
    idx = np.searchsorted(cum, r, side="right")
    return actions[int(idx)]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Create a synthetic pheromone matrix (e.g., 5x5)
    np.random.seed(0)
    M = np.random.rand(5, 5)

    # 2. Define spline grid and random coefficients
    grid = np.linspace(0.0, 1.0, 6)          # 6 interior knots
    coeffs = np.random.randn(len(grid) + 3, 5)  # (G+k, N)

    # 3. Run hybrid forward pass
    M_hat, fisher_vec = compute_hybrid_matrix(M, grid, coeffs)
    print("Transformed matrix (M̂):")
    print(M_hat)
    print("\nNormalized Fisher vector:")
    print(fisher_vec)

    # 4. Initialise dummy bandit actions (one per column)
    actions = [
        BanditAction(
            action_id=f"col_{j}",
            propensity=1.0 / M.shape[1],
            expected_reward=0.0,
            confidence_bound=0.0
        )
        for j in range(M.shape[1])
    ]

    # 5. Update actions with Fisher information
    actions = update_bandit_actions(actions, fisher_vec)
    print("\nUpdated actions (propensities):")
    for a in actions:
        print(f"{a.action_id}: {a.propensity:.4f}")

    # 6. Sample an action
    chosen = select_action(actions)
    print(f"\nChosen action: {chosen.action_id} (propensity={chosen.propensity:.4f})")