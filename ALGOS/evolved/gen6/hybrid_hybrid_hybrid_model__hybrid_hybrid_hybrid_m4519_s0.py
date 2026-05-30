# DARWIN HAMMER — match 4519, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_model_pool_hy_hybrid_hybrid_minimu_m1971_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hoeffd_hybrid_voronoi_parti_m1346_s1.py (gen5)
# born: 2026-05-29T23:56:19Z

"""Hybrid Darwin Hammer: Fusion of Bayesian Curvature Allocation (Parent A) and Hoeffding‑Voronoi‑Pheromone Splitting (Parent B).

Mathematical Bridge
-------------------
- **Prior**: The curvature matrix from Parent A is interpreted as a discrete probability
  distribution `π_i` over the Voronoi cells `i`.  Cells that lie in high‑curvature
  regions receive larger prior weight.
- **Evidence**: For each cell the Gini impurity gain `ΔG_i` (Parent B) provides a
  likelihood `ℓ_i = ΔG_i / Σ_j ΔG_j`.  This quantifies how well the cell separates the
  labels.
- **Bayesian Update**: Using `π_i` as prior and `ℓ_i` as likelihood we compute the
  posterior `τ_i = P(cell_i | evidence)` via `τ_i = π_i·ℓ_i / Σ_j π_j·ℓ_j`.  The posterior
  becomes the **pheromone** strength for the cell.
- **Hoeffding Certainty**: The Hoeffding bound `ε_i = sqrt( (ln(1/δ)) / (2·n_i) )`
  (with `R=1` for Gini) supplies a statistical confidence term.
- **Unified Score**: The final split score merges all three ingredients:
  
  `S_i = (ΔG_i / (ε_i + 1e-9)) * τ_i`

  Cells with high impurity gain, strong statistical certainty, and large posterior
  pheromone are favoured.  The cell with maximal `S_i` triggers a tree split.

The module implements this fused workflow with three public functions:
`compute_curvature_prior`, `bayesian_hoeffding_split`, and `hybrid_summary`.
"""

import math
import random
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Label = int

@dataclass(frozen=True)
class SplitDecision:
    """Result of a Hoeffding‑Gini split test for a Voronoi cell after Bayesian fusion."""
    should_split: bool
    cell_index: int
    epsilon: float
    gain_gap: float
    pheromone: float
    score: float
    reason: str

# ----------------------------------------------------------------------
# Helper utilities (shared)
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def gini_impurity(labels: List[Label]) -> float:
    if not labels:
        return 0.0
    counts = np.bincount(labels)
    probs = counts / counts.sum()
    return 1.0 - np.sum(probs ** 2)

def hoeffding_epsilon(n: int, delta: float = 0.05, R: float = 1.0) -> float:
    if n <= 0:
        return float('inf')
    return math.sqrt((R ** 2 * math.log(1.0 / delta)) / (2.0 * n))

def _hash_seed(text: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha256(text.encode()).digest(), "big")

# ----------------------------------------------------------------------
# 1️⃣ Curvature‑derived prior over Voronoi cells
# ----------------------------------------------------------------------
def compute_curvature_prior(
    seeds: List[Point],
    curvature_matrix: np.ndarray,
) -> np.ndarray:
    """
    Convert a curvature matrix (square, size = len(seeds)) into a probability
    distribution over Voronoi cells.  The curvature matrix is first symmetrised,
    its diagonal is taken as the cell‑specific curvature, then normalised.
    """
    if curvature_matrix.shape != (len(seeds), len(seeds)):
        raise ValueError("Curvature matrix must be square with size equal to number of seeds.")
    # Symmetrise and extract diagonal (self‑curvature)
    curv = np.diag((curvature_matrix + curvature_matrix.T) / 2.0)
    # Ensure non‑negative
    curv = np.clip(curv, a_min=0.0, a_max=None)
    total = curv.sum()
    if total == 0.0:
        # fallback to uniform prior
        return np.full(len(seeds), 1.0 / len(seeds))
    return curv / total

# ----------------------------------------------------------------------
# 2️⃣ Bayesian‑Hoeffding split decision
# ----------------------------------------------------------------------
class _PheromoneBank:
    """Keeps decaying pheromone values per cell."""
    def __init__(self, half_life: float = 10.0):
        self.values: Dict[int, float] = defaultdict(float)
        self.half_life = half_life
        self._last_update = 0

    def decay(self, step: int = 1) -> None:
        """Exponential decay based on half‑life."""
        factor = 0.5 ** (step / self.half_life)
        for k in list(self.values.keys()):
            self.values[k] *= factor
            if self.values[k] < 1e-12:
                del self.values[k]

    def reinforce(self, cell_idx: int, amount: float) -> None:
        self.values[cell_idx] += amount

    def get(self, cell_idx: int) -> float:
        return self.values.get(cell_idx, 0.0)

_pheromone_bank = _PheromoneBank()

def _assign_to_cells(
    data: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    """
    Assign each data point to the index of its nearest Voronoi seed.
    Returns an array of shape (n_samples,) with cell indices.
    """
    # Compute squared Euclidean distances (broadcast)
    diff = data[:, None, :] - seeds[None, :, :]
    dists = np.einsum('ijk,ijk->ij', diff, diff)
    return np.argmin(dists, axis=1)

def bayesian_hoeffding_split(
    data: np.ndarray,
    labels: np.ndarray,
    seeds: List[Point],
    curvature_prior: np.ndarray,
    delta: float = 0.05,
    min_samples: int = 30,
) -> SplitDecision:
    """
    Perform a single hybrid split evaluation.

    Steps:
    1. Assign samples to Voronoi cells.
    2. For each cell compute Gini impurity gain ΔG_i.
    3. Derive likelihood ℓ_i = ΔG_i / Σ ΔG_j.
    4. Bayesian update: posterior τ_i ∝ π_i·ℓ_i (π_i from curvature_prior).
    5. Apply Hoeffding bound ε_i based on cell sample count.
    6. Compute unified score S_i = (ΔG_i / (ε_i+1e-9)) * τ_i.
    7. Return the best cell decision.
    """
    if data.shape[0] != labels.shape[0]:
        raise ValueError("data and labels must have the same length.")
    n_cells = len(seeds)
    seeds_arr = np.asarray(seeds, dtype=float)

    # 1. Assignment
    cell_assignments = _assign_to_cells(data, seeds_arr)

    # Containers
    gain = np.zeros(n_cells)
    epsilon = np.full(n_cells, float('inf'))
    counts = np.zeros(n_cells, dtype=int)

    # Global impurity (before split)
    global_impurity = gini_impurity(labels.tolist())

    # 2. Compute impurity gain per cell
    for i in range(n_cells):
        mask = cell_assignments == i
        counts[i] = mask.sum()
        if counts[i] == 0:
            continue
        cell_labels = labels[mask]
        cell_impurity = gini_impurity(cell_labels.tolist())
        # Weighted impurity reduction
        gain[i] = (global_impurity - (counts[i] / len(labels)) * cell_impurity)

    # 3. Likelihood from gain (avoid division by zero)
    total_gain = gain.sum()
    if total_gain == 0.0:
        likelihood = np.full(n_cells, 1.0 / n_cells)
    else:
        likelihood = gain / total_gain

    # 4. Bayesian update → posterior pheromone
    marginal = curvature_prior * likelihood
    marginal_sum = marginal.sum()
    if marginal_sum == 0.0:
        posterior = np.full(n_cells, 1.0 / n_cells)
    else:
        posterior = marginal / marginal_sum

    # Update pheromone bank (decay then reinforce)
    _pheromone_bank.decay()
    for i in range(n_cells):
        _pheromone_bank.reinforce(i, posterior[i])

    # 5. Hoeffding bound per cell
    for i in range(n_cells):
        epsilon[i] = hoeffding_epsilon(counts[i], delta=delta, R=1.0)

    # 6. Unified score
    scores = np.zeros(n_cells)
    for i in range(n_cells):
        if counts[i] < min_samples:
            scores[i] = -np.inf  # not enough evidence
            continue
        scores[i] = (gain[i] / (epsilon[i] + 1e-9)) * _pheromone_bank.get(i)

    # 7. Choose best cell
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    should_split = best_score > 0 and counts[best_idx] >= min_samples

    reason = "split" if should_split else "insufficient evidence"
    return SplitDecision(
        should_split=should_split,
        cell_index=best_idx,
        epsilon=epsilon[best_idx],
        gain_gap=gain[best_idx],
        pheromone=_pheromone_bank.get(best_idx),
        score=best_score,
        reason=reason,
    )

# ----------------------------------------------------------------------
# 3️⃣ Hybrid summary / cost evaluation
# ----------------------------------------------------------------------
def hybrid_summary(
    decision: SplitDecision,
    cost_per_sample: float = 0.01,
) -> Dict[str, Any]:
    """
    Produce a human‑readable summary that also incorporates a minimum‑cost
    perspective (Parent A).  The total cost is estimated as
    `cost = cost_per_sample * n_samples_in_cell`.
    """
    summary = {
        "split": decision.should_split,
        "cell": decision.cell_index,
        "epsilon": round(decision.epsilon, 6),
        "gain_gap": round(decision.gain_gap, 6),
        "pheromone": round(decision.pheromone, 6),
        "score": round(decision.score, 6),
        "reason": decision.reason,
    }
    # Mock cost estimation (we don't have n_samples here, reuse gain as proxy)
    estimated_cost = cost_per_sample * max(decision.gain_gap, 0.0) * 1000
    summary["estimated_cost"] = round(estimated_cost, 4)
    return summary

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Deterministic random seed
    random.seed(42)
    np.random.seed(42)

    # Synthetic 2‑D data
    n_samples = 500
    X = np.random.randn(n_samples, 2)
    # Binary labels with a simple linear rule
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)

    # Voronoi seeds (fixed)
    seeds = [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0), (2.0, -2.0)]

    # Mock curvature matrix (positive definite)
    curv = np.array([
        [2.0, 0.5, 0.2, 0.1],
        [0.5, 3.0, 0.3, 0.2],
        [0.2, 0.3, 1.5, 0.4],
        [0.1, 0.2, 0.4, 2.5],
    ])

    prior = compute_curvature_prior(seeds, curv)

    decision = bayesian_hoeffding_split(
        data=X,
        labels=y,
        seeds=seeds,
        curvature_prior=prior,
        delta=0.05,
        min_samples=20,
    )

    print("Split Decision:", decision)
    print("Summary:", hybrid_summary(decision))
    sys.exit(0)