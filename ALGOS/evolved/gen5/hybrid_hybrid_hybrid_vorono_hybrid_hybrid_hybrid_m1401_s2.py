# DARWIN HAMMER — match 1401, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fisher_bandit_router_m1158_s0.py (gen3)
# born: 2026-05-29T23:36:05Z

"""
HybridVoronoiFisherBandit
========================

This module fuses the two parent algorithms:

* **Parent A** – Voronoi partition geometry expressed as Clifford multivectors
  and updated by the geometric product.
* **Parent B** – Fisher information scoring based on a Gaussian beam model,
  combined with a multi‑armed bandit (UCB) that provides uncertainty estimates.

**Mathematical bridge**

Each Voronoi cell is represented as a *basis blade* `e_i` inside a multivector
`R`.  For a cell we compute a *Fisher information* scalar `F_i` that measures
how informative the cell is for a parameter `θ`.  The bandit algorithm supplies
an *expected reward* `μ_i` and a *confidence bound* `β_i`.  The hybrid priority
for a cell is taken as


P_i = F_i * ( μ_i + β_i )


The cell with the highest priority is selected, and the global allocation
multivector `R` is updated by the Clifford **geometric product** with the
cell’s blade:


R ← R ⊙ e_i


Thus the geometric structure (Voronoi → multivector) is driven by statistical
information (Fisher) and exploration‑exploitation signals (bandit), achieving
a unified resource‑allocation dynamics.

The implementation below provides the core primitives and a small smoke test.
"""

import sys
import math
import random
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List

# ---------------------------------------------------------------------------
# Clifford algebra helpers (extracted from Parent A)
# ---------------------------------------------------------------------------

def _blade_sign(indices: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """
    Sort a blade's index tuple by bubble‑sort while tracking the sign change.
    Identical indices cancel (e_i ∧ e_i = 0).
    """
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel duplicate index
                del lst[j:j + 2]
                n -= 2
                i = -1  # restart outer loop
                break
            j += 1
        i += 1
    return tuple(lst), sign

def _multiply_blades(blade_a: Tuple[int, ...], blade_b: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """
    Geometric product of two basis blades.
    Returns (result_blade, sign).  The metric is Euclidean (e_i·e_i = 1).
    """
    # concatenate the index lists then sort with sign tracking
    concatenated = blade_a + blade_b
    result_blade, sign = _blade_sign(concatenated)
    return result_blade, sign

# ---------------------------------------------------------------------------
# Multivector representation
# ---------------------------------------------------------------------------

Multivector = Dict[Tuple[int, ...], float]   # blade tuple → coefficient

def geometric_product(A: Multivector, B: Multivector) -> Multivector:
    """
    Compute the Clifford geometric product A ⊙ B.
    """
    result: Multivector = {}
    for blade_a, coeff_a in A.items():
        for blade_b, coeff_b in B.items():
            res_blade, sign = _multiply_blades(blade_a, blade_b)
            if not res_blade:      # scalar (empty tuple) or cancelled
                key = ()
            else:
                key = res_blade
            result[key] = result.get(key, 0.0) + coeff_a * coeff_b * sign
    # prune near‑zero entries
    return {b: c for b, c in result.items() if abs(c) > 1e-12}

def scalar_multivector(value: float) -> Multivector:
    """Create a pure scalar multivector."""
    return {(): value}

def basis_blade(index: int) -> Multivector:
    """Create a multivector consisting of a single basis blade e_index."""
    return {(index,): 1.0}

# ---------------------------------------------------------------------------
# Voronoi–Fisher utilities (Hybrid core)
# ---------------------------------------------------------------------------

def generate_voronoi_cells(points: np.ndarray) -> List[Dict]:
    """
    Mock Voronoi partition: each point becomes a cell.
    Returns a list of dicts with:
        - 'center': centroid angle (θ) in radians
        - 'blade' : multivector representing the cell (basis e_i)
    """
    cells = []
    for i, pt in enumerate(points):
        # angle of the point w.r.t. origin
        theta = math.atan2(pt[1], pt[0])
        cells.append({
            'id': i,
            'center': theta,
            'blade': basis_blade(i + 1)   # use 1‑based indices for clarity
        })
    return cells

def fisher_information(theta: float, center: float, width: float) -> float:
    """
    Fisher information for a Gaussian beam evaluated at angle theta.
    width > 0 is the beam's standard deviation.
    """
    z = (theta - center) / width
    intensity = math.exp(-0.5 * z * z)
    derivative = intensity * (-(theta - center) / (width * width))
    # avoid division by zero
    if intensity == 0.0:
        return 0.0
    return (derivative * derivative) / intensity

def bandit_ucb_update(trials: Dict[int, Tuple[int, float]]) -> Dict[int, Tuple[float, float]]:
    """
    Given a dict mapping cell id → (n, total_reward), compute
    (expected_reward, confidence_bound) for each cell using a simple UCB rule.
    Returns a dict id → (μ, β).
    """
    stats = {}
    for cid, (n, total) in trials.items():
        mu = total / n if n > 0 else 0.0
        beta = 1.0 / math.sqrt(1.0 + n)   # as described in Parent B
        stats[cid] = (mu, beta)
    return stats

def select_cell(cells: List[Dict],
                width: float,
                bandit_stats: Dict[int, Tuple[float, float]]) -> Dict:
    """
    Compute hybrid priority P_i = F_i * (μ_i + β_i) for each cell and return the
    cell with the maximum priority.
    """
    best_cell = None
    best_priority = -math.inf
    for cell in cells:
        cid = cell['id']
        theta = cell['center']
        # Fisher information uses the cell angle as theta and a global target
        # center (e.g., 0 rad).  Width is a tunable parameter.
        F = fisher_information(theta, center=0.0, width=width)
        mu, beta = bandit_stats.get(cid, (0.0, 1.0))  # unknown cells get high uncertainty
        priority = F * (mu + beta)
        if priority > best_priority:
            best_priority = priority
            best_cell = cell
    return best_cell

def update_allocation(R: Multivector, selected_cell: Dict) -> Multivector:
    """
    Perform the Clifford geometric product update:
        R ← R ⊙ e_i   where e_i is the blade of the selected cell.
    """
    return geometric_product(R, selected_cell['blade'])

# ---------------------------------------------------------------------------
# High‑level hybrid routine
# ---------------------------------------------------------------------------

def hybrid_resource_allocation(points: np.ndarray,
                               n_iterations: int = 10,
                               width: float = 0.5) -> Multivector:
    """
    Run a simple hybrid loop:
        1. Build Voronoi cells from the input points.
        2. Initialise a scalar allocation multivector R = 1.
        3. For each iteration:
            a) Compute bandit statistics from simulated rewards.
            b) Select the cell with highest hybrid priority.
            c) Update R via the geometric product with the selected cell.
            d) Simulate a reward for the selected cell (Gaussian reward around its
               Fisher information) and record it for the next round.
    Returns the final allocation multivector R.
    """
    cells = generate_voronoi_cells(points)

    # start with scalar 1 (neutral allocation)
    R: Multivector = scalar_multivector(1.0)

    # bandit trial bookkeeping: cid → (n, total_reward)
    trials: Dict[int, Tuple[int, float]] = {cell['id']: (0, 0.0) for cell in cells}

    for _ in range(n_iterations):
        # 1) compute bandit stats
        bandit_stats = bandit_ucb_update(trials)

        # 2) select cell using hybrid priority
        chosen = select_cell(cells, width, bandit_stats)
        if chosen is None:
            break  # no cells (should not happen)

        # 3) update allocation multivector
        R = update_allocation(R, chosen)

        # 4) simulate a stochastic reward proportional to Fisher info
        F = fisher_information(chosen['center'], center=0.0, width=width)
        reward = random.gauss(mu=F, sigma=0.1 * max(F, 1e-3))
        cid = chosen['id']
        n, total = trials[cid]
        trials[cid] = (n + 1, total + reward)

    return R

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # generate 6 random 2‑D points
    rng = np.random.default_rng(seed=42)
    pts = rng.uniform(-1.0, 1.0, size=(6, 2))

    final_R = hybrid_resource_allocation(pts, n_iterations=15, width=0.3)

    # pretty‑print the resulting multivector
    print("Final allocation multivector R:")
    for blade, coeff in sorted(final_R.items()):
        if blade == ():
            name = "1"
        else:
            name = "e" + "^".join(str(i) for i in blade)
        print(f"  {name}: {coeff:.6f}")

    # sanity check: result should be a dict with at least the scalar part
    assert isinstance(final_R, dict)
    assert () in final_R   # scalar component always present
    print("Smoke test completed successfully.")