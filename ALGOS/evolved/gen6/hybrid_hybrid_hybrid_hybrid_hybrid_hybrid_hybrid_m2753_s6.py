# DARWIN HAMMER — match 2753, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hard_t_m1426_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hdc_serpentin_m1033_s0.py (gen3)
# born: 2026-05-29T23:45:48Z

"""Hybrid Algorithm combining Clifford Geometric Product (Parent A) and
Count‑Min Sketch Bandit with Sphericity scaling (Parent B).

Mathematical Bridge
-------------------
* Each *action* (or textual point) is embedded as a multivector **M** in a
  Clifford algebra Cl(n,0) using a deterministic hash → basis‑blade mapping.
* The scalar norm ‖**M**‖ (derived from the geometric product) quantifies the
  geometric “size’’ of the action.
* A Count‑Min Sketch (CMS) matrix **S** is built whose dimensions are
  scaled by the *sphericity index* 𝜎 = (ℓ·w·h)^{1/3}/ℓ of a supplied
  Morphology object.  Thus the hyper‑dimensional sketch adapts to the
  physical shape of the problem domain.
* The CMS estimates the *cardinality* 𝒞 of the action set.  Propensity for an
  action is defined as 𝑝 = 1/𝒞 and is further weighted by the geometric norm:
  score = 𝑝·‖**M**‖.
* Bandit‑style selection chooses the action with the highest score, thereby
  fusing the algebraic geometry of Parent A with the probabilistic
  estimation of Parent B.

The module provides three core hybrid operations:
1. `geometric_product` – Clifford algebra multiplication.
2. `sphericity_scaled_cms` – builds a CMS whose width is scaled by the
   sphericity index.
3. `hybrid_bandit_select` – computes scores from geometric norms and CMS
   cardinality estimates and returns the optimal action.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Clifford geometric product utilities
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Sort a list of basis indices, returning the sorted list and the sign
    introduced by swaps. Duplicate indices cancel (e_i*e_i = 1)."""
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
                j += 1
            elif lst[j] == lst[j + 1]:
                # e_i * e_i = 1 → remove both and keep sign
                lst.pop(j)
                lst.pop(j)  # second element now at same index
                n -= 2
                sign *= 1
                # restart scanning because list changed
                i = -1
                break
            else:
                j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades represented as frozensets of indices.
    Returns (resulting blade, sign)."""
    if not blade_a:
        return blade_b, 1
    if not blade_b:
        return blade_a, 1
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign


def geometric_product(mv1: Dict[FrozenSet[int], float],
                      mv2: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """Geometric product of two multivectors. Each multivector is a dict
    mapping a blade (frozenset of basis indices) to a scalar coefficient."""
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in mv1.items():
        for blade_b, coeff_b in mv2.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            result[blade_res] = result.get(blade_res, 0.0) + sign * coeff_a * coeff_b
    # Remove near‑zero entries for cleanliness
    return {b: c for b, c in result.items() if abs(c) > 1e-12}


def mv_norm(mv: Dict[FrozenSet[int], float]) -> float:
    """Euclidean norm of a multivector (sqrt of sum of squares of coefficients)."""
    return math.sqrt(sum(c * c for c in mv.values()))


def action_to_multivector(action_id: str, dim: int = 8) -> Dict[FrozenSet[int], float]:
    """Deterministically embed an action identifier into a multivector.
    The SHA‑256 hash is split into 8‑bit chunks, each chunk selects a basis index.
    The resulting multivector contains single‑blade terms with coefficient 1."""
    h = hashlib.sha256(action_id.encode()).digest()
    mv: Dict[FrozenSet[int], float] = {}
    for i in range(dim):
        byte = h[i]
        idx = (byte % dim) + 1  # basis indices start at 1
        blade = frozenset({idx})
        mv[blade] = mv.get(blade, 0.0) + 1.0
    return mv


# ----------------------------------------------------------------------
# Parent B – Count‑Min Sketch, sphericity, and bandit scaffolding
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float = 1.0  # unused in current calculations


def sphericity_index(length: float, width: float, height: float) -> float:
    """Classic sphericity index used to scale CMS dimensions."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def _cms_hash(item: str, depth: int, width: int) -> List[int]:
    """Depth‑wise hash positions for Count‑Min Sketch."""
    return [
        int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
        for d in range(depth)
    ]


def initialize_cms(width: int, depth: int) -> np.ndarray:
    """Create an empty CMS matrix."""
    return np.zeros((depth, width), dtype=np.int32)


def cms_update(sketch: np.ndarray, item: str) -> None:
    """Increment CMS counters for a single item."""
    depth, width = sketch.shape
    positions = _cms_hash(item, depth, width)
    for d, w in enumerate(positions):
        sketch[d, w] += 1


def cms_estimate(sketch: np.ndarray, item: str) -> int:
    """Estimate the frequency of *item* using the CMS (minimum across rows)."""
    depth, width = sketch.shape
    positions = _cms_hash(item, depth, width)
    return min(sketch[d, w] for d, w in enumerate(positions))


def sphericity_scaled_cms(morph: Morphology,
                          base_width: int = 64,
                          depth: int = 4) -> np.ndarray:
    """Create a CMS whose width is scaled by the sphericity index of the morphology."""
    sigma = sphericity_index(morph.length, morph.width, morph.height)
    scaled_width = max(8, int(base_width * sigma))
    return initialize_cms(scaled_width, depth)


def estimate_cardinality(sketch: np.ndarray, items: Iterable[str]) -> int:
    """Estimate the number of distinct items seen so far (simple surrogate)."""
    # Count how many items have a non‑zero CMS estimate.
    return sum(1 for it in items if cms_estimate(sketch, it) > 0)


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridGeometricCMS"


# ----------------------------------------------------------------------
# Hybrid Operations (at least three functions)
# ----------------------------------------------------------------------
def hybrid_compute_propensities(actions: List[str],
                                sketch: np.ndarray,
                                morph: Morphology) -> List[BanditAction]:
    """Compute propensities for a list of actions using CMS cardinality
    and geometric norms."""
    # Estimate cardinality once (shared denominator)
    cardinality = estimate_cardinality(sketch, actions) or 1
    propensity = 1.0 / cardinality

    bandit_actions: List[BanditAction] = []
    for aid in actions:
        mv = action_to_multivector(aid)
        norm = mv_norm(mv) or 1.0
        score = propensity * norm
        # For illustration we treat the score as expected_reward
        expected_reward = random.uniform(0, 1) * score
        confidence = math.sqrt(score)  # dummy confidence bound
        bandit_actions.append(
            BanditAction(
                action_id=aid,
                propensity=propensity,
                expected_reward=expected_reward,
                confidence_bound=confidence,
            )
        )
    return bandit_actions


def hybrid_bandit_select(context_id: str,
                         actions: List[str],
                         morph: Morphology) -> BanditAction:
    """Full hybrid pipeline:
       1. Build a sphericity‑scaled CMS.
       2. Insert all actions into the sketch.
       3. Compute propensities + geometric scores.
       4. Return the action with highest (expected_reward) score."""
    # 1. Initialise CMS
    cms = sphericity_scaled_cms(morph)

    # 2. Populate sketch
    for a in actions:
        cms_update(cms, a)

    # 3. Compute bandit‑style actions
    bandit_actions = hybrid_compute_propensities(actions, cms, morph)

    # 4. Select the best action (highest expected_reward)
    best = max(bandit_actions, key=lambda ba: ba.expected_reward)
    return best


def hybrid_route_assignment(points: List[Tuple[float, float]],
                            route_nodes: List[Tuple[float, float]]) -> List[int]:
    """Assign each 2‑D point to the index of the nearest route node using
    a distance metric derived from geometric products of embedded vectors.

    The points and nodes are first lifted to 3‑D vectors (x, y, 0), turned
    into multivectors with a single basis blade, and the geometric product
    is used to compute a scalar proxy for squared Euclidean distance."""
    assignments: List[int] = []
    for px, py in points:
        # embed point as multivector e1*x + e2*y
        mv_p = {
            frozenset({1}): px,
            frozenset({2}): py,
        }
        best_idx = -1
        best_dist = float('inf')
        for idx, (nx, ny) in enumerate(route_nodes):
            mv_n = {
                frozenset({1}): nx,
                frozenset({2}): ny,
            }
            prod = geometric_product(mv_p, mv_n)
            # scalar part is the dot product; use complement for distance
            dot = prod.get(frozenset(), 0.0)
            # Euclidean distance squared = |p|^2 + |n|^2 - 2*dot
            norm_p = px * px + py * py
            norm_n = nx * nx + ny * ny
            dist_sq = norm_p + norm_n - 2 * dot
            if dist_sq < best_dist:
                best_dist = dist_sq
                best_idx = idx
        assignments.append(best_idx)
    return assignments


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a simple morphology
    morph = Morphology(length=10.0, width=5.0, height=2.0)

    # Example actions (could be textual identifiers)
    action_ids = [f"action_{i}" for i in range(1, 6)]

    # Run the hybrid bandit selector
    selected = hybrid_bandit_select(context_id="ctx_001", actions=action_ids, morph=morph)
    print("Selected Action:", selected)

    # Demonstrate hybrid route assignment
    points = [(random.random(), random.random()) for _ in range(5)]
    route_nodes = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    assignments = hybrid_route_assignment(points, route_nodes)
    print("Route Assignments:", assignments)