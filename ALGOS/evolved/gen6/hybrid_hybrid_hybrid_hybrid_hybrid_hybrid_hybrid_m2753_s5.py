# DARWIN HAMMER — match 2753, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hard_t_m1426_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hdc_serpentin_m1033_s0.py (gen3)
# born: 2026-05-29T23:45:48Z

"""
Hybrid Algorithm: Geometric‑CMS‑Bandit Fusion

Parents:
- hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hard_t_m1426_s0.py (Clifford geometric product + ternary routing + stylometry)
- hybrid_hybrid_hybrid_bandit_hybrid_hdc_serpentin_m1033_s0.py (Count‑Min Sketch, sphericity‑driven HDC, multi‑armed bandit)

Mathematical Bridge:
The geometric product of multivectors yields scalar‑valued distances between
points in the ternary‑route graph. These scalar distances are streamed as
tokens into a Count‑Min Sketch (CMS). The CMS width is dynamically scaled by
the sphericity index of a morphological object, thus linking the HDC‑derived
shape measure to the sketch’s resolution. The estimated frequencies of
distance tokens serve as a proxy for “novelty” of a route node; a bandit
algorithm consumes these novelty estimates (converted to propensities) to
select actions (route nodes) with an exploration‑exploitation trade‑off.

The three core functions below illustrate this fused pipeline:
1. geometric_distance – computes a Euclidean‑like norm from the geometric product.
2. update_cms_with_distance – hashes the distance into a CMS whose width is set by sphericity.
3. select_action_via_bandit – uses CMS‑estimated novelty to drive a UCB‑style bandit.
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
# Clifford algebra helpers (Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list.

    Each transposition flips the sign (anti‑commutativity). Duplicate indices
    cancel because e_i * e_i = 1.
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
                # cancel pair
                lst.pop(j)
                lst.pop(j)
                n -= 2
                i = -1  # restart outer loop
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades.

    Returns (resulting_blade, sign). The coefficient is handled outside.
    """
    if not blade_a:
        return blade_b, 1
    if not blade_b:
        return blade_a, 1
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign


def geometric_product(mv_a: Dict[FrozenSet[int], float],
                      mv_b: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """Geometric product of two multivectors represented as dicts."""
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in mv_a.items():
        for blade_b, coeff_b in mv_b.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            coeff_res = coeff_a * coeff_b * sign
            result[blade_res] = result.get(blade_res, 0.0) + coeff_res
    # Remove near‑zero entries
    return {b: c for b, c in result.items() if abs(c) > 1e-12}


def multivector_norm(mv: Dict[FrozenSet[int], float]) -> float:
    """L2 norm of a multivector (treating coefficients as vector components)."""
    return math.sqrt(sum(c * c for c in mv.values()))


def point_to_multivector(coords: Tuple[float, ...]) -> Dict[FrozenSet[int], float]:
    """Encode a Euclidean point as a grade‑1 multivector (sum_i x_i e_i)."""
    mv: Dict[FrozenSet[int], float] = {}
    for idx, val in enumerate(coords, start=1):
        if abs(val) > 1e-12:
            mv[frozenset({idx})] = val
    return mv


def geometric_distance(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
    """Distance derived from the geometric product of two points."""
    mv1 = point_to_multivector(p1)
    mv2 = point_to_multivector(p2)
    gp = geometric_product(mv1, mv2)          # grade‑0 (scalar) + grade‑2 parts
    # The scalar part encodes the inner product; the bivector part encodes the exterior.
    # Use the norm of the full product as a symmetric distance measure.
    return multivector_norm(gp)


# ----------------------------------------------------------------------
# Sphericity & Morphology (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float = 1.0


def sphericity_index(length: float, width: float, height: float) -> float:
    """Dimensionless sphericity: (V)^{1/3} / L where V = l·w·h."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive.")
    return (length * width * height) ** (1.0 / 3.0) / length


# ----------------------------------------------------------------------
# Count‑Min Sketch (Parent B)
# ----------------------------------------------------------------------
class CountMinSketch:
    """Simple Count‑Min Sketch with integer counters."""
    def __init__(self, width: int = 64, depth: int = 4, seed: int = 0):
        if width <= 0 or depth <= 0:
            raise ValueError("Width and depth must be positive integers.")
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.int64)
        self.seed = seed
        random.seed(seed)

    @staticmethod
    def _hash(item: str, depth: int, width: int) -> List[int]:
        return [
            int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            for d in range(depth)
        ]

    def update(self, item: str, inc: int = 1) -> None:
        indices = self._hash(item, self.depth, self.width)
        for d, idx in enumerate(indices):
            self.tables[d, idx] += inc

    def estimate(self, item: str) -> int:
        indices = self._hash(item, self.depth, self.width)
        return int(min(self.tables[d, idx] for d, idx in enumerate(indices)))


# ----------------------------------------------------------------------
# Bandit structures (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float = 0.0
    expected_reward: float = 0.0
    confidence_bound: float = 0.0
    algorithm: str = "HybridGeometricCMSBandit"


def _ucb_score(action: BanditAction, total_counts: int, exploration_coef: float = 2.0) -> float:
    """Upper‑Confidence‑Bound score used for action selection."""
    if action.propensity == 0:
        return float('inf')
    avg_reward = action.expected_reward / action.propensity
    bonus = exploration_coef * math.sqrt(math.log(max(total_counts, 1)) / action.propensity)
    return avg_reward + bonus


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def update_cms_with_distance(cms: CountMinSketch,
                             distance: float,
                             scale: float = 1.0) -> None:
    """
    Quantize a floating distance, optionally scale it, and feed it to the CMS.
    The quantization uses a simple fixed‑point representation to keep the token
    deterministic across runs.
    """
    token = f"{int(distance * scale):08d}"
    cms.update(token)


def compute_novelty_estimate(cms: CountMinSketch,
                             distance: float,
                             scale: float = 1.0) -> float:
    """
    Return an inverse‑frequency estimate: rarer distances (lower CMS count)
    are considered more novel and thus receive higher novelty scores.
    """
    token = f"{int(distance * scale):08d}"
    freq = cms.estimate(token)
    # Add 1 to avoid division by zero and to keep scores bounded.
    return 1.0 / (freq + 1)


def select_action_via_bandit(actions: List[BanditAction],
                             cms: CountMinSketch,
                             distances: Dict[str, float],
                             morphology: Morphology,
                             exploration_coef: float = 2.0) -> BanditAction:
    """
    Fuse geometry, CMS novelty, and sphericity:
    * The CMS width is set proportionally to the sphericity index.
    * Each action corresponds to a route node identified by its label.
    * Propensity for an action is updated with the novelty estimate of its
      associated distance token.
    * UCB scores are computed and the best action is returned.
    """
    # Adjust CMS width if not already sized appropriately.
    desired_width = max(32, int(64 * sphericity_index(morphology.length,
                                                    morphology.width,
                                                    morphology.height)))
    if cms.width != desired_width:
        # Re‑initialise a fresh sketch with the new width while preserving counts.
        old_tables = cms.tables.copy()
        cms.__init__(width=desired_width, depth=cms.depth, seed=cms.seed)
        # Simple transfer: re‑hash old tokens (approximate).
        for d in range(old_tables.shape[0]):
            for idx, cnt in enumerate(old_tables[d]):
                if cnt > 0:
                    # Re‑insert dummy tokens to keep mass; not exact but sufficient.
                    for _ in range(cnt):
                        cms.tables[d, idx % desired_width] += 1

    total_counts = int(cms.tables.sum())
    updated_actions: List[BanditAction] = []
    for act in actions:
        dist = distances.get(act.action_id, 0.0)
        novelty = compute_novelty_estimate(cms, dist, scale=1.0)
        # Propensity is treated as a count of "visits" weighted by novelty.
        new_propensity = act.propensity + novelty
        # Expected reward is left unchanged (placeholder for external feedback).
        updated_actions.append(BanditAction(
            action_id=act.action_id,
            propensity=new_propensity,
            expected_reward=act.expected_reward,
            confidence_bound=act.confidence_bound,
            algorithm=act.algorithm
        ))

    # Choose action with highest UCB score.
    best_action = max(updated_actions,
                      key=lambda a: _ucb_score(a, total_counts, exploration_coef))
    return best_action


def hybrid_step(points: List[Tuple[float, ...]],
                route_nodes: List[Tuple[float, ...]],
                morphology: Morphology,
                cms: CountMinSketch,
                actions: List[BanditAction]) -> BanditAction:
    """
    One iteration of the hybrid algorithm:
    1. Compute geometric distances from each point to each route node.
    2. Update the CMS with those distances (scaled by sphericity).
    3. Assemble a distance map keyed by action_id (assumed to match node index).
    4. Run the bandit selector and return the chosen action.
    """
    # Scale factor derived from sphericity (makes distances coarser for low sphericity).
    scale = sphericity_index(morphology.length, morphology.width, morphology.height)

    # Compute distances and feed CMS.
    distance_map: Dict[str, float] = {}
    for i, node in enumerate(route_nodes):
        # Aggregate distance to all points (mean distance) for this node.
        dists = [geometric_distance(p, node) for p in points]
        mean_dist = sum(dists) / max(len(dists), 1)
        distance_map[str(i)] = mean_dist
        update_cms_with_distance(cms, mean_dist, scale=scale)

    # Select action based on updated CMS and geometry.
    chosen = select_action_via_bandit(actions, cms, distance_map, morphology)
    return chosen


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic data
    random.seed(42)
    np.random.seed(42)

    # Define a few points in 3‑D space
    points = [(random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5))
              for _ in range(5)]

    # Define route nodes (also 3‑D)
    route_nodes = [(0.0, 0.0, 0.0),
                   (2.0, 2.0, 2.0),
                   (-3.0, 1.0, 4.0)]

    # Morphology influencing CMS width
    morph = Morphology(length=4.0, width=3.0, height=2.0)

    # Initialise CMS with a width that will be adapted inside the algorithm
    cms = CountMinSketch(width=64, depth=4, seed=7)

    # Create placeholder actions (one per route node)
    actions = [BanditAction(action_id=str(i)) for i in range(len(route_nodes))]

    # Run a few hybrid steps
    for step in range(3):
        chosen = hybrid_step(points, route_nodes, morph, cms, actions)
        print(f"Step {step + 1}: Chosen action -> {chosen.action_id}")
        # Simulate a reward update (dummy reward = inverse of chosen distance)
        chosen_dist = chosen.action_id
        reward = 1.0 / (float(chosen_dist) + 1.0)  # placeholder
        # Update the action list with the simulated reward
        actions = [
            BanditAction(
                action_id=a.action_id,
                propensity=a.propensity,
                expected_reward=(a.expected_reward + reward) if a.action_id == chosen.action_id else a.expected_reward,
                confidence_bound=a.confidence_bound,
                algorithm=a.algorithm
            )
            for a in actions
        ]
    print("Smoke test completed without errors.")