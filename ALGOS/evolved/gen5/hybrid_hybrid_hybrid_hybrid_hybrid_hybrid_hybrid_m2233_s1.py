# DARWIN HAMMER — match 2233, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_regret_engine_m136_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s0.py (gen4)
# born: 2026-05-29T23:41:26Z

"""Hybrid Fusion of Clifford Geometric Algebra and Fisher‑Information Decision Engine

Parents:
- hybrid_hybrid_geometric_pro_hybrid_hybrid_model__m176_s1.py (Clifford algebra)
- hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s0.py (Gaussian beam, Fisher information, tree metrics)

Mathematical Bridge:
The Fisher information of a Gaussian sensing model is used as a scalar weight that
modulates the coefficients of a multivector (Clifford algebra element).  By
scaling each blade’s coefficient with the Fisher score evaluated at a sensing
angle θ, the geometric product can be turned into a regret‑aware, information‑
driven operator.  The resulting hybrid product simultaneously encodes
directional information (via the Fisher score) and the exterior‑algebraic
structure of actions or counterfactuals.

The module provides three core hybrid functions:
1. `hybrid_geometric_product` – weighted geometric product of two multivectors.
2. `expected_information_gain` – Fisher‑weighted sum of tree‑distance entropies.
3. `regret_weighted_hybrid` – builds a multivector from actions/counterfactuals,
   applies Fisher weighting, and returns its self‑product as a regret‑aware
   measure.
"""

import sys
import math
import random
import pathlib
from dataclasses import dataclass
from typing import Dict, Tuple, List, FrozenSet, Any
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Clifford algebra utilities
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list.
    Duplicate indices cancel (e_i ^ e_i = 0)."""
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
                # cancel duplicate basis vector
                lst.pop(j)
                lst.pop(j)  # second element shifts to j after first pop
                n -= 2
                i = -1  # restart outer loop because length changed
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def geometric_product(a: Dict[FrozenSet[int], float],
                      b: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """Full Clifford product `ab`.  Multivectors are dicts mapping blades → scalar."""
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coef_a in a.items():
        for blade_b, coef_b in b.items():
            blade, sign = _multiply_blades(blade_a, blade_b)
            result[blade] = result.get(blade, 0.0) + sign * coef_a * coef_b
    return result

# ----------------------------------------------------------------------
# Parent B – Fisher information, Gaussian beam, tree metrics
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(nodes: Dict[str, Tuple[float, float]],
                edges: List[Tuple[str, str]],
                root: str) -> Tuple[Dict[str, List[str]],
                                   Dict[Tuple[str, str], float],
                                   Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths, and root‑to‑node distances.
    Returns (adjacency, edge_lengths, root_distances).
    """
    adj: Dict[str, List[str]] = {node: [] for node in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}

    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        d = length(nodes[u], nodes[v])
        edge_len[(u, v)] = d
        edge_len[(v, u)] = d

    # BFS to compute distances from root
    root_dist: Dict[str, float] = {root: 0.0}
    visited = {root}
    queue = [root]
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in visited:
                root_dist[nb] = root_dist[cur] + edge_len[(cur, nb)]
                visited.add(nb)
                queue.append(nb)

    return adj, edge_len, root_dist

# ----------------------------------------------------------------------
# Shared data structures (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def _action_to_blade(action: MathAction, max_index: int = 8) -> FrozenSet[int]:
    """
    Deterministically map an action ID to a basis blade.
    The hash of the ID is reduced modulo `max_index` and the resulting
    integer(s) form a blade.  For simplicity we use a single‑index blade.
    """
    idx = (hash(action.id) % max_index) + 1  # indices start at 1
    return frozenset({idx})


def actions_to_multivector(actions: List[MathAction]) -> Dict[FrozenSet[int], float]:
    """
    Convert a list of actions into a multivector.
    Coefficient = expected_value - cost - risk (regret proxy).
    """
    mv: Dict[FrozenSet[int], float] = {}
    for act in actions:
        blade = _action_to_blade(act)
        coef = act.expected_value - act.cost - act.risk
        mv[blade] = mv.get(blade, 0.0) + coef
    return mv


def counterfactuals_to_multivector(counterfactuals: List[MathCounterfactual]) -> Dict[FrozenSet[int], float]:
    """
    Convert counterfactuals into a multivector.
    Each counterfactual contributes its outcome value weighted by probability.
    The blade is derived from the action_id similarly to actions.
    """
    mv: Dict[FrozenSet[int], float] = {}
    for cf in counterfactuals:
        idx = (hash(cf.action_id) % 8) + 1
        blade = frozenset({idx})
        coef = cf.outcome_value * cf.probability
        mv[blade] = mv.get(blade, 0.0) + coef
    return mv

# ----------------------------------------------------------------------
# Hybrid Functions (core of the fusion)
# ----------------------------------------------------------------------
def hybrid_geometric_product(
    mv_a: Dict[FrozenSet[int], float],
    mv_b: Dict[FrozenSet[int], float],
    theta: float,
    center: float,
    width: float
) -> Dict[FrozenSet[int], float]:
    """
    Perform a Fisher‑weighted geometric product.
    Each coefficient of `mv_a` is multiplied by the Fisher score at (theta,center,width)
    before the Clifford product with `mv_b`.
    """
    f_score = fisher_score(theta, center, width)
    weighted_a = {blade: coef * f_score for blade, coef in mv_a.items()}
    return geometric_product(weighted_a, mv_b)


def expected_information_gain(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    multivector: Dict[FrozenSet[int], float],
    theta: float,
    center: float,
    width: float
) -> float:
    """
    Compute a Fisher‑weighted expected information gain over a tree.
    For each node we treat its root distance as an entropy proxy and weight it
    by the sum of absolute multivector coefficients whose blade index matches the
    node hash modulo the number of basis vectors.
    """
    _, _, root_dist = tree_metrics(nodes, edges, root)
    f_score = fisher_score(theta, center, width)

    gain = 0.0
    for node, dist in root_dist.items():
        # map node name to a basis index
        idx = (hash(node) % 8) + 1
        # accumulate magnitude of coefficients that contain this index
        weight = sum(abs(coef) for blade, coef in multivector.items() if idx in blade)
        gain += f_score * weight * dist
    return gain


def regret_weighted_hybrid(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    theta: float,
    center: float,
    width: float
) -> Dict[FrozenSet[int], float]:
    """
    Build a combined multivector from actions and counterfactuals,
    apply Fisher weighting, and return its self‑geometric product.
    This yields a scalar‑rich multivector encoding both regret and
    information‑gain characteristics.
    """
    mv_actions = actions_to_multivector(actions)
    mv_cfs = counterfactuals_to_multivector(counterfactuals)

    # Combine the two multivectors (simple addition of coefficients for matching blades)
    combined: Dict[FrozenSet[int], float] = {}
    for mv in (mv_actions, mv_cfs):
        for blade, coef in mv.items():
            combined[blade] = combined.get(blade, 0.0) + coef

    # Apply Fisher weighting and compute self‑product
    return hybrid_geometric_product(combined, combined, theta, center, width)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample actions and counterfactuals
    actions = [
        MathAction(id="a1", expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="a2", expected_value=5.0, cost=1.5, risk=0.5),
    ]
    counterfactuals = [
        MathCounterfactual(action_id="a1", outcome_value=8.0, probability=0.7),
        MathCounterfactual(action_id="a2", outcome_value=4.0, probability=0.4),
    ]

    # Build multivectors
    mv_a = actions_to_multivector(actions)
    mv_b = counterfactuals_to_multivector(counterfactuals)

    # Parameters for Fisher information
    theta, center, width = 0.3, 0.0, 1.0

    # Hybrid geometric product
    prod = hybrid_geometric_product(mv_a, mv_b, theta, center, width)
    print("Hybrid geometric product:", prod)

    # Tree structure for information gain test
    nodes = {
        "root": (0.0, 0.0),
        "n1": (1.0, 0.0),
        "n2": (0.0, 1.0),
        "n3": (1.0, 1.0),
    }
    edges = [("root", "n1"), ("root", "n2"), ("n1", "n3"), ("n2", "n3")]
    gain = expected_information_gain(nodes, edges, "root", mv_a, theta, center, width)
    print("Expected information gain:", gain)

    # Regret‑weighted hybrid self‑product
    self_prod = regret_weighted_hybrid(actions, counterfactuals, theta, center, width)
    print("Regret‑weighted hybrid self product:", self_prod)