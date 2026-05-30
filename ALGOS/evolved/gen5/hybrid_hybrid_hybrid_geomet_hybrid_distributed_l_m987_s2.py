# DARWIN HAMMER — match 987, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s2.py (gen4)
# parent_b: hybrid_distributed_leader_e_thanatosis_m65_s2.py (gen1)
# born: 2026-05-29T23:32:12Z

import math
import random
from typing import Dict, Set, Tuple, List, Iterable, Any

# ----------------------------------------------------------------------
# Clifford algebra utilities (bit‑mask blade representation)
# ----------------------------------------------------------------------
def _blade_sign(mask_a: int, mask_b: int) -> int:
    """
    Sign of the geometric product of two basis blades represented by bit masks.
    The sign is (-1)^{#inversions} where an inversion is a pair of basis vectors
    that need to be swapped to bring the concatenated list into canonical order.
    This can be computed by counting the number of set bits of mask_a that lie
    to the left of each set bit of mask_b.
    """
    sign = 1
    while mask_b:
        lowest = mask_b & -mask_b          # isolate lowest set bit of b
        # count bits of a that are lower than this bit
        if (mask_a & ((lowest << 1) - 1)).bit_count() % 2:
            sign = -sign
        mask_b ^= lowest
    return sign


def _geometric_product(mask_a: int, mask_b: int) -> Tuple[int, int]:
    """
    Returns (result_mask, sign) of the geometric product of two basis blades.
    Identical basis vectors cancel (e_i*e_i = 1) which corresponds to XOR of masks.
    The sign is given by _blade_sign.
    """
    sign = _blade_sign(mask_a, mask_b)
    result_mask = mask_a ^ mask_b
    return result_mask, sign


# ----------------------------------------------------------------------
# Multivector class
# ----------------------------------------------------------------------
class Multivector:
    """
    Sparse multivector where keys are integer bit‑masks of basis blades
    and values are the corresponding scalar coefficients.
    """
    __slots__ = ("blades",)

    def __init__(self, blades: Dict[int, float] = None):
        self.blades: Dict[int, float] = dict(blades) if blades else {}

    # ---- basic arithmetic ------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = self.blades.copy()
        for m, c in other.blades.items():
            result[m] = result.get(m, 0.0) + c
            if abs(result[m]) < 1e-15:
                del result[m]
        return Multivector(result)

    def __sub__(self, other: "Multivector") -> "Multivector":
        result = self.blades.copy()
        for m, c in other.blades.items():
            result[m] = result.get(m, 0.0) - c
            if abs(result[m]) < 1e-15:
                del result[m]
        return Multivector(result)

    def __neg__(self) -> "Multivector":
        return Multivector({m: -c for m, c in self.blades.items()})

    def __rmul__(self, scalar: float) -> "Multivector":
        return Multivector({m: scalar * c for m, c in self.blades.items()})

    def __mul__(self, other: Any) -> "Multivector":
        if isinstance(other, (int, float)):
            return other * self
        if not isinstance(other, Multivector):
            raise TypeError("Multiplication only defined between Multivectors or scalars")
        result: Dict[int, float] = {}
        for mask_a, coeff_a in self.blades.items():
            for mask_b, coeff_b in other.blades.items():
                mask_res, sign = _geometric_product(mask_a, mask_b)
                coeff = coeff_a * coeff_b * sign
                result[mask_res] = result.get(mask_res, 0.0) + coeff
                if abs(result[mask_res]) < 1e-15:
                    del result[mask_res]
        return Multivector(result)

    # ---- norm and distance ------------------------------------------------
    def norm(self) -> float:
        """Euclidean norm of the multivector (sqrt of sum of squares)."""
        return math.sqrt(sum(c * c for c in self.blades.values()))

    def distance(self, other: "Multivector") -> float:
        """Geometric distance defined as ||self - other||."""
        return (self - other).norm()

    # ---- representation ---------------------------------------------------
    def __repr__(self) -> str:
        if not self.blades:
            return "0"
        terms = []
        for mask, coeff in sorted(self.blades.items()):
            if mask == 0:
                blade = "1"
            else:
                indices = [i for i in range(mask.bit_length()) if mask & (1 << i)]
                blade = "e" + "^".join(str(i) for i in indices)
            terms.append(f"{coeff:.3g}{blade}")
        return " + ".join(terms)


# ----------------------------------------------------------------------
# Helper functions for the hybrid algorithm
# ----------------------------------------------------------------------
def broadcast_probability(phases: int, phase: int) -> float:
    """
    Probability that decays exponentially with the remaining phases.
    Clamped to the interval [0, 1].
    """
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive integers")
    exponent = max(0, phases - phase)
    return min(1.0, 1.0 / (2 ** exponent))


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """
    Classic exponential cooling schedule: T_k = t0 * alpha^k.
    """
    if k < 0:
        raise ValueError("iteration index k must be non‑negative")
    if t0 < 0 or not (0.0 < alpha <= 1.0):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


def hybrid_temperature(phases: int, phase: int, t0: float = 1.0,
                       alpha: float = 0.95, beta: float = 0.5) -> float:
    """
    A deeper integration of the two decay mechanisms:
        T = (t0 * alpha^{phase-1}) * (p)^{beta}
    where p = broadcast_probability.  The exponent beta lets the user
    control how strongly the broadcast probability influences the temperature.
    """
    p = broadcast_probability(phases, phase)
    T_exp = cooling_temperature(phase - 1, t0, alpha)
    return T_exp * (p ** beta)


# ----------------------------------------------------------------------
# Mapping integers to basis vectors (simple 1‑to‑1 embedding)
# ----------------------------------------------------------------------
def basis_vector(idx: int) -> Multivector:
    """
    Returns the unit vector e_idx as a multivector.
    idx must be non‑negative and small enough to fit in a Python int.
    """
    if idx < 0:
        raise ValueError("basis index must be non‑negative")
    return Multivector({1 << idx: 1.0})


# ----------------------------------------------------------------------
# Hybrid leader election using simulated annealing
# ----------------------------------------------------------------------
def node_energy(node: Any, graph: Dict[Any, Set[Any]],
                assignments: Dict[int, Any]) -> float:
    """
    Energy of a node is defined as the sum of distances from all points
    currently assigned to it.  Smaller energy is better (more compact cluster).
    """
    total = 0.0
    node_mv = basis_vector(node) if isinstance(node, int) else None
    for pt, assigned in assignments.items():
        if assigned == node:
            total += basis_vector(pt).distance(basis_vector(node))
    return total


def hybrid_leader_election(graph: Dict[Any, Set[Any]],
                           phases: int, t0: float, alpha: float,
                           assignments: Dict[int, Any]) -> Any:
    """
    Simulated annealing over the graph nodes to minimise the energy defined
    by node_energy.  The current temperature follows hybrid_temperature().
    """
    current = random.choice(list(graph.keys()))
    best = current
    best_energy = node_energy(best, graph, assignments)

    for phase in range(1, phases + 1):
        T = hybrid_temperature(phases, phase, t0, alpha)
        neighbor = random.choice(list(graph[current]))
        delta = node_energy(neighbor, graph, assignments) - node_energy(current, graph, assignments)

        # Metropolis acceptance
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
            current = neighbor
            if node_energy(current, graph, assignments) < best_energy:
                best = current
                best_energy = node_energy(best, graph, assignments)

    return best


# ----------------------------------------------------------------------
# Hybrid ternary route: assign points to nearest graph node using Clifford geometry
# ----------------------------------------------------------------------
def hybrid_ternary_route(graph: Dict[Any, Set[Any]],
                         points: Iterable[int],
                         phases: int,
                         t0: float = 1.0,
                         alpha: float = 0.95) -> Tuple[Dict[int, Any], Any]:
    """
    For each integer point, construct its unit basis vector and assign it to the
    graph node whose basis vector is geometrically closest (using the norm of
    the difference).  After all assignments, run the hybrid leader election.
    """
    # Pre‑compute multivectors for graph nodes that are integers; otherwise fall back to 0‑blade.
    node_mv: Dict[Any, Multivector] = {}
    for node in graph.keys():
        if isinstance(node, int):
            node_mv[node] = basis_vector(node)
        else:
            # Non‑integer identifiers get the scalar 1 (grade‑0) as a placeholder.
            node_mv[node] = Multivector({0: 1.0})

    assignments: Dict[int, Any] = {}
    for pt in points:
        pt_mv = basis_vector(pt)
        # Find nearest node by geometric distance
        nearest = min(node_mv.items(),
                      key=lambda item: pt_mv.distance(item[1]))[0]
        assignments[pt] = nearest

    leader = hybrid_leader_election(graph, phases, t0, alpha, assignments)
    return assignments, leader


# ----------------------------------------------------------------------
# Example execution (can be removed or guarded by __name__ == "__main__")
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple undirected graph (node identifiers are integers for direct embedding)
    graph = {
        0: {1, 2},
        1: {0, 3, 4},
        2: {0, 5},
        3: {1},
        4: {1, 5},
        5: {2, 4}
    }
    points = [10, 11, 12, 13, 14]   # arbitrary point identifiers (treated as basis indices)
    phases = 8
    t0 = 1.0
    alpha = 0.93

    assignments, leader = hybrid_ternary_route(graph, points, phases, t0, alpha)
    print("Assignments:", assignments)
    print("Leader node:", leader)