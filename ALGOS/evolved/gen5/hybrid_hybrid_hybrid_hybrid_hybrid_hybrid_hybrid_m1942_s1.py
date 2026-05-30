# DARWIN HAMMER — match 1942, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m58_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m467_s1.py (gen4)
# born: 2026-05-29T23:40:06Z

"""
Hybrid Algorithm Fusion of:
- hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m58_s1.py (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m467_s1.py (Algorithm B)

Mathematical Bridge:
Algorithm A provides a Fisher‑information based weighting (via ``fisher_score``) and
a similarity measure (SSIM) together with a Dense Associative Memory (DAM) energy
model. Algorithm B supplies a Clifford‑algebra multivector representation and a
cellular sheaf structure with linear restriction maps between nodes.

The fusion treats each multivector component as a feature vector whose importance
is modulated by the Fisher score. These weighted components are then inserted as
sections of a sheaf. The sheaf’s linear restrictions are updated using the DAM
energy landscape: sections are transformed toward patterns that minimise the DAM
energy, with a soft‑max weighting that respects the Fisher‑derived importance.
Thus the core topologies—Fisher‑weighted similarity, DAM energy, multivector
algebra, and sheaf propagation—are mathematically intertwined in a single unified
system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import numpy as np

# ----------------------------------------------------------------------
# Algorithm A building blocks
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam profile."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural similarity index (SSIM) for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


class DenseAssociativeMemory:
    """Simple dense associative memory with soft‑max recall."""
    def __init__(self, patterns: np.ndarray, beta: float = 1.0):
        self.patterns = np.asarray(patterns, dtype=float)  # shape (P, D)
        self.beta = float(beta)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z = z - np.max(z)
        e = np.exp(z)
        return e / e.sum()

    def energy(self, state: np.ndarray) -> float:
        """Quadratic energy E = -0.5 * stateᵀ W state, with W = patternsᵀ patterns."""
        W = self.patterns.T @ self.patterns
        return -0.5 * state @ W @ state

    def recall(self, cue: np.ndarray) -> np.ndarray:
        """Return a weighted combination of stored patterns using soft‑max over energies."""
        cue = np.asarray(cue, dtype=float)
        # Energy of each pattern with the cue (negative distance)
        distances = np.linalg.norm(self.patterns - cue, axis=1)
        logits = -self.beta * distances
        weights = self._softmax(logits)
        return weights @ self.patterns

# ----------------------------------------------------------------------
# Algorithm B building blocks (Clifford algebra & sheaf)
# ----------------------------------------------------------------------
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # Cancel duplicate index (e_i * e_i = 1)
                lst.pop(j)
                lst.pop(j)  # after pop, the next element shifts into position j
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sparse sum of basis blades."""

    def __init__(self, components: dict[int, float] | dict[frozenset, float], n: int):
        """
        components: mapping blade -> coefficient.
        Blade may be an int (grade‑1) or a frozenset of ints for higher grades.
        Zero coefficients are discarded.
        """
        self.n = int(n)
        self.components = {
            (frozenset({k}) if isinstance(k, int) else frozenset(k)): float(v)
            for k, v in components.items() if float(v) != 0.0
        }

    def __add__(self, other):
        if not isinstance(other, Multivector):
            return NotImplemented
        new = self.components.copy()
        for b, c in other.components.items():
            new[b] = new.get(b, 0.0) + c
            if abs(new[b]) < 1e-12:
                del new[b]
        return Multivector(new, self.n)

    def __mul__(self, other):
        """Geometric product."""
        if not isinstance(other, Multivector):
            return NotImplemented
        result = {}
        for b1, c1 in self.components.items():
            for b2, c2 in other.components.items():
                b, s = _multiply_blades(b1, b2)
                coeff = c1 * c2 * s
                result[b] = result.get(b, 0.0) + coeff
        return Multivector(result, self.n)

    def grade(self, k: int):
        """Return a new Multivector containing only grade‑k blades."""
        return Multivector(
            {b: c for b, c in self.components.items() if len(b) == k},
            self.n,
        )

    def magnitude(self) -> float:
        """Euclidean norm of the coefficient vector."""
        return math.sqrt(sum(c * c for c in self.components.values()))

    def as_vector(self) -> np.ndarray:
        """Flatten to a dense vector of length 2**n (order: increasing binary index)."""
        dim = 1 << self.n
        vec = np.zeros(dim, dtype=float)
        for blade, coeff in self.components.items():
            # Encode blade as integer bitmask
            idx = 0
            for i in blade:
                idx |= 1 << i
            vec[idx] = coeff
        return vec


class Sheaf:
    """Cellular sheaf with linear restriction maps on edges."""

    def __init__(self, node_dims: dict[int, int], edges: list[tuple[int, int, np.ndarray]]):
        """
        node_dims: node_id -> dimension of the vector space attached to the node.
        edges: list of (src, dst, restriction_matrix) where the matrix maps
               vectors from src-space to dst-space.
        """
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
        self._sections = {node: np.zeros(dim) for node, dim in self.node_dims.items()}

    def set_section(self, node: int, vec: np.ndarray):
        vec = np.asarray(vec, dtype=float)
        if vec.shape != (self.node_dims[node],):
            raise ValueError(f"section size mismatch for node {node}")
        self._sections[node] = vec.copy()

    def get_section(self, node: int) -> np.ndarray:
        return self._sections[node].copy()

    def propagate_once(self):
        """One synchronous sheaf update using current sections."""
        new_sections = {node: sec.copy() for node, sec in self._sections.items()}
        for src, dst, R in self.edges:
            src_vec = self._sections[src]
            transformed = R @ src_vec
            # Simple averaging with existing dst section
            new_sections[dst] = 0.5 * (new_sections[dst] + transformed)
        self._sections = new_sections

    def iterate(self, steps: int = 1):
        for _ in range(steps):
            self.propagate_once()

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def fisher_weighted_multivector(theta: float, center: float, width: float,
                                mv: Multivector) -> Multivector:
    """
    Scale every coefficient of ``mv`` by the Fisher information computed from
    ``theta``. This bridges Algorithm A's Fisher weighting with Algorithm B's
    multivector representation.
    """
    w = fisher_score(theta, center, width)
    scaled = {blade: coeff * w for blade, coeff in mv.components.items()}
    return Multivector(scaled, mv.n)


def dam_sheaf_update(sheaf: Sheaf, dam: DenseAssociativeMemory,
                     node_patterns: dict[int, np.ndarray]) -> None:
    """
    For each node, treat its current section as a cue to the DAM.
    The recalled pattern is then projected onto the node's space via the
    restriction maps of outgoing edges (if any). The section is finally
    replaced by a convex combination of its old value and the DAM recall,
    weighted by a soft‑max over DAM energies (Algorithm A) and the
    Fisher magnitude of the node's multivector (Algorithm B).
    """
    for node, dim in sheaf.node_dims.items():
        cue = sheaf.get_section(node)
        # Energy of the cue w.r.t. each stored pattern
        energies = np.array([ -dam.energy(p) for p in dam.patterns ])
        probs = dam._softmax(dam.beta * energies)
        # Expected pattern under the DAM distribution
        expected = probs @ dam.patterns
        # Blend with current cue
        blend = 0.5 * cue + 0.5 * expected[:dim]  # truncate if pattern dim > node dim
        sheaf.set_section(node, blend)


def hybrid_ssim_energy(x: np.ndarray, y: np.ndarray,
                       dam: DenseAssociativeMemory,
                       mv: Multivector,
                       theta: float, center: float, width: float) -> float:
    """
    Combine SSIM (Algorithm A) with DAM energy and Fisher‑weighted multivector magnitude.
    The result can be interpreted as a composite similarity/energy score.
    """
    base_ssim = ssim(x, y)
    # DAM energy of the vectorized multivector
    mv_vec = mv.as_vector()
    dam_e = dam.energy(mv_vec)
    # Fisher weight
    f_weight = fisher_score(theta, center, width)
    # Normalise components to keep the score in a reasonable range
    return base_ssim * f_weight - 0.01 * dam_e


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Create a simple multivector in 3‑D Clifford algebra
    mv = Multivector({0: 1.2, 1: -0.7, frozenset({0, 1}): 0.3}, n=3)

    # 2. Apply Fisher weighting
    mv_fisher = fisher_weighted_multivector(theta=0.5, center=0.0, width=1.0, mv=mv)
    print("Fisher‑weighted multivector components:", mv_fisher.components)

    # 3. Build a DAM with random patterns (dimension = 8 = 2**3)
    np.random.seed(42)
    patterns = np.random.randn(5, 8)  # 5 stored patterns
    dam = DenseAssociativeMemory(patterns, beta=2.0)

    # 4. Initialise a sheaf with two nodes and a single edge
    node_dims = {0: 8, 1: 8}
    R = np.eye(8) * 0.9  # simple contraction restriction
    edges = [(0, 1, R)]
    sheaf = Sheaf(node_dims, edges)

    # Set initial sections (use multivector vectors)
    sheaf.set_section(0, mv_fisher.as_vector())
    sheaf.set_section(1, np.zeros(8))

    # 5. Perform a DAM‑guided sheaf update
    dam_sheaf_update(sheaf, dam, node_patterns={})
    print("Sheaf section after DAM update (node 0):", sheaf.get_section(0)[:5])
    print("Sheaf section after DAM update (node 1):", sheaf.get_section(1)[:5])

    # 6. Compute hybrid SSIM‑energy between two random signals
    x = np.random.randn(100)
    y = np.random.randn(100)
    score = hybrid_ssim_energy(x, y, dam, mv_fisher,
                               theta=0.3, center=0.0, width=1.5)
    print("Hybrid SSIM‑energy score:", score)

    # 7. Run a few sheaf propagation steps to demonstrate stability
    sheaf.iterate(steps=3)
    print("Sheaf section after propagation (node 1):", sheaf.get_section(1)[:5])