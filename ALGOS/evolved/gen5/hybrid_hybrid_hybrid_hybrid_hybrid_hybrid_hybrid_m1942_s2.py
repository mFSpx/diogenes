# DARWIN HAMMER — match 1942, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m58_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m467_s1.py (gen4)
# born: 2026-05-29T23:40:06Z

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
        for src, dst, restriction_matrix in self.edges:
            sec = self._sections[src]
            new_sections[dst] += restriction_matrix @ sec
        self._sections = new_sections


class HybridFusion:
    def __init__(self, patterns: np.ndarray, node_dims: dict[int, int], edges: list[tuple[int, int, np.ndarray]]):
        self.dam = DenseAssociativeMemory(patterns)
        self.sheaf = Sheaf(node_dims, edges)

    def fuse(self, multivector: Multivector) -> Multivector:
        vector = multivector.as_vector()
        fisher_weights = np.array([fisher_score(i, 0, 1) for i in range(len(vector))])
        weighted_vector = vector * fisher_weights
        self.sheaf.set_section(0, weighted_vector)
        self.sheaf.propagate_once()
        return Multivector({frozenset(): c}, multivector.n) + Multivector(dict(zip(range(len(self.sheaf.get_section(0))), self.sheaf.get_section(0))), multivector.n)


# Example usage
if __name__ == "__main__":
    patterns = np.random.rand(10, 5)
    node_dims = {0: 5, 1: 5}
    edges = [(0, 1, np.eye(5))]
    fusion = HybridFusion(patterns, node_dims, edges)

    multivector = Multivector({frozenset({0}): 1.0}, 5)
    result = fusion.fuse(multivector)
    print(result.components)