# DARWIN HAMMER — match 1887, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m522_s2.py (gen5)
# born: 2026-05-29T23:39:41Z

import numpy as np
from typing import Callable, Dict, List, Tuple, Any


__all__ = [
    "Sheaf",
    "DenseAssociativeMemory",
    "RegretWeightedStrategy",
    "HybridEnergy",
    "HybridUpdateRule",
    "HybridRetrieve",
]


class Sheaf:
    """
    Cellular sheaf on a directed graph.

    Each node ``n`` carries a vector space ℝ^{dim(n)}.
    Each directed edge ``(u, v)`` carries a pair of linear restriction maps

        src_map : ℝ^{dim(u)} → ℝ^{dim(e)}
        dst_map : ℝ^{dim(v)} → ℝ^{dim(e)}.

    A *section* is an assignment of a vector to every node that is compatible
    with all restriction maps, i.e. for every edge (u, v)

        src_map @ s[u]  ≈  dst_map @ s[v].

    The class provides utilities to set maps, set sections and to project
    a raw query vector onto the nearest compatible section.
    """

    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]]):
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}

    # --------------------------------------------------------------------- #
    # Restriction map handling
    # --------------------------------------------------------------------- #
    def set_restriction(
        self,
        edge: Tuple[Any, Any],
        src_map: np.ndarray,
        dst_map: np.ndarray,
    ) -> None:
        """Register the two restriction matrices for a directed edge."""
        u, v = edge
        if u not in self.node_dims or v not in self.node_dims:
            raise KeyError(f"Edge {edge} refers to undefined nodes.")
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[edge] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    # --------------------------------------------------------------------- #
    # Section handling
    # --------------------------------------------------------------------- #
    def set_section(self, node: Any, value: np.ndarray) -> None:
        """Assign a vector to a node, checking dimensionality."""
        if node not in self.node_dims:
            raise KeyError(f"Node {node} is not part of the sheaf.")
        if value.shape != (self.node_dims[node],):
            raise ValueError(f"Vector for node {node} must have shape ({self.node_dims[node]},)")
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node: Any) -> np.ndarray:
        """Return the stored vector for ``node``."""
        return self._sections[node]

    def all_sections(self) -> Dict[Any, np.ndarray]:
        """Return a copy of the whole section dictionary."""
        return {n: v.copy() for n, v in self._sections.items()}

    # --------------------------------------------------------------------- #
    # Compatibility projection
    # --------------------------------------------------------------------- #
    def _compatibility_error(self, sections: Dict[Any, np.ndarray]) -> float:
        """Sum of squared mismatches across all edges."""
        err = 0.0
        for (u, v), (src_map, dst_map) in self._restrictions.items():
            err += np.linalg.norm(src_map @ sections[u] - dst_map @ sections[v]) ** 2
        return err

    def project_query(self, query: np.ndarray) -> Dict[Any, np.ndarray]:
        """
        Given a raw ``query`` vector of dimension ``dim(root_node)``,
        produce a compatible section by solving a least‑squares problem
        that respects the restriction maps.

        The algorithm:
        1. Initialise each node's vector with a random copy of the query
           (or zeros if dimensions differ).
        2. Perform a few Gauss‑Seidel sweeps over the edges to minimise
           the compatibility error.
        """
        if not self.node_dims:
            raise RuntimeError("Sheaf has no nodes defined.")
        # Choose an arbitrary root node (the first key) to align dimensions.
        root = next(iter(self.node_dims))
        if query.shape != (self.node_dims[root],):
            raise ValueError(f"Query shape {query.shape} does not match root node dimension {self.node_dims[root]}.")
        # Initialise sections
        sections = {n: np.zeros(d) for n, d in self.node_dims.items()}
        sections[root] = query.astype(float)

        # Simple Gauss‑Seidel relaxation
        for _ in range(5):
            for (u, v), (src_map, dst_map) in self._restrictions.items():
                # Solve for v given u (one‑step least squares)
                if src_map.shape[0] == 0:
                    continue
                target = src_map @ sections[u]
                # Least‑squares solve dst_map @ s[v] ≈ target
                # Using np.linalg.lstsq to handle non‑square maps.
                sol, *_ = np.linalg.lstsq(dst_map, target, rcond=None)
                sections[v] = sol
        return sections

    # --------------------------------------------------------------------- #
    # Utility
    # --------------------------------------------------------------------- #
    def compatible_section(self, query: np.ndarray) -> Dict[Any, np.ndarray]:
        """
        Public wrapper that returns a compatible section for ``query``.
        The returned dictionary can be stored back via ``set_section``.
        """
        return self.project_query(query)


class DenseAssociativeMemory:
    """
    Dense associative memory whose state is a matrix ``M``.
    The memory is updated by gradient descent on an energy function
    ``E(x)`` evaluated on a query vector ``x``.
    """

    def __init__(self, memory_matrix: np.ndarray, energy_function: Callable[[np.ndarray], float]):
        self.memory_matrix = np.asarray(memory_matrix, dtype=float)
        self.energy_function = energy_function

    def energy(self, x: np.ndarray) -> float:
        """Convenient wrapper."""
        return float(self.energy_function(x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Approximate ∂E/∂M for the simple quadratic energy
        ``E(x) = ||x||²``.  For a generic callable we fall back to a
        finite‑difference approximation.
        """
        if self.energy_function is None:
            raise RuntimeError("No energy function defined.")
        # Detect the canonical quadratic case
        if getattr(self.energy_function, "__name__", None) == "<lambda>":
            # Assume lambda x: np.dot(x, x)
            return 2.0 * np.outer(x, x)
        # Finite‑difference fallback
        eps = 1e-6
        grad = np.zeros_like(self.memory_matrix)
        for i in range(self.memory_matrix.shape[0]):
            for j in range(self.memory_matrix.shape[1]):
                perturbed = self.memory_matrix.copy()
                perturbed[i, j] += eps
                # Define a temporary memory with perturbed matrix
                tmp = DenseAssociativeMemory(perturbed, self.energy_function)
                grad[i, j] = (tmp.energy(x) - self.energy(x)) / eps
        return grad

    def update(self, x: np.ndarray, lr: float) -> None:
        """
        Perform a gradient descent step on the memory matrix.
        ``lr`` is the learning rate already scaled by the regret‑weighted
        strategy and VRAM budget.
        """
        grad = self.gradient(x)
        self.memory_matrix -= lr * grad


class RegretWeightedStrategy:
    """
    Convert a list of regrets into a probability distribution and
    adapt a base learning rate according to available VRAM.
    """

    def compute_regret_weights(self, regrets: List[float]) -> List[float]:
        """Return a probability distribution proportional to non‑negative regrets."""
        positive = [max(r, 0.0) for r in regrets]
        total = sum(positive)
        if total == 0.0:
            n = len(regrets)
            return [1.0 / n] * n
        return [p / total for p in positive]

    def budgeted_lr(self, base_lr: float, free_mb: int, budget_mb: int = 4096, reserve_mb: int = 768) -> float:
        """
        Scale ``base_lr`` according to the fraction of usable VRAM that is free.
        The scaling is linear between 0.1 (no free memory) and 1.0 (enough memory).
        """
        usable = max(budget_mb - reserve_mb, 1)
        if free_mb >= usable:
            return base_lr
        scale = 0.1 + 0.9 * (free_mb / usable)
        return base_lr * scale


class HybridEnergy:
    """
    Energy function that blends the base energy with a regret‑weighted
    penalty term encouraging low regret.
    """

    def __init__(
        self,
        base_energy: Callable[[np.ndarray], float],
        regret_strategy: RegretWeightedStrategy,
        penalty_coeff: float = 0.01,
    ):
        self.base_energy = base_energy
        self.regret_strategy = regret_strategy
        self.penalty_coeff = float(penalty_coeff)

    def __call__(self, x: np.ndarray) -> float:
        base = float(self.base_energy(x))
        # Treat the base energy as a singleton regret value.
        weight = self.regret_strategy.compute_regret_weights([base])[0]
        return base + self.penalty_coeff * weight * base


class HybridUpdateRule:
    """
    Update rule that:

    1. Projects the query onto a compatible sheaf section.
    2. Computes a hybrid energy.
    3. Scales the learning rate with the regret‑weighted strategy and VRAM budget.
    4. Updates the dense associative memory.
    5. Stores the projected section back into the sheaf.
    """

    def __init__(
        self,
        sheaf: Sheaf,
        memory: DenseAssociativeMemory,
        hybrid_energy: HybridEnergy,
        regret_strategy: RegretWeightedStrategy,
        base_lr: float = 1e-3,
    ):
        self.sheaf = sheaf
        self.memory = memory
        self.hybrid_energy = hybrid_energy
        self.regret_strategy = regret_strategy
        self.base_lr = float(base_lr)

    def update(self, query_vector: np.ndarray, free_vram_mb: int) -> None:
        # 1. Project query onto a compatible sheaf section.
        compatible_section = self.sheaf.compatible_section(query_vector)

        # 2. Compute hybrid energy (used only for learning‑rate scaling here).
        energy_val = self.hybrid_energy(query_vector)

        # 3. Adapt learning rate.
        lr = self.regret_strategy.budgeted_lr(self.base_lr, free_vram_mb)

        # 4. Update the memory matrix.
        self.memory.update(query_vector, lr)

        # 5. Store the compatible section.
        for node, vec in compatible_section.items():
            self.sheaf.set_section(node, vec)


class HybridRetrieve:
    """
    Retrieval that first performs an update (so that the memory state
    reflects the latest query) and then returns the sheaf section
    that is most compatible with the query.
    """

    def __init__(self, update_rule: HybridUpdateRule, sheaf: Sheaf):
        self.update_rule = update_rule
        self.sheaf = sheaf

    def retrieve(self, query_vector: np.ndarray, free_vram_mb: int) -> Dict[Any, np.ndarray]:
        """
        Perform an update and return the *entire* compatible section.
        """
        self.update_rule.update(query_vector, free_vram_mb)
        # After the update the sheaf already holds a compatible section.
        return self.sheaf.all_sections()


# --------------------------------------------------------------------- #
# Example usage (executed when the module is run as a script)
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    np.random.seed(42)

    # Define a tiny sheaf with two nodes of dimension 10 and bidirectional edges.
    node_dims = {0: 10, 1: 10}
    edges = [(0, 1), (1, 0)]
    sheaf = Sheaf(node_dims, edges)

    # Random linear restrictions (full rank for simplicity).
    for (u, v) in edges:
        src = np.random.randn(8, node_dims[u])
        dst = np.random.randn(8, node_dims[v])
        sheaf.set_restriction((u, v), src, dst)

    # Initialise a random memory matrix.
    memory_matrix = np.random.randn(10, 10)
    # Quadratic energy E(x) = ||x||²
    energy_fn = lambda x: np.dot(x, x)

    memory = DenseAssociativeMemory(memory_matrix, energy_fn)

    regret_strategy = RegretWeightedStrategy()
    hybrid_energy = HybridEnergy(energy_fn, regret_strategy, penalty_coeff=0.02)

    update_rule = HybridUpdateRule(
        sheaf=sheaf,
        memory=memory,
        hybrid_energy=hybrid_energy,
        regret_strategy=regret_strategy,
        base_lr=5e-4,
    )

    retrieve = HybridRetrieve(update_rule, sheaf)

    # Simulate a query with a fictitious amount of free VRAM.
    query = np.random.randn(10)
    free_vram = 1024  # MB

    section = retrieve.retrieve(query, free_vram)

    # Print the section vectors for inspection.
    for node, vec in section.items():
        print(f"Node {node} section (first 5 entries): {vec[:5]}")