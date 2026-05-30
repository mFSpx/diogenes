# DARWIN HAMMER — match 32, survivor 4
# gen: 3
# parent_a: hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s4.py (gen2)
# parent_b: dense_associative_memory.py (gen0)
# born: 2026-05-29T23:25:19Z

import numpy as np

__all__ = [
    "Sheaf",
    "DenseAssociativeMemory",
    "hybrid_energy",
    "hybrid_update_rule",
    "hybrid_retrieve",
]

class Sheaf:
    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions = {}
        self._sections = {}

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    def set_section(self, node: any, value: np.ndarray) -> None:
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node: any):
        return self._sections[node]

    def get_restriction(self, edge: tuple):
        return self._restrictions[edge]


class DenseAssociativeMemory:
    def __init__(self, patterns: np.ndarray, beta: float = 1.0):
        self.patterns = np.asarray(patterns, dtype=float)
        self.beta = beta

    def _softmax(self, z: np.ndarray):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def _lse(self, z: np.ndarray):
        m = z.max()
        return m + np.log(np.exp(z - m).sum())

    def energy(self, xi: np.ndarray):
        xi = np.asarray(xi, dtype=float)
        scores = self.beta * (self.patterns @ xi)
        lse_term = self._lse(scores) / self.beta
        quadratic_term = 0.5 * xi @ xi
        return -np.log(self._softmax(scores)).sum() + lse_term + quadratic_term


def hybrid_energy(sheaf: Sheaf, dam: DenseAssociativeMemory):
    energy_values = []
    for node in sheaf.node_dims:
        if node in sheaf._sections:
            energy_value = dam.energy(sheaf.get_section(node))
            energy_values.append(energy_value)
    return np.mean(energy_values) if energy_values else 0


def hybrid_update_rule(sheaf: Sheaf, dam: DenseAssociativeMemory):
    for node in sheaf.node_dims:
        if node in sheaf._sections:
            updated_section = dam.patterns.T @ dam._softmax(dam.beta * (dam.patterns @ sheaf.get_section(node)))
            sheaf.set_section(node, updated_section)


def hybrid_retrieve(sheaf: Sheaf, dam: DenseAssociativeMemory, query: np.ndarray):
    closest_pattern = None
    min_distance = float('inf')
    for node in sheaf.node_dims:
        if node in sheaf._sections:
            distance = np.linalg.norm(query - sheaf.get_section(node))
            if distance < min_distance:
                min_distance = distance
                closest_pattern = sheaf.get_section(node)
    return closest_pattern


def improved_hybrid_energy(sheaf: Sheaf, dam: DenseAssociativeMemory):
    energy_values = []
    for node in sheaf.node_dims:
        if node in sheaf._sections:
            energy_value = dam.energy(sheaf.get_section(node))
            for edge in sheaf.edges:
                if node == edge[0]:
                    restriction = sheaf.get_restriction(edge)
                    src_map, dst_map = restriction
                    energy_value += np.linalg.norm(sheaf.get_section(node) - src_map @ sheaf.get_section(edge[1]))
            energy_values.append(energy_value)
    return np.mean(energy_values) if energy_values else 0


def improved_hybrid_update_rule(sheaf: Sheaf, dam: DenseAssociativeMemory):
    for node in sheaf.node_dims:
        if node in sheaf._sections:
            updated_section = dam.patterns.T @ dam._softmax(dam.beta * (dam.patterns @ sheaf.get_section(node)))
            for edge in sheaf.edges:
                if node == edge[0]:
                    restriction = sheaf.get_restriction(edge)
                    src_map, dst_map = restriction
                    updated_section += src_map @ sheaf.get_section(edge[1])
            sheaf.set_section(node, updated_section)


if __name__ == "__main__":
    node_dims = {'A': 2, 'B': 2}
    edges = [('A', 'B')]
    sheaf = Sheaf(node_dims, edges)
    sheaf.set_restriction(('A', 'B'), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]))
    sheaf.set_section('A', np.array([1, 0]))
    sheaf.set_section('B', np.array([0, 1]))

    patterns = np.array([[1, 0], [0, 1]])
    dam = DenseAssociativeMemory(patterns)

    hybrid_energy_value = hybrid_energy(sheaf, dam)
    hybrid_update_rule(sheaf, dam)
    retrieved_pattern = hybrid_retrieve(sheaf, dam, np.array([1, 0]))

    print("Hybrid Energy:", hybrid_energy_value)
    print("Retrieved Pattern:", retrieved_pattern)

    improved_hybrid_energy_value = improved_hybrid_energy(sheaf, dam)
    improved_hybrid_update_rule(sheaf, dam)
    improved_retrieved_pattern = hybrid_retrieve(sheaf, dam, np.array([1, 0]))

    print("Improved Hybrid Energy:", improved_hybrid_energy_value)
    print("Improved Retrieved Pattern:", improved_retrieved_pattern)