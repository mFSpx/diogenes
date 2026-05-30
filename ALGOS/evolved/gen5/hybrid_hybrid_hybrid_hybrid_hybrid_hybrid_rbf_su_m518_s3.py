# DARWIN HAMMER — match 518, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_physarum_netw_m334_s1.py (gen4)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_jepa_energy_h_m89_s0.py (gen4)
# born: 2026-05-29T23:29:15Z

"""Hybrid Physarum‑RBF Algorithm
Parent A: hybrid_hybrid_hybrid_geomet_hybrid_physarum_netw_m334_s1.py
Parent B: hybrid_hybrid_rbf_surrogate_hybrid_jepa_energy_h_m89_s0.py

Mathematical bridge:
The physarum network state is encoded as a multivector **C** = Σ g_i e_i,
where g_i are edge conductances and e_i are orthogonal basis vectors of a
Clifford algebra.  The surrogate model of Parent B provides a scalar
functional 𝔈(**C**) ≈ free‑energy of the network by evaluating a radial‑basis
function (RBF) on the conductance vector (the scalar part of **C**).  The
gradient of 𝔈 w.r.t. conductances is obtained via the inner product
⟨∂𝔈/∂**C**, e_i⟩, which yields a real number for each edge.  This gradient
is fused with the flux‑based physarum update to obtain a hybrid rule

    g_i ← g_i + η ( Φ_i – λ ∂𝔈/∂g_i ),

where Φ_i is the physarum flux on edge i, η a learning rate and λ a
coupling constant.  The code below implements this unified system.
"""

import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple, Sequence
import numpy as np

# ----------------------------------------------------------------------
# Multivector (geometric algebra) – core of Parent A
# ----------------------------------------------------------------------
class Multivector:
    """Sparse multivector in a Clifford algebra with n basis vectors."""
    def __init__(self, components: Dict[frozenset[int], float], n: int):
        self.n = int(n)
        # discard near‑zero entries for sparsity
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    def grade(self, k: int) -> "Multivector":
        """Return the grade‑k part."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) component."""
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            label = "1" if not blade else "e" + "".join(str(i) for i in sorted(blade))
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if abs(v) > 1e-15}, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        neg = Multivector({k: -v for k, v in other.components.items()}, other.n)
        return self.__add__(neg)

    def inner_product(self, other: "Multivector") -> float:
        """Return the scalar inner product ⟨self, other⟩ = Σ a_i b_i."""
        # Because basis vectors are orthonormal, the inner product reduces to
        # component‑wise multiplication summed over matching blades.
        total = 0.0
        for blade, a in self.components.items():
            b = other.components.get(blade, 0.0)
            total += a * b
        return total

# ----------------------------------------------------------------------
# Radial‑Basis Function surrogate – core of Parent B
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

class RBFSurrogate:
    """Simple RBF surrogate model."""
    def __init__(self, centers: List[Tuple[float, ...]], weights: List[float], epsilon: float = 1.0):
        if len(centers) != len(weights):
            raise ValueError("centers and weights must have same length")
        self.centers = centers
        self.weights = weights
        self.epsilon = float(epsilon)

    def predict(self, x: Sequence[float]) -> float:
        """Predict scalar output for input vector x."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

# ----------------------------------------------------------------------
# Physarum network utilities (flux, potentials, Laplacian)
# ----------------------------------------------------------------------
def build_incidence_matrix(num_nodes: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """Return the signed node‑edge incidence matrix B (num_nodes × num_edges)."""
    B = np.zeros((num_nodes, len(edges)), dtype=float)
    for idx, (i, j) in enumerate(edges):
        B[i, idx] = 1.0
        B[j, idx] = -1.0
    return B

def laplacian_from_conductances(G: np.ndarray, B: np.ndarray) -> np.ndarray:
    """L = B·G·Bᵀ where G is diagonal matrix of edge conductances."""
    return B @ G @ B.T

def solve_potentials(L: np.ndarray, source: int, sink: int) -> np.ndarray:
    """Solve L·v = b with Dirichlet condition v_sink = 0."""
    n = L.shape[0]
    b = np.zeros(n)
    b[source] = 1.0
    b[sink] = -1.0
    # Remove the sink row/col to enforce v_sink = 0 (ground)
    mask = np.arange(n) != sink
    L_reduced = L[np.ix_(mask, mask)]
    b_reduced = b[mask]
    v_reduced = np.linalg.solve(L_reduced, b_reduced)
    v = np.zeros(n)
    v[mask] = v_reduced
    v[sink] = 0.0
    return v

def edge_fluxes(G_diag: np.ndarray, B: np.ndarray, potentials: np.ndarray) -> np.ndarray:
    """Compute flux Φ = G·Bᵀ·v (signed according to edge orientation)."""
    return G_diag @ (B.T @ potentials)

# ----------------------------------------------------------------------
# Hybrid operations tying together both parents
# ----------------------------------------------------------------------
def multivector_from_conductances(g: np.ndarray) -> Multivector:
    """Encode conductance vector g as a grade‑1 multivector Σ g_i e_i."""
    components = {frozenset({i}): float(val) for i, val in enumerate(g)}
    return Multivector(components, n=len(g))

def surrogate_energy_from_state(surrogate: RBFSurrogate, g: np.ndarray) -> float:
    """Treat the conductance vector as the feature vector for the RBF surrogate."""
    return surrogate.predict(g.tolist())

def hybrid_conductance_update(
    g: np.ndarray,
    flux: np.ndarray,
    energy: float,
    eta: float = 0.1,
    lam: float = 0.05,
) -> np.ndarray:
    """
    Perform the hybrid update:
        g_i ← g_i + η ( Φ_i – λ ∂𝔈/∂g_i )
    The gradient ∂𝔈/∂g_i is approximated by finite differences on the
    surrogate (since the surrogate is smooth, a central difference works).
    """
    eps = 1e-6
    grad = np.zeros_like(g)
    for i in range(len(g)):
        g_plus = g.copy()
        g_minus = g.copy()
        g_plus[i] += eps
        g_minus[i] -= eps
        e_plus = surrogate_energy_from_state(global_surrogate, g_plus)
        e_minus = surrogate_energy_from_state(global_surrogate, g_minus)
        grad[i] = (e_plus - e_minus) / (2 * eps)

    g_new = g + eta * (flux - lam * grad)
    # Keep conductances positive
    g_new = np.clip(g_new, 1e-6, None)
    return g_new

# Global surrogate placeholder – will be instantiated in __main__
global_surrogate: RBFSurrogate

def hybrid_step(
    g: np.ndarray,
    B: np.ndarray,
    source: int,
    sink: int,
    eta: float = 0.1,
    lam: float = 0.05,
) -> Tuple[np.ndarray, float]:
    """
    Execute one simulation step:
      1. Build Laplacian L from conductances.
      2. Solve node potentials.
      3. Compute edge fluxes.
      4. Evaluate surrogate energy.
      5. Update conductances via hybrid rule.
    Returns the updated conductances and the scalar energy.
    """
    G_diag = np.diag(g)
    L = laplacian_from_conductances(G_diag, B)
    potentials = solve_potentials(L, source, sink)
    flux = edge_fluxes(G_diag, B, potentials)
    energy = surrogate_energy_from_state(global_surrogate, g)
    g_next = hybrid_conductance_update(g, flux, energy, eta, lam)
    return g_next, energy

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny triangular network
    num_nodes = 3
    edges = [(0, 1), (1, 2), (0, 2)]
    B = build_incidence_matrix(num_nodes, edges)

    # Initialise conductances (positive values)
    g = np.array([0.5, 0.5, 0.5], dtype=float)

    # Create a random RBF surrogate with 5 centers in 3‑D conductance space
    dim = len(g)
    random.seed(42)
    centers = [tuple(random.uniform(0.1, 1.0) for _ in range(dim)) for _ in range(5)]
    weights = [random.uniform(-1.0, 1.0) for _ in range(5)]
    global_surrogate = RBFSurrogate(centers, weights, epsilon=2.0)

    source_node = 0
    sink_node = 2

    print("Initial conductances:", g)
    for step in range(10):
        g, energy = hybrid_step(g, B, source_node, sink_node, eta=0.2, lam=0.1)
        print(f"Step {step+1:02d} | Energy: {energy:.4f} | Conductances: {g}")

    # Show multivector representation of final state
    mv = multivector_from_conductances(g)
    print("Final multivector:", mv)