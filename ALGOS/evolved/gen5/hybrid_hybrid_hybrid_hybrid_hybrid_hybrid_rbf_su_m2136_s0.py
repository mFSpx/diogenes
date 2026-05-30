# DARWIN HAMMER — match 2136, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_caputo_fracti_m149_s0.py (gen4)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hybrid_pherom_m411_s2.py (gen3)
# born: 2026-05-29T23:40:58Z

"""
Hybrid Algorithm: Caputo Fractional Pheromones + Entropy‑Regularized RBF Surrogate

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_caputo_fracti_m149_s0.py (fractional Caputo decay of pheromones,
  pheromone‑influenced tree cost)
- hybrid_hybrid_rbf_surrogate_hybrid_hybrid_pherom_m411_s2.py (radial‑basis surrogate model
  regularized by entropy of pheromone signals)

Mathematical Bridge:
The pheromone signal evolves according to a Caputo fractional derivative, producing a
fractional decay kernel.  The instantaneous pheromone intensity on each edge is interpreted
as a probability distribution; its Shannon entropy is used as an additional regularization
term when fitting the RBF surrogate.  The surrogate then predicts a base edge weight from
node features, which is finally modulated by the fractional pheromone signal to obtain the
effective cost used in a minimum‑cost spanning tree (MST) computation.
"""

import math
import random
import sys
import pathlib
import numpy as np

# ----------------------------------------------------------------------
# 1. Fractional calculus utilities (from Parent A)
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of Γ(z) for z > 0."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

def caputo_derivative(f, alpha: float, t: float, dt: float = 0.01) -> float:
    """
    Caputo fractional derivative of order ``alpha`` of a scalar function ``f`` at time ``t``.
    ``f`` must accept a NumPy array and return an array of the same shape.
    """
    tau = np.arange(0, t, dt)
    f_tau = f(tau)
    integrand = f_tau / (t - tau) ** alpha
    integral = np.trapz(integrand, tau)
    return integral / gamma_lanczos(1 - alpha)

def fractional_decay(alpha: float, t: np.ndarray) -> np.ndarray:
    """Fractional decay kernel  t^{α‑1} / Γ(α)."""
    return t ** (alpha - 1) / gamma_lanczos(alpha)

def calculate_pheromone_signal(signal_value: float, half_life_seconds: float,
                               alpha: float) -> np.ndarray:
    """
    Returns the time‑evolution of a pheromone signal using a fractional decay kernel.
    The signal is sampled at 0.01 s intervals up to ``half_life_seconds``.
    """
    times = np.arange(0, half_life_seconds, 0.01) + 1e-12  # avoid t=0 singularity
    decay = fractional_decay(alpha, times)
    return signal_value * decay

# ----------------------------------------------------------------------
# 2. RBF surrogate with entropy regularisation (from Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def solve_linear(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b using Gaussian elimination (no external libs)."""
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1, 1)])
    for col in range(n):
        pivot = np.argmax(np.abs(M[col:, col])) + col
        if abs(M[pivot, col]) < 1e-12:
            raise np.linalg.LinAlgError("Singular matrix in surrogate solve")
        if pivot != col:
            M[[col, pivot]] = M[[pivot, col]]
        M[col] = M[col] / M[col, col]
        for row in range(n):
            if row == col:
                continue
            M[row] -= M[row, col] * M[col]
    return M[:, -1]

class RBFSurrogate:
    """
    Radial‑basis‑function surrogate.
    ``centers`` – list of feature vectors (shape (d,))
    ``weights`` – learned coefficients
    ``epsilon`` – shape parameter of the Gaussian RBF
    """
    def __init__(self, centers: list[np.ndarray], epsilon: float = 1.0):
        self.centers = [np.asarray(c, dtype=float) for c in centers]
        self.epsilon = epsilon
        self.weights: np.ndarray | None = None

    def _phi(self, x: np.ndarray) -> np.ndarray:
        """RBF feature vector Φ(x)."""
        return np.array([gaussian(euclidean(x, c), self.epsilon) for c in self.centers])

    def predict(self, x: np.ndarray) -> float:
        if self.weights is None:
            raise RuntimeError("Surrogate not fitted yet")
        phi = self._phi(x)
        return float(phi @ self.weights)

    def fit(self, X: list[np.ndarray], y: list[float],
            pheromone_signals: np.ndarray,
            lam: float = 0.1) -> None:
        """
        Fit the surrogate to data (X, y) with entropy regularisation.
        ``pheromone_signals`` is a 1‑D array of the same length as X,
        interpreted as a probability distribution after normalisation.
        The regularisation term added to the least‑squares objective is
            λ * Σ p_i log p_i
        where p_i are the normalised pheromone values.
        """
        X_arr = [np.asarray(v, dtype=float) for v in X]
        Phi = np.vstack([self._phi(x) for x in X_arr])          # (n, m)
        y_vec = np.asarray(y, dtype=float)

        # --- entropy term -------------------------------------------------
        p = pheromone_signals / (pheromone_signals.sum() + 1e-12)
        entropy = -np.sum(p * np.log(p + 1e-12))               # scalar
        # We embed the scalar entropy as a ridge‑like penalty on the diagonal.
        reg_matrix = lam * entropy * np.eye(Phi.shape[1])

        # Solve (ΦᵀΦ + λ·H) w = Φᵀ y
        A = Phi.T @ Phi + reg_matrix
        b = Phi.T @ y_vec
        self.weights = solve_linear(A, b)

# ----------------------------------------------------------------------
# 3. Hybrid tree cost that merges both parents
# ----------------------------------------------------------------------
def hybrid_edge_weight(node_a_feat: np.ndarray, node_b_feat: np.ndarray,
                       surrogate: RBFSurrogate,
                       pheromone_signal: np.ndarray,
                       time_index: int) -> float:
    """
    Compute the effective cost of an edge (a,b) at a given discrete time step.
    Base cost = surrogate prediction on the concatenated feature vector.
    Modulation = 1 / (1 + pheromone_intensity) where intensity is taken from the
    fractional decay signal at ``time_index``.
    """
    base_feat = np.concatenate([node_a_feat, node_b_feat])
    base = surrogate.predict(base_feat)

    # Pheromone intensity is a scalar; we use the same signal for all edges
    # (in a full implementation each edge would have its own signal).
    intensity = pheromone_signal[time_index] if time_index < len(pheromone_signal) else pheromone_signal[-1]
    mod = 1.0 / (1.0 + intensity)   # stronger pheromone → cheaper (or more attractive)
    return max(base * mod, 1e-8)    # avoid zero weight

def prim_mst(nodes: list[int],
            edge_weight_func) -> tuple[float, list[tuple[int, int]]]:
    """
    Simple Prim's algorithm.
    ``edge_weight_func(i, j)`` must return the weight for edge (i, j).
    Returns total weight and list of edges in the MST.
    """
    n = len(nodes)
    if n == 0:
        return 0.0, []
    in_tree = {nodes[0]}
    edge_list = []
    total = 0.0
    while len(in_tree) < n:
        min_w = math.inf
        min_edge = None
        for u in in_tree:
            for v in nodes:
                if v in in_tree:
                    continue
                w = edge_weight_func(u, v)
                if w < min_w:
                    min_w = w
                    min_edge = (u, v)
        if min_edge is None:  # disconnected graph
            break
        u, v = min_edge
        in_tree.add(v)
        edge_list.append(min_edge)
        total += min_w
    return total, edge_list

def compute_hybrid_tree_cost(node_features: dict[int, np.ndarray],
                             surrogate: RBFSurrogate,
                             pheromone_signal: np.ndarray,
                             time_index: int = 0) -> float:
    """
    Build a minimum‑cost spanning tree where edge weights are hybridised.
    """
    nodes = list(node_features.keys())

    def weight(i: int, j: int) -> float:
        return hybrid_edge_weight(node_features[i], node_features[j],
                                  surrogate, pheromone_signal, time_index)

    total, _ = prim_mst(nodes, weight)
    return total

# ----------------------------------------------------------------------
# 4. Demonstration functions
# ----------------------------------------------------------------------
def demo_pheromone_fractional() -> np.ndarray:
    """Generate a sample pheromone signal using fractional decay."""
    signal_value = 5.0          # arbitrary amplitude
    half_life = 2.0             # seconds
    alpha = 0.7                 # fractional order (0 < α < 1)
    return calculate_pheromone_signal(signal_value, half_life, alpha)

def demo_rbf_surrogate(node_feats: dict[int, np.ndarray],
                       pheromone: np.ndarray) -> RBFSurrogate:
    """
    Fit an RBF surrogate to synthetic edge‑cost data using entropy‑regularised fitting.
    """
    # Create synthetic training data: each edge gets a target cost = Euclidean distance.
    centers = []
    targets = []
    pher_signals = []

    ids = list(node_feats.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            feat = np.concatenate([node_feats[a], node_feats[b]])
            centers.append(feat)
            targets.append(euclidean(node_feats[a], node_feats[b]))
            # Use the same pheromone signal for all edges (simplification)
            pher_signals.append(pheromone[min(i, len(pheromone)-1)])

    surrogate = RBFSurrogate(centers, epsilon=1.0)
    surrogate.fit(centers, targets, np.array(pher_signals), lam=0.05)
    return surrogate

def demo_hybrid_tree() -> float:
    """
    End‑to‑end demo:
    1. Create random node features.
    2. Produce a fractional pheromone signal.
    3. Fit entropy‑regularised RBF surrogate.
    4. Compute hybrid MST cost at a chosen time index.
    """
    random.seed(42)
    np.random.seed(42)

    # 1. Node features (2‑D points)
    node_features = {i: np.random.rand(2) for i in range(5)}

    # 2. Fractional pheromone signal
    pheromone = demo_pheromone_fractional()

    # 3. Fit surrogate
    surrogate = demo_rbf_surrogate(node_features, pheromone)

    # 4. Compute cost at time index 50 (≈0.5 s)
    cost = compute_hybrid_tree_cost(node_features, surrogate, pheromone, time_index=50)
    return cost

# ----------------------------------------------------------------------
# 5. Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        cost = demo_hybrid_tree()
        print(f"Hybrid MST cost (demo): {cost:.6f}")
    except Exception as e:
        print("Demo failed:", e, file=sys.stderr)
        sys.exit(1)