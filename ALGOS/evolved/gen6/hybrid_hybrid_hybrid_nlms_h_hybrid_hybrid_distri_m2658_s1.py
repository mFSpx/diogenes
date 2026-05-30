# DARWIN HAMMER — match 2658, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_nlms_hybrid_h_hybrid_hard_truth_ma_m1061_s0.py (gen5)
# parent_b: hybrid_hybrid_distributed_l_hybrid_physarum_netw_m110_s4.py (gen4)
# born: 2026-05-29T23:43:26Z

import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel (unused in the final version, kept for backward compatibility)."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 1‑D arrays."""
    if a.shape != b.shape:
        raise ValueError("vectors must have same dimension")
    return np.linalg.norm(a - b)


def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash based on the median of the vector."""
    if not values:
        return 0
    median = np.median(values)
    bits = 0
    for v in values[:64]:  # limit to 64 bits for a fixed‑size hash
        bits = (bits << 1) | int(v >= median)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


def similarity_matrix(features: Dict[int, List[float]]) -> Tuple[np.ndarray, List[int]]:
    """
    Build a similarity matrix using Hamming distance of perceptual hashes.
    Returns the matrix and the ordered list of node identifiers.
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    hashes = [compute_phash(features[n]) for n in nodes]

    for i in range(n):
        for j in range(i, n):
            d = hamming_distance(hashes[i], hashes[j])
            sim = 1.0 - d / 64.0
            S[i, j] = sim
            S[j, i] = sim
    return S, nodes


def rbf_kernel(S: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Compute an RBF kernel from a similarity matrix.
    The kernel is K_ij = exp(-gamma * (1 - S_ij)^2).
    """
    return np.exp(-gamma * (1.0 - S) ** 2)


@dataclass
class PhysarumNetwork:
    """
    Conductance matrix G (symmetric, non‑negative) and node pressures p.
    """
    conductance: np.ndarray
    pressures: np.ndarray


def solve_pressures(G: np.ndarray, source: int, sink: int, flow: float = 1.0) -> np.ndarray:
    """
    Solve the linear system L p = b where L = D - G (graph Laplacian),
    b is a source‑sink injection vector ( +flow at source, -flow at sink ),
    and one node (sink) is fixed to zero to make the system full rank.
    """
    n = G.shape[0]
    D = np.diag(G.sum(axis=1))
    L = D - G

    b = np.zeros(n)
    b[source] = flow
    b[sink] = -flow

    # Fix sink pressure to zero by removing its row/column
    mask = np.arange(n) != sink
    L_reduced = L[np.ix_(mask, mask)]
    b_reduced = b[mask]

    p_reduced = np.linalg.solve(L_reduced, b_reduced)

    p = np.zeros(n)
    p[mask] = p_reduced
    p[sink] = 0.0
    return p


def nlms_update(weights: np.ndarray, input_vec: np.ndarray, target: float, mu: float) -> np.ndarray:
    """
    Normalized Least Mean Squares update.
    `weights` and `input_vec` must have the same shape (1‑D).
    """
    if weights.shape != input_vec.shape:
        raise ValueError("weights and input vector must share the same shape")
    prediction = np.dot(input_vec, weights)
    error = target - prediction
    norm_factor = np.dot(input_vec, input_vec) + 1e-12
    return weights + (mu / norm_factor) * error * input_vec


def physarum_update_conductance(
    network: PhysarumNetwork,
    K: np.ndarray,
    source: int,
    sink: int,
    learning_rate: float = 0.01,
) -> PhysarumNetwork:
    """
    Update conductances using a Physarum‑inspired rule:
        G_ij ← G_ij + η * (|Δp_ij| - G_ij) * K_ij
    where Δp_ij is the pressure difference across edge (i, j) and K is the RBF kernel.
    """
    p = solve_pressures(network.conductance, source, sink)
    delta_p = np.abs(p[:, None] - p[None, :])  # |p_i - p_j|
    update = learning_rate * (delta_p - network.conductance) * K
    G_new = np.maximum(network.conductance + update, 0.0)  # keep conductances non‑negative
    # Enforce symmetry
    G_new = (G_new + G_new.T) / 2.0
    return PhysarumNetwork(conductance=G_new, pressures=p)


def hybrid_temperature(
    phase: int,
    total_phases: int,
    G: np.ndarray,
    p: np.ndarray,
    t0: float = 1.0,
    alpha: float = 0.95,
    eps: float = 1e-12,
) -> float:
    """
    Temperature schedule that couples the current conductance magnitude
    with the pressure gradient magnitude.
    """
    grad_norm = np.linalg.norm(G * np.abs(p[:, None] - p[None, :]))
    decay = alpha ** phase
    return t0 * decay * grad_norm / (grad_norm + eps)


def hybrid_operation(
    features: Dict[int, List[float]],
    total_phases: int,
    source: int,
    sink: int,
    gamma: float = 1.0,
    nlms_mu: float = 0.1,
    physarum_lr: float = 0.01,
    t0: float = 1.0,
    alpha: float = 0.95,
) -> Tuple[np.ndarray, PhysarumNetwork, List[float]]:
    """
    Core hybrid routine.

    1. Build similarity matrix → RBF kernel K.
    2. Initialise a symmetric conductance matrix G (identity) and random NLMS weights.
    3. For each phase:
        a. Update NLMS weights using the flattened kernel as input.
        b. Update conductances via Physarum dynamics.
        c. Compute temperature (returned for diagnostics).

    Returns the final NLMS weights, the final Physarum network, and a list of temperatures per phase.
    """
    S, nodes = similarity_matrix(features)
    K = rbf_kernel(S, gamma=gamma)

    n = len(nodes)
    # Initialise NLMS weights as a vector matching the flattened kernel
    w = np.random.rand(n * n)

    # Initialise conductance matrix as identity (ensures connectivity)
    G0 = np.eye(n)
    network = PhysarumNetwork(conductance=G0, pressures=np.zeros(n))

    temperatures: List[float] = []

    for phase in range(total_phases):
        # ---- NLMS step -------------------------------------------------
        input_vec = K.flatten()
        # Target is chosen as the average similarity; any scalar works for demonstration
        target = np.mean(K)
        w = nlms_update(w, input_vec, target, nlms_mu)

        # ---- Physarum step ---------------------------------------------
        network = physarum_update_conductance(
            network, K, source=source, sink=sink, learning_rate=physarum_lr
        )

        # ---- Temperature schedule ---------------------------------------
        T = hybrid_temperature(
            phase=phase,
            total_phases=total_phases,
            G=network.conductance,
            p=network.pressures,
            t0=t0,
            alpha=alpha,
        )
        temperatures.append(T)

    return w, network, temperatures


if __name__ == "__main__":
    # Example usage with a tiny graph of 4 nodes.
    example_features = {
        0: [1.0, 2.0, 3.0],
        1: [4.0, 5.0, 6.0],
        2: [7.0, 8.0, 9.0],
        3: [2.0, 1.0, 0.0],
    }
    total_phases = 15
    source_node = 0
    sink_node = 3

    final_weights, final_network, temps = hybrid_operation(
        features=example_features,
        total_phases=total_phases,
        source=source_node,
        sink=sink_node,
        gamma=0.5,
        nlms_mu=0.05,
        physarum_lr=0.02,
        t0=2.0,
        alpha=0.9,
    )

    print("Final NLMS weights (first 10 entries):", final_weights[:10])
    print("Final conductance matrix:\n", final_network.conductance)
    print("Final pressures:", final_network.pressures)
    print("Temperature schedule:", temps)