# DARWIN HAMMER — match 4465, survivor 3
# gen: 6
# parent_a: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_xgboos_m642_s0.py (gen5)
# born: 2026-05-29T23:55:55Z

"""
Hybrid Perceptual‑RBF + Regret‑Weighted Tropical XGBoost

Parents:
- hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s3.py (perceptual hashing + radial‑basis surrogate)
- hybrid_hybrid_hybrid_distri_hybrid_hybrid_xgboos_m642_s0.py (tropical max‑plus broadcast + regret‑weighted XGBoost)

Mathematical bridge:
The tropical broadcast vector `b` (max‑plus propagation over a graph) is interpreted as a
*margin* term for the regret‑weighted split test.  We use `b` to weight the radial‑basis
function (RBF) interpolation built from perceptual‑hash clusters.  In this way the
similarity‑based surrogate model (RBF) is driven by the global broadcast strengths
computed in the tropical field, while the perceptual hashes provide a principled
selection of RBF centres.  The resulting hybrid predictor evaluates

    ŷ(x) = Σ_i w_i · φ(‖x−c_i‖)  +  λ·b_i

where `φ` is the Gaussian RBF, `c_i` are cluster representatives, `w_i` are solved
weights, and `b_i` are the tropical broadcast strengths associated with each centre.
The regret‑weighted acceptance probability from the XGBoost side is reused for a
simulated‑annealing style update of the centre set.
"""

import math
import random
import sys
import pathlib
from typing import Callable, Iterable, Sequence, Mapping, Hashable, List, Tuple, Dict

import numpy as np

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Core primitives from Parent A
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Simple Gauss‑Jordan elimination for Ax = b."""
    n = len(b)
    # augment matrix
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        # pivot
        pivot_row = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot_row][col]) < 1e-12:
            raise ValueError("Singular matrix")
        m[col], m[pivot_row] = m[pivot_row], m[col]
        # normalize
        piv = m[col][col]
        m[col] = [v / piv for v in m[col]]
        # eliminate
        for r in range(n):
            if r != col:
                factor = m[r][col]
                m[r] = [vr - factor * vc for vr, vc in zip(m[r], m[col])]
    return [row[-1] for row in m]

def compute_phash(values: List[float]) -> int:
    """Average‑based perceptual hash (max 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def cluster_by_phash(hashes: Dict[str, int], max_distance: int = 4) -> List[List[str]]:
    """Single‑pass clustering based on Hamming distance of hashes."""
    clusters: List[List[str]] = []
    for k, h in hashes.items():
        for c in clusters:
            if hamming_distance(h, hashes[c[0]]) <= max_distance:
                c.append(k)
                break
        else:
            clusters.append([k])
    return clusters

# ----------------------------------------------------------------------
# Tropical (max‑plus) primitives from Parent B
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]

def max_plus_matvec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Max‑plus matrix‑vector product: y_i = max_j (A_ij + x_j)."""
    return np.max(A + x, axis=1)

def tropical_broadcast(adj: np.ndarray, steps: int = 10) -> np.ndarray:
    """
    Repeatedly apply max‑plus multiplication to propagate a broadcast strength.
    Starts from a zero vector (log‑space) and returns the steady‑state vector.
    """
    n = adj.shape[0]
    b = np.zeros(n)  # log‑space strengths (zero == 0 in max‑plus)
    for _ in range(steps):
        b = max_plus_matvec(adj, b)
    return b

def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance used in simulated annealing."""
    if delta_e < 0:
        return 1.0
    return math.exp(-delta_e / temperature)

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def build_rbf_model(
    data: Dict[str, List[float]],
    epsilon: float = 1.0,
) -> Tuple[List[Vector], List[float], List[float]]:
    """
    Construct an RBF surrogate:
    1. Compute perceptual hashes for each datum.
    2. Cluster by hash distance.
    3. Choose a representative centre (first key of each cluster).
    4. Solve for weights that reproduce the original scalar values (assumed to be the
       last element of each vector).
    Returns (centres, weights, target_values).
    """
    # 1. hashes
    hashes = {k: compute_phash(v) for k, v in data.items()}
    # 2. clusters
    clusters = cluster_by_phash(hashes)
    # 3. centres (pick first element of each cluster)
    centres: List[Vector] = []
    targets: List[float] = []
    for cluster in clusters:
        key = cluster[0]
        vec = data[key]
        centres.append(vec[:-1])          # all but last entry are features
        targets.append(vec[-1])           # last entry is the scalar target
    # 4. build kernel matrix
    n = len(centres)
    K = [[gaussian(euclidean(centres[i], centres[j]), epsilon) for j in range(n)]
         for i in range(n)]
    # Solve K * w = targets
    weights = solve_linear(K, targets)
    return centres, weights, targets

def hybrid_predict(
    x: Vector,
    centres: List[Vector],
    weights: List[float],
    broadcast: np.ndarray,
    centre_to_node: Dict[int, Node],
    lambda_regret: float = 0.5,
) -> float:
    """
    Predict a scalar for input vector `x` by mixing:
    - RBF surrogate evaluated on the perceptual‑hash centres.
    - Tropical broadcast strength associated with the nearest centre (regret term).
    """
    # RBF component
    rbf_val = sum(
        w * gaussian(euclidean(x, c))
        for c, w in zip(centres, weights)
    )
    # Find nearest centre index (Euclidean)
    dists = [euclidean(x, c) for c in centres]
    nearest_idx = int(np.argmin(dists))
    # Regret term: broadcast strength of the node linked to this centre
    node = centre_to_node[nearest_idx]
    # Assuming node indices correspond to broadcast vector order
    node_idx = int(node)  # type: ignore
    regret_val = broadcast[node_idx] if 0 <= node_idx < len(broadcast) else 0.0
    return rbf_val + lambda_regret * regret_val

def simulated_annealing_update(
    centres: List[Vector],
    weights: List[float],
    broadcast: np.ndarray,
    centre_to_node: Dict[int, Node],
    temperature: float = 1.0,
) -> Tuple[List[Vector], List[float]]:
    """
    Propose a random centre replacement, evaluate the change in total loss
    (MSE on the original targets approximated by the current model),
    and accept according to Metropolis probability.
    """
    # Randomly pick a centre to replace
    idx = random.randrange(len(centres))
    old_c = centres[idx]
    # Perturb centre slightly
    new_c = [v + random.uniform(-0.1, 0.1) for v in old_c]
    # Build a temporary model with the new centre
    temp_centres = centres.copy()
    temp_centres[idx] = new_c
    # Re‑solve weights for the temporary set
    n = len(temp_centres)
    K = [[gaussian(euclidean(temp_centres[i], temp_centres[j])) for j in range(n)]
         for i in range(n)]
    # For the test we reuse the original targets (placeholder)
    # In a full implementation we would keep the original target vector.
    # Here we simply keep the old weights to compute a surrogate loss.
    try:
        temp_weights = solve_linear(K, weights)  # rough proxy
    except ValueError:
        # Singular matrix – reject move
        return centres, weights
    # Compute simple proxy loss: sum of squared weight differences
    delta_e = sum((tw - w) ** 2 for tw, w in zip(temp_weights, weights))
    if random.random() < acceptance_probability(delta_e, temperature):
        return temp_centres, temp_weights
    else:
        return centres, weights

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Synthetic dataset: 8 nodes, each datum = 3 features + target
    synthetic_data: Dict[str, List[float]] = {
        f"{i}": [random.random() for _ in range(3)] + [random.random() * 10]
        for i in range(8)
    }

    # 2. Build RBF surrogate from perceptual hashes
    centres, weights, _ = build_rbf_model(synthetic_data, epsilon=1.5)

    # 3. Create a simple undirected graph adjacency (max‑plus weights = 1 for edges)
    n_nodes = 8
    adj = np.full((n_nodes, n_nodes), -np.inf)  # max‑plus uses -inf as zero element
    np.fill_diagonal(adj, 0.0)  # self‑loop weight 0
    # connect each node to the next (ring)
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        adj[i, j] = 0.0  # weight 0 (log‑space)
        adj[j, i] = 0.0

    # 4. Tropical broadcast strengths
    broadcast_vec = tropical_broadcast(adj, steps=15)

    # 5. Map centre index to node id (here we simply use same ordering)
    centre_to_node = {i: str(i) for i in range(len(centres))}

    # 6. Predict for a random query point
    query = [random.random() for _ in range(3)]
    pred = hybrid_predict(
        query,
        centres,
        weights,
        broadcast_vec,
        centre_to_node,
        lambda_regret=0.3,
    )
    print(f"Hybrid prediction for query {query[:3]}: {pred:.4f}")

    # 7. Perform a single simulated‑annealing update (demo)
    new_centres, new_weights = simulated_annealing_update(
        centres, weights, broadcast_vec, centre_to_node, temperature=0.5
    )
    # Show that code runs without exception
    print("Simulated annealing update completed.")