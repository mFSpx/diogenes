# DARWIN HAMMER — match 4297, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m42_s2.py (gen4)
# parent_b: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s1.py (gen3)
# born: 2026-05-29T23:54:52Z

"""Hybrid Sheaf‑RBF Dedupe Module
---------------------------------
Parents:
* `hybrid_hybrid_hybrid_hybrid_m42_s2.py` – defines a cellular sheaf, restriction
  maps and a VRAM planning dataclass. Its core mathematics revolves around
  linear restriction maps on a directed graph and Bayesian‑style resource
  allocation.
* `hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s1.py` – provides perceptual
  hashing, radial‑basis‑function (RBF) surrogate modelling and a Gaussian‑kernel
  linear solver.

Mathematical Bridge
-------------------
Node sections of the sheaf are vectors in ℝⁿ.  We treat each section as a data
point and:
1. **Cluster** similar sections with a perceptual hash (`phash`).  The hash
   creates equivalence classes that act as RBF centres.
2. **Surrogate Modelling** – an RBF matrix built from Euclidean distances
   between node vectors and the hash‑derived centres is solved (via the
   `solve_linear` routine from the second parent) to obtain weights that
   predict a scalar attribute (e.g. estimated memory usage) for every node.
3. **VRAM Planning** – the predicted attribute becomes the likelihood in a
   simple Bayesian update that yields a posterior estimate of required VRAM.
   This posterior feeds the `VramSlotPlan` dataclass from the first parent.

The three functions below demonstrate this unified workflow."""


import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Sheaf and VRAM planning
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Sheaf:
    """
    Cellular sheaf on a directed graph.

    * Nodes carry a vector space of dimension given by ``node_dims``.
    * Each directed edge ``(u, v)`` carries a linear restriction map
      ``src_map : ℝ^{dim(u)} → ℝ^{dim(e)}`` and
      ``dst_map : ℝ^{dim(v)} → ℝ^{dim(e)}``.
    * A *section* assigns a vector to every node.
    """

    def __init__(self, node_dims: Dict[int, int], edges: List[Tuple[int, int]]):
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[int, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[int, int], src_map: np.ndarray, dst_map: np.ndarray) -> None:
        """Register the two restriction matrices for a directed edge."""
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float),
                                      np.asarray(dst_map, dtype=float))

    def set_section(self, node: int, vector: Sequence[float]) -> None:
        """Assign a vector (section) to a node."""
        if len(vector) != self.node_dims[node]:
            raise ValueError(f"vector length {len(vector)} does not match node dimension {self.node_dims[node]}")
        self._sections[node] = np.asarray(vector, dtype=float)

    def get_section(self, node: int) -> np.ndarray:
        return self._sections[node]

    def all_sections(self) -> Dict[int, np.ndarray]:
        return self._sections


# ----------------------------------------------------------------------
# Parent B – Perceptual hashing, RBF surrogate, linear solver
# ----------------------------------------------------------------------


Vector = Sequence[float]


def compute_phash(values: List[float]) -> int:
    """Perceptual hash: 1‑bit per entry up to 64 entries based on average."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Gauss‑Jordan elimination for a dense linear system."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]

        div = m[col][col]
        m[col] = [v / div for v in m[col]]

        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]

    return [row[-1] for row in m]


# ----------------------------------------------------------------------
# Hybrid Functions (the required three)
# ----------------------------------------------------------------------


def hash_sections(sheaf: Sheaf) -> Dict[int, int]:
    """
    Compute a perceptual hash for each node's section.
    Returns a mapping node → hash (int).
    """
    hashes = {}
    for node, vec in sheaf.all_sections().items():
        hashes[node] = compute_phash(vec.tolist())
    return hashes


def rbf_surrogate_predict(
    sheaf: Sheaf,
    target: Dict[int, float],
    epsilon: float = 1.0
) -> Dict[int, float]:
    """
    Build an RBF surrogate using hash‑derived centres and predict the scalar
    `target` for every node.

    Parameters
    ----------
    sheaf : Sheaf
        The sheaf whose node vectors are the data points.
    target : dict[node, float]
        Known scalar values for a (possibly subset of) nodes.
    epsilon : float
        Kernel width for the Gaussian RBF.

    Returns
    -------
    dict[node, float]
        Predicted scalar for each node in the sheaf.
    """
    # 1. Determine centres = unique hashes of nodes that have a target value.
    hashes = hash_sections(sheaf)
    centre_nodes = [n for n in target.keys()]
    centres = [sheaf.get_section(n) for n in centre_nodes]

    # 2. Build kernel matrix K_{ij} = φ(||x_i - c_j||)
    K = [[gaussian(euclidean(sheaf.get_section(i), c), epsilon) for c in centres]
         for i in sheaf.node_dims.keys()]

    # 3. Solve K * w = y  for weights w (y are target values ordered as centre_nodes)
    y = [target[n] for n in centre_nodes]
    w = solve_linear(K, y)

    # 4. Predict for all nodes using the same kernel expansion.
    predictions = {}
    for node in sheaf.node_dims.keys():
        k_vec = [gaussian(euclidean(sheaf.get_section(node), c), epsilon) for c in centres]
        pred = sum(w_i * k_i for w_i, k_i in zip(w, k_vec))
        predictions[node] = pred
    return predictions


def plan_vram_allocation(
    sheaf: Sheaf,
    predictions: Dict[int, float],
    prior_mean: float = 100.0,
    prior_std: float = 50.0
) -> List[VramSlotPlan]:
    """
    Perform a Bayesian‑style update where the RBF prediction acts as the
    likelihood (Gaussian with unit variance).  The posterior mean is taken as
    the estimated VRAM requirement for each node.

    Returns a list of `VramSlotPlan` objects, one per node.
    """
    plans = []
    for node, pred in predictions.items():
        # Likelihood: N(pred, 1)
        # Prior:   N(prior_mean, prior_std^2)
        # Posterior variance = 1 / (1 + 1/prior_std^2)
        post_var = 1.0 / (1.0 + 1.0 / (prior_std ** 2))
        post_std = math.sqrt(post_var)

        # Posterior mean (weighted average)
        post_mean = (pred / 1.0 + prior_mean / (prior_std ** 2)) * post_var

        est_mb = max(0, int(round(post_mean)))
        plan = VramSlotPlan(
            artifact_id=f"node_{node}",
            artifact_kind="sheaf_section",
            action="allocate",
            estimated_mb=est_mb,
            reason="RBF‑surrogate posterior estimate",
            detail={
                "prediction": pred,
                "posterior_mean": post_mean,
                "posterior_std": post_std,
                "hash": hash_sections(sheaf)[node],
            },
        )
        plans.append(plan)
    return plans


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Construct a tiny sheaf with 3 nodes.
    node_dims = {0: 4, 1: 4, 2: 4}
    edges = [(0, 1), (1, 2)]
    sheaf = Sheaf(node_dims, edges)

    # Random sections for each node.
    rng = np.random.default_rng(42)
    for n, dim in node_dims.items():
        vec = rng.random(dim).tolist()
        sheaf.set_section(n, vec)

    # Define a synthetic target: memory usage proportional to L2 norm.
    target = {0: euclidean(sheaf.get_section(0), [0, 0, 0, 0]),
              1: euclidean(sheaf.get_section(1), [0, 0, 0, 0])}  # node 2 will be predicted.

    # Hybrid pipeline
    hashes = hash_sections(sheaf)
    print("Perceptual hashes:", hashes)

    predictions = rbf_surrogate_predict(sheaf, target, epsilon=0.5)
    print("RBF predictions:", predictions)

    plans = plan_vram_allocation(sheaf, predictions)
    for p in plans:
        print(p.as_dict())