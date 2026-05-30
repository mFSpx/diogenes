# DARWIN HAMMER — match 3198, survivor 3
# gen: 7
# parent_a: minimum_cost_tree.py (gen0)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_hybrid_hybrid_m2726_s3.py (gen6)
# born: 2026-05-29T23:48:27Z

"""Hybrid algorithm merging Minimum‑Cost Tree scoring (Parent A) with NLMS‑based adaptive prediction
and model‑pool resource awareness (Parent B).

Mathematical bridge:
- Parent A yields a vector of root‑to‑node distances `d_i` and a scalar material cost `M`.
- Parent B adapts a weight vector `w` to predict a scalar from a feature vector via the
  Normalised Least‑Mean‑Squares (NLMS) rule: `y = w·x`.
- The hybrid treats the distance vector `d` as the NLMS feature vector `x`.  The NLMS
  prediction `y = w·d` estimates a “latent cost” that is blended with the explicit
  tree cost `M + λ·Σd_i`.  The learning‑rate of NLMS is modulated by the doomsday rule,
  providing a time‑varying factor that couples the calendar logic of Parent B.
- Additionally, a Bayesian Information Criterion (BIC) term penalises model complexity,
  using the material cost as a surrogate log‑likelihood.

The resulting cost function is:
    C_hybrid = M + λ·Σd_i + BIC(−M, n_params, n_samples) + α·(w·d)

where `α` derives from the doomsday‑adjusted learning rate.
"""

import math
import random
import sys
from pathlib import Path
from datetime import date
import numpy as np

# ----- Types -------------------------------------------------
Point = tuple[float, float]
Edge = tuple[str, str]
NodeId = str
ModelEdge = tuple[NodeId, NodeId, int]

# ----- Parent A core -------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
    path_weight: float = 0.2,
) -> tuple[float, dict[str, float]]:
    """Return total material cost and a dict of root‑to‑node distances."""
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])

    dist: dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)

    return material, dist

# ----- Parent B core -------------------------------------------------
def doomsday_rule(year: int, month: int, day: int) -> int:
    """Map calendar day to a cyclic weight in [0,6]."""
    return (date(year, month, day).weekday() + 1) % 7

def bayesian_information_criterion(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """Standard BIC."""
    if n_samples <= 0:
        return float('inf')
    return -2.0 * log_likelihood + n_params * math.log(n_samples)

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """NLMS weight update."""
    error = target - nlms_predict(weights, x)
    norm = float(x @ x) + eps
    weights = weights + (mu * error / norm) * x
    return weights, error

class ModelTier:
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # Evict the oldest entry (dict preserves insertion order)
            evicted = next(iter(self.loaded))
            self.loaded.pop(evicted)
        self.load(model)

# ----- Hybrid functions ----------------------------------------------

def compute_hybrid_features(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
    path_weight: float = 0.2,
) -> tuple[float, np.ndarray]:
    """
    Run Parent A's tree_cost, then return:
        material_cost  – sum of edge lengths
        distance_vec   – ordered numpy vector of distances from root (excluding root)
    """
    material, dist = tree_cost(nodes, edges, root, path_weight)
    # Exclude the root (distance 0) for NLMS features
    distance_vec = np.array([d for nid, d in sorted(dist.items()) if nid != root], dtype=float)
    return material, distance_vec

def hybrid_predict(
    model_pool: ModelPool,
    weights: np.ndarray,
    distances: np.ndarray,
) -> float:
    """
    Blend NLMS prediction with a resource‑aware scaling factor.
    The scaling factor α is derived from the doomsday rule (Parent B) and the
    current RAM utilisation (Parent B).  This creates a true mathematical
    coupling between the two parent systems.
    """
    today = date.today()
    doomsday_factor = 1.0 + doomsday_rule(today.year, today.month, today.day) / 7.0
    ram_usage = model_pool._used() or 1  # avoid division by zero
    alpha = 0.5 * doomsday_factor * (ram_usage / model_pool.ram_ceiling_mb)
    nlms_out = nlms_predict(weights, distances)
    return alpha * nlms_out

def hybrid_tree_cost_update(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
    model_pool: ModelPool,
    weights: np.ndarray,
    path_weight: float = 0.2,
    mu: float = 0.5,
) -> tuple[float, np.ndarray]:
    """
    Compute the hybrid cost:
        C = M + λ·Σd_i + BIC(−M, n_params, n_samples) + hybrid_predict(...)
    Then perform an NLMS update where the target is the negative material cost
    (so the algorithm learns to offset material expense with the weight vector).

    Returns the scalar hybrid cost and the updated weight vector.
    """
    # ----- Parent A contribution -----
    material, distances = compute_hybrid_features(nodes, edges, root, path_weight)
    path_sum = float(distances.sum())
    base_cost = material + path_weight * path_sum

    # ----- Parent B statistical penalty -----
    n_params = weights.size
    n_samples = len(distances) if len(distances) > 0 else 1
    # Use -material as a surrogate log‑likelihood (larger material ⇒ worse likelihood)
    bic = bayesian_information_criterion(log_likelihood=-material, n_params=n_params, n_samples=n_samples)

    # ----- Hybrid predictive term -----
    pred_term = hybrid_predict(model_pool, weights, distances)

    total_cost = base_cost + bic + pred_term

    # ----- NLMS weight adaptation -----
    # Target chosen as the negative material cost to push weights towards compensating expense.
    target = -material
    updated_weights, _ = nlms_update(weights, distances, target, mu=mu)

    return total_cost, updated_weights

# ----- Example usage / smoke test ------------------------------------
if __name__ == "__main__":
    # Simple tree with 4 nodes
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D")]
    root = "A"

    # Initialise model pool with a tiny model
    pool = ModelPool(ram_ceiling_mb=1024)
    pool.load(ModelTier(name="tiny", ram_mb=200, tier="T1"))

    # Initialise NLMS weights (one per non‑root node)
    initial_weights = np.random.randn(len(nodes) - 1)

    cost, new_weights = hybrid_tree_cost_update(
        nodes=nodes,
        edges=edges,
        root=root,
        model_pool=pool,
        weights=initial_weights,
        path_weight=0.2,
        mu=0.5,
    )

    print(f"Hybrid cost: {cost:.4f}")
    print(f"Updated weights: {new_weights}")

    # Ensure the code runs without raising exceptions
    sys.exit(0)