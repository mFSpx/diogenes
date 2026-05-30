# DARWIN HAMMER — match 2461, survivor 4
# gen: 5
# parent_a: hybrid_voronoi_partition_hybrid_hybrid_hybrid_m325_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m765_s0.py (gen4)
# born: 2026-05-29T23:42:27Z

"""
Hybrid Voronoi‑Associative‑Memory (HVAM) system.

Parents:
- hybrid_voronoi_partition_hybrid_hybrid_hybrid_m325_s2.py
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m765_s0.py

Mathematical bridge:
Both parents manipulate high‑dimensional vectors with linear maps.
The Voronoi partition groups feature vectors into cells defined by
centroids; the Dense Associative Memory (DAM) defines an energy
E(ξ)=−log∑_i exp(β·M_i·ξ)+½‖ξ‖² for a memory matrix M_i.
We fuse them by assigning a distinct memory matrix M_c to each Voronoi
cell c.  Retrieval first selects the nearest centroid (Voronoi step) and
then performs DAM energy minimisation inside that cell.  The Bayesian
update of model‑selection probabilities (parent B) is used to adapt the
choice of a ModelTier based on RAM constraints, while a discrete
Ollivier‑Ricci curvature estimate on the Voronoi adjacency graph
provides a regularisation term that can be added to the DAM energy.
"""

import math
import random
import sys
import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core structures from Parent A (Sheaf + DAM utilities)
# ----------------------------------------------------------------------
class Sheaf:
    """A collection of nodes (Voronoi cells) with linear restriction maps."""
    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]]):
        self.node_dims: Dict[Any, int] = dict(node_dims)
        self.edges: List[Tuple[Any, Any]] = list(edges)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}

    def set_restriction(
        self,
        edge: Tuple[Any, Any],
        src_map: np.ndarray,
        dst_map: np.ndarray,
    ) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float),
                                      np.asarray(dst_map, dtype=float))

    def set_section(self, node: Any, value: np.ndarray) -> None:
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node: Any) -> np.ndarray:
        return self._sections.get(node, np.zeros(self.node_dims[node]))

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def _lse(z: np.ndarray) -> float:
    m = z.max()
    return m + np.log(np.exp(z - m).sum())

def dam_energy(xi: np.ndarray, M: np.ndarray, beta: float = 1.0) -> float:
    """Dense Associative Memory energy for a single memory matrix M."""
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    scores = beta * (M @ xi)
    return -_lse(scores) / beta + 0.5 * np.sum(xi ** 2)

def dam_retrieve(M: np.ndarray, query: np.ndarray, beta: float = 1.0,
                 lr: float = 0.1, steps: int = 50) -> np.ndarray:
    """
    Gradient descent on the DAM energy to retrieve a pattern from memory M.
    Returns the converged state vector.
    """
    x = np.asarray(query, dtype=float).copy()
    for _ in range(steps):
        grad = beta * (M.T @ _softmax(beta * (M @ x))) - x
        x += lr * grad
    return x

# ----------------------------------------------------------------------
# Stylometry & Model selection utilities from Parent B
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

def stylometry_vector(text: str) -> np.ndarray:
    """
    Simple bag‑of‑category vector: for each FUNCTION_CATS key count the
    occurrence of words belonging to that category.
    Returns a normalized (L2) vector.
    """
    words = [w.strip(".,!?:;\"'").lower() for w in text.split()]
    counts = []
    for cat in FUNCTION_CATS:
        cat_set = FUNCTION_CATS[cat]
        cnt = sum(1 for w in words if w in cat_set)
        counts.append(cnt)
    vec = np.array(counts, dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

class ModelTier:
    """Represents a model with a RAM footprint and a qualitative tier."""
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

class ModelPool:
    """Container for ModelTier objects with Bayesian selection."""
    def __init__(self, models: List[ModelTier]):
        self.models = list(models)
        # Prior probabilities (uniform)
        self.prior = np.full(len(self.models), 1.0 / len(self.models))

    def likelihood(self, required_ram: int) -> np.ndarray:
        """
        Likelihood of each model given a RAM requirement.
        Simple step function: 1 if model.ram_mb >= required_ram else 0.
        """
        return np.array([1.0 if m.ram_mb >= required_ram else 0.0 for m in self.models])

    def posterior(self, required_ram: int) -> np.ndarray:
        """Bayesian update of model probabilities given a RAM constraint."""
        L = self.likelihood(required_ram)
        unnorm = self.prior * L
        if unnorm.sum() == 0:
            # No model satisfies the constraint; keep prior unchanged.
            return self.prior
        return unnorm / unnorm.sum()

    def best_model(self, required_ram: int) -> ModelTier:
        """Select the model with highest posterior probability."""
        post = self.posterior(required_ram)
        idx = int(np.argmax(post))
        return self.models[idx]

# ----------------------------------------------------------------------
# Fusion utilities
# ----------------------------------------------------------------------
def build_voronoi_sheaf(features: np.ndarray,
                       n_cells: int,
                       random_state: int | None = None) -> Tuple[Sheaf, np.ndarray]:
    """
    Partition `features` (N×D) into `n_cells` Voronoi regions using k‑means‑like
    random centroids. Returns a Sheaf where each node corresponds to a cell
    and carries a random memory matrix M_c (D×D). Also returns the centroids.
    """
    rng = np.random.default_rng(random_state)
    N, D = features.shape
    # Initialise centroids by sampling random points from the dataset
    centroids = features[rng.choice(N, size=n_cells, replace=False)]

    # Simple Lloyd iteration (few steps)
    for _ in range(5):
        distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
        assignments = distances.argmin(axis=1)
        for i in range(n_cells):
            pts = features[assignments == i]
            if len(pts) > 0:
                centroids[i] = pts.mean(axis=0)

    # Build node dimensions (all equal to D) and edges (connect neighboring centroids)
    node_dims = {i: D for i in range(n_cells)}
    # Edge creation: connect each centroid to its k nearest neighbours (k=2)
    k = 2
    edge_list: List[Tuple[int, int]] = []
    for i in range(n_cells):
        dists = np.linalg.norm(centroids - centroids[i], axis=1)
        nearest = np.argsort(dists)[1:k+1]  # skip self
        for j in nearest:
            if (i, j) not in edge_list and (j, i) not in edge_list:
                edge_list.append((i, j))

    sheaf = Sheaf(node_dims, edge_list)

    # Assign a random dense memory matrix to each node and store as a section
    for i in range(n_cells):
        M_i = rng.normal(scale=0.1, size=(D, D))
        sheaf.set_section(i, M_i)

    # Store restriction maps as identity (no transformation across edges)
    for (u, v) in edge_list:
        I = np.eye(D)
        sheaf.set_restriction((u, v), I, I)

    return sheaf, centroids

def curvature_regularisation(sheaf: Sheaf,
                             centroids: np.ndarray,
                             alpha: float = 0.05) -> Dict[Tuple[int, int], float]:
    """
    Approximate discrete Ollivier‑Ricci curvature on the Voronoi graph.
    For each edge (u,v) we compute
        κ_uv = 1 - (d_uv / avg_neighbor_distance)
    and return a dict of curvature values scaled by `alpha`.
    """
    curvatures: Dict[Tuple[int, int], float] = {}
    # Pre‑compute average neighbor distance for each node
    avg_dist: Dict[int, float] = {}
    for u in sheaf.node_dims:
        neigh = [v for (a, v) in sheaf.edges if a == u] + [a for (a, b) in sheaf.edges if b == u]
        if not neigh:
            avg_dist[u] = 0.0
            continue
        dists = [np.linalg.norm(centroids[u] - centroids[v]) for v in neigh]
        avg_dist[u] = sum(dists) / len(dists)

    for (u, v) in sheaf.edges:
        d_uv = np.linalg.norm(centroids[u] - centroids[v])
        avg = (avg_dist[u] + avg_dist[v]) / 2.0 if (avg_dist[u] + avg_dist[v]) > 0 else 1.0
        kappa = 1.0 - (d_uv / avg)
        curvatures[(u, v)] = alpha * kappa
    return curvatures

def hybrid_query(sheaf: Sheaf,
                 centroids: np.ndarray,
                 query_vec: np.ndarray,
                 beta: float = 1.0,
                 lr: float = 0.1,
                 steps: int = 60) -> Tuple[int, np.ndarray]:
    """
    Perform a hybrid retrieval:
    1. Locate the Voronoi cell whose centroid is nearest to `query_vec`.
    2. Retrieve a pattern from the cell's memory matrix using DAM gradient descent.
    3. Add curvature regularisation to the final state (optional).
    Returns the cell index and the retrieved vector.
    """
    # 1. Voronoi selection
    dists = np.linalg.norm(centroids - query_vec, axis=1)
    cell = int(dists.argmin())

    # 2. DAM retrieval inside the selected cell
    M = sheaf.get_section(cell)          # shape (D, D)
    retrieved = dam_retrieve(M, query_vec, beta=beta, lr=lr, steps=steps)

    # 3. Curvature regularisation (push towards neighbours)
    curv = curvature_regularisation(sheaf, centroids)
    # Compute a simple Laplacian‑like correction
    correction = np.zeros_like(retrieved)
    for (u, v), kappa in curv.items():
        if u == cell:
            correction += kappa * (sheaf.get_section(v).mean(axis=1) - retrieved)
        elif v == cell:
            correction += kappa * (sheaf.get_section(u).mean(axis=1) - retrieved)
    retrieved += correction

    return cell, retrieved

def select_model_for_query(pool: ModelPool,
                           query_vec: np.ndarray,
                           base_ram: int = 200) -> ModelTier:
    """
    Estimate required RAM from the norm of the query vector and
    select the most probable model using the Bayesian posterior.
    """
    # Simple heuristic: larger norm → larger RAM need
    required = base_ram + int(100 * np.linalg.norm(query_vec))
    return pool.best_model(required)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy texts
    texts = [
        "I think therefore I am.",
        "The quick brown fox jumps over the lazy dog.",
        "She does not like the rain but loves the sunshine.",
        "We will meet at the conference in June.",
        "No one can deny the importance of data."
    ]

    # 1. Convert texts to stylometry vectors
    vectors = np.stack([stylometry_vector(t) for t in texts])
    if vectors.shape[1] == 0:
        print("No stylometry features extracted; exiting.")
        sys.exit(0)

    # 2. Build Voronoi Sheaf
    sheaf, centroids = build_voronoi_sheaf(vectors, n_cells=3, random_state=42)

    # 3. Create a ModelPool
    models = [
        ModelTier("tiny", ram_mb=150, tier="low"),
        ModelTier("small", ram_mb=300, tier="mid"),
        ModelTier("medium", ram_mb=600, tier="high"),
        ModelTier("large", ram_mb=1200, tier="ultra")
    ]
    pool = ModelPool(models)

    # 4. Perform hybrid queries on each text
    for i, txt in enumerate(texts):
        qvec = stylometry_vector(txt)
        cell, retrieved = hybrid_query(sheaf, centroids, qvec,
                                       beta=1.2, lr=0.05, steps=80)
        model = select_model_for_query(pool, qvec, base_ram=200)
        print(f"Text {i+1}:")
        print(f"  Voronoi cell   -> {cell}")
        print(f"  Retrieved norm -> {np.linalg.norm(retrieved):.4f}")
        print(f"  Chosen model   -> {model.name} ({model.tier}, {model.ram_mb} MB)")
        print("-" * 40)