# DARWIN HAMMER — match 4029, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_ssim_m1265_s3.py (gen6)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_hybrid_krampu_m196_s2.py (gen4)
# born: 2026-05-29T23:53:19Z

import numpy as np
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Sequence
from collections import defaultdict


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """A simple reservoir model whose last delta is stored for later use."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0  # internal, not part of the public state

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Update the reservoir level and remember the last delta."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """A derived quantity that mixes the last delta with static parameters."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))


# ----------------------------------------------------------------------
# Core mathematics
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Return a dimensionless sphericity index (0‑1)."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def calculate_health_score(morphology: Morphology) -> float:
    """Health score is defined as the sphericity index of the object."""
    return spheric_index(morphology.length, morphology.width, morphology.height)


def ssim(x: Sequence[float], y: Sequence[float],
         dynamic_range: float = 1.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """
    Compute a 1‑D analogue of the Structural Similarity Index Measure.
    The implementation follows the standard SSIM formula but works on
    plain sequences (e.g. feature vectors) rather than images.
    """
    if len(x) != len(y):
        raise ValueError("Sequences must have equal length")
    if not x:
        raise ValueError("Sequences must not be empty")

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx ** 2 + my ** 2 + c1) * (vx + vy + c2)

    return float(numerator / denominator)


def extract_full_features(text: str) -> Dict[str, float]:
    """
    Stub feature extractor – in a real system this would parse *text*.
    The returned dictionary is deliberately ordered to guarantee a stable
    vector representation across calls.
    """
    return {
        "visceral_ratio": 0.5,
        "tech_ratio": 0.3,
        "legal_osint_ratio": 0.2,
        "ledger_density": 0.1,
        "recursion_score": 0.4,
        "directive_ratio": 0.6,
        "target_density": 0.7,
        "forensic_shield_ratio": 0.8,
        "poetic_entropy": 0.9,
        "dissociative_index": 0.1,
        "wrath_velocity": 0.2,
        "bureaucratic_weaponization_index": 0.3,
        "resource_exhaustion_metric": 0.4,
        "swarm_orchestration_density": 0.5,
        "logic_crucifixion_index": 0.6,
        "conspiracy_grounding_ratio": 0.7,
        "chaotic_good_tax": 0.8,
        "corporate_grit_tension": 0.9,
        "countdown_density": 0.1,
        "asset_structuring_weight": 0.2,
        "pitch_formatting_ratio": 0.3,
        "agent_symmetry_ratio": 0.4,
        "protocol_discipline": 0.5,
        "manic_velocity": 0.6,
    }


def lazy_rw_distribution(adj: Dict[int, List[int]],
                          node: int,
                          alpha: float = 0.5) -> Dict[int, float]:
    """
    Return a one‑step lazy random‑walk distribution from *node*.
    With probability *alpha* we stay, otherwise we move uniformly to a neighbour.
    """
    neighbours = adj.get(node, [])
    deg = len(neighbours)
    dist = {node: alpha}
    if deg:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            dist[nb] = dist.get(nb, 0.0) + spread
    return dist


# ----------------------------------------------------------------------
# Graph construction – the weak point of the original code was the
# reliance on a raw dot‑product threshold before computing SSIM.
# The improved pipeline computes SSIM for *all* pairs, then thresholds
# the resulting similarity matrix, guaranteeing that the edge weights
# are meaningful and comparable.
# ----------------------------------------------------------------------
def _pairwise_ssim_matrix(vectors: List[Sequence[float]],
                          dynamic_range: float = 1.0) -> np.ndarray:
    """
    Compute a symmetric matrix M where M[i, j] = ssim(v_i, v_j).
    The diagonal is 1.0 by definition.
    """
    n = len(vectors)
    M = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            sim = ssim(vectors[i], vectors[j], dynamic_range=dynamic_range)
            M[i, j] = M[j, i] = sim
    return M


def build_similarity_graph(vectors: List[Sequence[float]],
                           similarity_threshold: float = 0.5,
                           dynamic_range: float = 1.0) -> Dict[int, Dict[int, float]]:
    """
    Build a weighted adjacency dictionary where an edge exists iff the
    SSIM similarity exceeds *similarity_threshold*.
    Edge weights are the raw SSIM values.
    """
    if not vectors:
        raise ValueError("Feature vector list must not be empty")

    # Normalise each vector to unit L2 norm – this makes SSIM less sensitive
    # to absolute scale while preserving relative patterns.
    normed = [np.asarray(v, dtype=float) / (np.linalg.norm(v) + 1e-12) for v in vectors]

    sim_matrix = _pairwise_ssim_matrix(normed, dynamic_range=dynamic_range)

    graph: Dict[int, Dict[int, float]] = defaultdict(dict)
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= similarity_threshold:
                weight = float(sim_matrix[i, j])
                graph[i][j] = weight
                graph[j][i] = weight
    return dict(graph)


def weight_edges_with_store_state(graph: Dict[int, Dict[int, float]],
                                 store_state: StoreState) -> Dict[int, Dict[int, float]]:
    """
    Modulate each edge weight by a factor derived from the last delta of
    *store_state*.  The factor is ``1 + tanh(delta)`` which stays in (0, 2)
    and preserves the ordering of deltas while keeping weights bounded.
    """
    factor = 1.0 + math.tanh(store_state._last_delta)
    weighted_graph: Dict[int, Dict[int, float]] = defaultdict(dict)
    for src, neighbours in graph.items():
        for dst, w in neighbours.items():
            weighted_graph[src][dst] = w * factor
    return dict(weighted_graph)


# ----------------------------------------------------------------------
# Curvature – a more principled definition based on the weighted
# adjacency matrix.  The curvature of a node is the harmonic mean of
# the incident edge weights, which penalises nodes with many weak
# connections and rewards tightly‑coupled neighbourhoods.
# ----------------------------------------------------------------------
def node_curvature(weighted_adj: Dict[int, Dict[int, float]],
                   node: int) -> float:
    """
    Return the harmonic‑mean curvature of *node*.
    If the node has no incident edges, curvature is defined as 0.0.
    """
    incident = weighted_adj.get(node, {})
    if not incident:
        return 0.0
    inv_sum = sum(1.0 / (w + 1e-12) for w in incident.values())
    return len(incident) / inv_sum


# ----------------------------------------------------------------------
# High‑level API – combines all pieces into a single, easy‑to‑use call.
# ----------------------------------------------------------------------
def run_hybrid_pipeline(texts: List[str],
                        similarity_threshold: float = 0.5,
                        store_state: StoreState | None = None,
                        alpha: float = 0.5) -> Tuple[
                            Dict[int, Dict[int, float]],
                            StoreState,
                            Dict[int, float]
                        ]:
    """
    1. Extract feature vectors from *texts*.
    2. Build a similarity graph using SSIM.
    3. Update *store_state* with a synthetic inflow/outflow (demo purpose).
    4. Re‑weight the graph edges with the store‑state information.
    5. Compute node curvatures.

    Returns:
        weighted_graph – adjacency dict with final edge weights,
        store_state    – the (possibly updated) StoreState instance,
        curvatures     – mapping node → curvature.
    """
    # ------------------------------------------------------------------
    # Step 1 – feature extraction
    # ------------------------------------------------------------------
    feature_vectors = [list(extract_full_features(t).values()) for t in texts]

    # ------------------------------------------------------------------
    # Step 2 – similarity graph
    # ------------------------------------------------------------------
    base_graph = build_similarity_graph(feature_vectors,
                                        similarity_threshold=similarity_threshold)

    # ------------------------------------------------------------------
    # Step 3 – optional StoreState update (demo values)
    # ------------------------------------------------------------------
    if store_state is None:
        store_state = StoreState()
    # In a real system inflow/outflow would be derived from the domain.
    dummy_inflow = [1.0 for _ in range(3)]
    dummy_outflow = [0.5 for _ in range(3)]
    store_state.update(dummy_inflow, dummy_outflow)

    # ------------------------------------------------------------------
    # Step 4 – weight edges with the StoreState delta
    # ------------------------------------------------------------------
    weighted_graph = weight_edges_with_store_state(base_graph, store_state)

    # ------------------------------------------------------------------
    # Step 5 – curvature per node
    # ------------------------------------------------------------------
    curvatures = {node: node_curvature(weighted_graph, node) for node in weighted_graph}

    return weighted_graph, store_state, curvatures


# ----------------------------------------------------------------------
# Simple demonstration when the module is executed directly.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_texts = ["sample text"] * 8
    graph, state, curv = run_hybrid_pipeline(demo_texts, similarity_threshold=0.4)

    print("Weighted similarity graph (sparse representation):")
    for src, nbrs in graph.items():
        print(f"  {src} -> {nbrs}")

    print("\nStoreState after dummy update:")
    print(asdict(state))

    print("\nNode curvatures:")
    for node, val in curv.items():
        print(f"  node {node}: {val:.4f}")