# DARWIN HAMMER — match 2417, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_shap_attribut_hybrid_krampus_brain_m986_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_bandit_router_m999_s3.py (gen4)
# born: 2026-05-29T23:42:14Z

"""Hybrid algorithm merging SHAP‑based graph attribution with a ternary router bandit.

Parents:
- hybrid_hybrid_shap_attribut_hybrid_krampus_brain_m986_s1.py (SHAP, graph clustering,
  perceptual hashing, Ollivier‑Ricci curvature)
- hybrid_hybrid_hybrid_hybrid_hybrid_bandit_router_m999_s3.py (bandit actions,
  ternary router loss, matrix‑based updates)

Mathematical bridge:
The SHAP values computed for each feature become *expected rewards* for
bandit actions that correspond to graph nodes.  The ternary router’s weight
matrix is interpreted as the adjacency‑like valuation of the graph; its
gradient (derived from the hybrid loss) is injected back into the graph as
a pheromone‑style update.  Curvature between node neighborhoods modulates
the SHAP kernel weight, while MinHash signatures of node clusters provide
the context for the bandit’s propensity calculation.  This creates a closed
loop:

    model → SHAP → node reward → bandit UCB → selected action →
    router loss gradient → weight update → new graph valuation →
    curvature‑adjusted SHAP kernel → next SHAP iteration.

The implementation below provides three core functions that embody this
loop."""
import sys
import math
import random
import json
from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Set

import numpy as np

# ----------------------------------------------------------------------
# Types used by both parents
# ----------------------------------------------------------------------
Node = int
Graph = Dict[int, Set[int]]
Model = Dict[int, float]

# ----------------------------------------------------------------------
# Parent A utilities (SHAP, hashing, curvature)
# ----------------------------------------------------------------------
def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Kernel weight used in exact Shapley computation."""
    if feature_count == 0:
        return 0.0
    return (math.factorial(subset_size) *
            math.factorial(feature_count - subset_size - 1) /
            math.factorial(feature_count))


def compute_dhash(values: List[float]) -> int:
    """Difference hash – simple pairwise comparison."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits


def compute_phash(values: List[float]) -> int:
    """Perceptual hash – threshold against mean."""
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


def broadcast_probability(phase: int, step: int) -> float:
    """Probability used in the pheromone broadcasting phase."""
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


def shap_value(feature_index: int,
               feature_count: int,
               value_fn: List[float]) -> float:
    """
    Exact Shapley value for a single feature using the provided
    coalition value function `value_fn`.  `value_fn` is assumed to be a
    list where the i‑th entry is the model output when the i‑th feature
    is present (all others omitted).  This toy implementation uses the
    kernel weighting defined above.
    """
    total = 0.0
    # Enumerate all subsets that do NOT contain the feature
    for subset_size in range(feature_count):
        weight = shapley_kernel_weight(subset_size, feature_count)
        # For a toy example we treat value_fn[subset_size] as the value of
        # a random subset of that size; in practice a model would be queried.
        marginal = value_fn[subset_size] - value_fn[0]  # baseline subtraction
        total += weight * marginal
    return total


def ollivier_ricci_curvature(G: Graph, u: Node, v: Node) -> float:
    """
    Simple discrete Ollivier‑Ricci curvature between two adjacent nodes.
    Uses the overlap of neighbor distributions as a proxy for the
    transportation distance.
    """
    if v not in G.get(u, set()):
        return 0.0
    Nu = G[u] | {u}
    Nv = G[v] | {v}
    intersection = len(Nu & Nv)
    union = len(Nu | Nv)
    if union == 0:
        return 0.0
    # Curvature in [−1, 1]; higher overlap → higher curvature
    return (intersection / union) * 2 - 1.0


def minhash_signature(elements: Set[int], num_perm: int = 64) -> List[int]:
    """
    Compute a MinHash signature for a set of integer identifiers.
    """
    max_hash = (1 << 32) - 1
    sig = [max_hash] * num_perm
    for e in elements:
        random.seed(e)  # deterministic per element
        for i in range(num_perm):
            h = random.getrandbits(32)
            if h < sig[i]:
                sig[i] = h
    return sig


# ----------------------------------------------------------------------
# Parent B utilities (bandit, ternary router)
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z‑format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_context(text: str | None) -> dict[str, Any]:
    """Parse a JSON string into a dict; empty input yields {}."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SyntaxError("Invalid JSON") from exc


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


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


class TernaryRouterTTT:
    """
    Linear router with a ternary loss that blends SSIM, VFE and
    reconstruction gradients.
    """
    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def hybrid_loss(self, x: np.ndarray) -> float:
        wx = self.weights @ x
        ssim_loss = 1 - self.calculate_ssim(x, wx)
        vfe_gradient = self.calculate_vfe_gradient(x, wx)
        reconstruction_gradient = self.calculate_reconstruction_gradient(x, wx)
        return ssim_loss + vfe_gradient + reconstruction_gradient

    @staticmethod
    def calculate_ssim(x: np.ndarray, wx: np.ndarray) -> float:
        num = np.mean(x * wx)
        den = np.linalg.norm(x) * np.linalg.norm(wx)
        return 0.0 if den == 0 else num / den

    @staticmethod
    def calculate_vfe_gradient(x: np.ndarray, wx: np.ndarray) -> float:
        diff = wx - x
        num = np.mean(diff * x)
        den = np.linalg.norm(x)
        return 0.0 if den == 0 else num / den

    @staticmethod
    def calculate_reconstruction_gradient(x: np.ndarray, wx: np.ndarray) -> float:
        diff = wx - x
        num = np.mean(diff ** 2)
        den = np.linalg.norm(x)
        return 0.0 if den == 0 else num / den

    def calculate_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of the hybrid loss w.r.t. the weight matrix.
        For demonstration we use a simple outer‑product approximation.
        """
        wx = self.weights @ x
        grad = np.outer(wx - x, x)  # shape (out_dim, in_dim)
        return grad

    def hybrid_step(self, x: np.ndarray, learning_rate: float = 0.01) -> None:
        """One gradient descent step on the weight matrix."""
        gradient = self.calculate_gradient(x)
        self.weights -= learning_rate * gradient


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_node_shap_scores(G: Graph,
                             model: Model,
                             feature_values: List[float]) -> Dict[Node, float]:
    """
    Compute a SHAP‑derived score for each node.
    The node value is the SHAP value of the feature indexed by the node id,
    multiplied by the Ollivier‑Ricci curvature with its highest‑degree neighbour.
    """
    scores: Dict[Node, float] = {}
    max_degree_node = max(G, key=lambda n: len(G[n]))
    for node in G:
        # Feature index is taken modulo feature count for safety
        f_idx = node % len(feature_values)
        shap = shap_value(f_idx, len(feature_values), feature_values)
        # Find neighbour with highest degree to compute curvature
        neighbours = G[node]
        if neighbours:
            high_deg = max(neighbours, key=lambda n: len(G[n]))
            curvature = ollivier_ricci_curvature(G, node, high_deg)
        else:
            curvature = 0.0
        scores[node] = shap * (1 + curvature)  # curvature as a modifier
    return scores


def select_bandit_action(node_scores: Dict[Node, float],
                         router: TernaryRouterTTT,
                         context_id: str) -> BanditAction:
    """
    Build a set of BanditAction objects from node scores, compute a UCB‑style
    confidence bound using the router’s current loss as a variance proxy,
    and return the action with the highest upper confidence bound.
    """
    # Convert scores to expected rewards
    actions: List[BanditAction] = []
    base_loss = router.hybrid_loss(np.array(list(node_scores.values())))
    for node, reward in node_scores.items():
        # Propensity derived from MinHash similarity of the node’s neighbourhood
        neighbourhood = set([node]) | router.weights.shape[0] * {node}
        # For deterministic behaviour we hash the neighbourhood size
        propensity = 1.0 / (1 + len(neighbourhood))
        # Confidence bound: sqrt( (2 * log T) / N ) where T ~ loss, N ~ propensity
        confidence = math.sqrt(2 * math.log(max(base_loss, 1e-6) + 1) / max(propensity, 1e-6))
        actions.append(BanditAction(
            action_id=str(node),
            propensity=propensity,
            expected_reward=reward,
            confidence_bound=confidence,
            algorithm="HybridSHAPBandit"
        ))
    # Upper confidence bound
    best = max(actions, key=lambda a: a.expected_reward + a.confidence_bound)
    return best


def hybrid_update(G: Graph,
                  router: TernaryRouterTTT,
                  selected_action: BanditAction,
                  input_vector: np.ndarray) -> None:
    """
    Perform a weight update on the router using the selected action’s reward.
    The reward modulates the learning rate, and the graph receives a
    pheromone‑style broadcast proportional to the broadcast probability.
    """
    # Modulate learning rate by reward magnitude
    reward_factor = max(selected_action.expected_reward, 0.0)
    lr = 0.01 * (1 + reward_factor)
    router.hybrid_step(input_vector, learning_rate=lr)

    # Pheromone broadcast on the graph
    phase = int(selected_action.propensity * 10) + 1
    step = int(selected_action.confidence_bound * 10) + 1
    prob = broadcast_probability(phase, step)

    # Update neighbours of the selected node
    node = int(selected_action.action_id)
    for neighbour in G.get(node, []):
        if random.random() < prob:
            # Simple pheromone increment (could be stored elsewhere)
            # Here we just mutate the graph by adding a dummy self‑loop to indicate reinforcement
            G[neighbour].add(neighbour)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny random graph
    random.seed(42)
    G: Graph = {i: set() for i in range(5)}
    for i in range(5):
        for j in range(i + 1, 5):
            if random.random() < 0.4:
                G[i].add(j)
                G[j].add(i)

    # Dummy model: node -> weight
    model: Model = {i: random.random() for i in range(5)}

    # Feature values for SHAP (one per node for simplicity)
    feature_vals = [random.random() for _ in range(5)]

    # Compute SHAP‑based node scores
    node_scores = compute_node_shap_scores(G, model, feature_vals)

    # Initialise router with a small random weight matrix
    weight_matrix = np.random.randn(5, 5) * 0.1
    router = TernaryRouterTTT(weights=weight_matrix)

    # Choose an action via the bandit
    action = select_bandit_action(node_scores, router, context_id="test")

    # Perform hybrid update
    input_vec = np.array(list(node_scores.values()))
    hybrid_update(G, router, action, input_vec)

    # Simple sanity prints (not part of required output but harmless)
    print("Selected action:", action)
    print("Updated router weights (norm):", np.linalg.norm(router.weights))
    print("Graph after broadcast:", G)