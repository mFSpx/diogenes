# DARWIN HAMMER — match 1835, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_ternary_route_hybrid_ternary_route_m98_s1.py (gen3)
# parent_b: hybrid_krampus_brainmap_bandit_router_m129_s1.py (gen1)
# born: 2026-05-29T23:39:12Z

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Basic I/O helpers
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNTIME_DIR = ROOT / "04_RUNTIME" / "fairyfuse"
DEFAULT_HEARTBEAT = RUNTIME_DIR / "ternary_router_heartbeat.jsonl"


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def emit_json(obj: Any) -> None:
    """Print a JSON object with deterministic key order."""
    print(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


# ----------------------------------------------------------------------
# Geometry utilities
# ----------------------------------------------------------------------
def euclidean_length(a: Point, b: Point) -> float:
    """Straight‑line distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_length_matrix(
    nodes: Dict[str, Point], edges: List[Edge]
) -> Tuple[np.ndarray, Dict[str, int], List[Edge]]:
    """
    Build a symmetric matrix L where L[i, j] = Euclidean length if (i, j) is an edge,
    otherwise 0. Returns the matrix, a name‑to‑index map, and the canonical edge list.
    """
    idx_map = {name: i for i, name in enumerate(sorted(nodes))}
    n = len(nodes)
    L = np.zeros((n, n), dtype=float)

    for a, b in edges:
        i, j = idx_map[a], idx_map[b]
        length = euclidean_length(nodes[a], nodes[b])
        L[i, j] = L[j, i] = length

    # keep edges in the same order as input for later reference
    return L, idx_map, edges


# ----------------------------------------------------------------------
# Contextual LinUCB router
# ----------------------------------------------------------------------
class HybridLinUCBRouter:
    """
    A deeper fusion of the ternary‑route geometry and the bandit router.

    * The *context* for a node is its row in the Euclidean length matrix
      (distances to every other node).  This captures the full geometric
      neighbourhood of the node.
    * For each possible *action* (i.e. destination node) we maintain a
      LinUCB estimator (A, b) that learns a linear model
      `reward ≈ θᵀ·context`.
    * Action selection uses the classic LinUCB upper‑confidence bound:
          p = θ̂ᵀ·x + α·√(xᵀ·A⁻¹·x)
      where α controls exploration.
    """

    def __init__(
        self,
        nodes: Dict[str, Point],
        edges: List[Edge],
        alpha: float = 1.0,
        regularization: float = 1.0,
    ):
        """
        Parameters
        ----------
        nodes
            Mapping from node identifier to its 2‑D coordinates.
        edges
            List of undirected edges (used only to build the length matrix).
        alpha
            Exploration coefficient for the UCB term.
        regularization
            Initial diagonal value for each A matrix (acts as ridge regularisation).
        """
        self.length_matrix, self.idx_map, self.edge_list = build_length_matrix(
            nodes, edges
        )
        self.node_names = sorted(nodes)  # deterministic ordering
        self.num_nodes = len(self.node_names)
        self.context_dim = self.num_nodes  # each row of L is a d‑dim vector
        self.alpha = alpha

        # Initialise LinUCB parameters per action (destination node)
        # A_j = λ·I, b_j = 0
        self.A: Dict[int, np.ndarray] = {
            j: regularization * np.eye(self.context_dim) for j in range(self.num_nodes)
        }
        self.b: Dict[int, np.ndarray] = {
            j: np.zeros((self.context_dim, 1)) for j in range(self.num_nodes)
        }

        # Cache of A⁻¹ for speed; updated lazily after each observation
        self.A_inv: Dict[int, np.ndarray] = {
            j: np.linalg.inv(self.A[j]) for j in range(self.num_nodes)
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _name_to_index(self, name: str) -> int:
        """Convert a node name to its integer index; raises KeyError if unknown."""
        return self.idx_map[name]

    def _context_vector(self, node_idx: int) -> np.ndarray:
        """
        Return the context column for a given node as a column vector (d×1).
        The context is the row of the length matrix; we reshape to column for
        linear‑algebra convenience.
        """
        return self.length_matrix[node_idx].reshape((-1, 1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select_action(self, context_name: str) -> str:
        """
        Choose the destination node with the highest LinUCB score for the given
        context node.

        Returns
        -------
        str
            The name of the selected action node.
        """
        i = self._name_to_index(context_name)
        x = self._context_vector(i)  # d×1

        best_score = -np.inf
        best_action = None

        for j in range(self.num_nodes):
            A_inv_j = self.A_inv[j]
            theta_j = A_inv_j @ self.b[j]  # d×1
            p = float(theta_j.T @ x + self.alpha * np.sqrt(x.T @ A_inv_j @ x))
            if p > best_score:
                best_score = p
                best_action = j

        return self.node_names[best_action]  # type: ignore[return-value]

    def update_policy(self, context_name: str, action_name: str, reward: float) -> None:
        """
        Perform a LinUCB update for the (context, action) pair with the observed reward.

        Parameters
        ----------
        context_name
            Source node identifier.
        action_name
            Destination node identifier.
        reward
            Observed scalar reward (higher is better).
        """
        i = self._name_to_index(context_name)
        j = self._name_to_index(action_name)
        x = self._context_vector(i)  # d×1

        # A_j ← A_j + x·xᵀ
        self.A[j] += x @ x.T
        # b_j ← b_j + reward·x
        self.b[j] += reward * x

        # Update cached inverse via Sherman‑Morrison formula for efficiency
        A_inv = self.A_inv[j]
        numerator = A_inv @ x @ x.T @ A_inv
        denominator = 1.0 + float(x.T @ A_inv @ x)
        self.A_inv[j] = A_inv - numerator / denominator

    def get_expected_reward(self, context_name: str, action_name: str) -> float:
        """
        Return the current linear estimate θ̂ᵀ·x for the given pair.
        This is the exploitation‑only component of the LinUCB score.
        """
        i = self._name_to_index(context_name)
        j = self._name_to_index(action_name)
        x = self._context_vector(i)
        theta = self.A_inv[j] @ self.b[j]
        return float(theta.T @ x)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def dump_state(self) -> Dict[str, Any]:
        """Return a JSON‑serialisable snapshot of the internal state."""
        return {
            "alpha": self.alpha,
            "node_names": self.node_names,
            "A_matrices": {
                self.node_names[j]: self.A[j].tolist() for j in range(self.num_nodes)
            },
            "b_vectors": {
                self.node_names[j]: self.b[j].flatten().tolist()
                for j in range(self.num_nodes)
            },
        }


def demonstrate_hybrid_operation():
    """
    Small sanity‑check demo mirroring the original example but now using
    a true contextual bandit.  The reward signal is fabricated for illustration.
    """
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}
    edges = [("A", "B"), ("A", "C"), ("B", "C")]

    router = HybridLinUCBRouter(nodes, edges, alpha=0.5)

    # Simulated interactions
    router.update_policy("A", "B", reward=1.0)   # good outcome from A→B
    router.update_policy("A", "C", reward=0.2)   # weaker outcome from A→C
    router.update_policy("B", "C", reward=1.5)   # strong outcome from B→C

    # Choose an action for context "A"
    chosen = router.select_action("A")
    print(f"Selected action for context 'A': {chosen}")

    # Expected (linear) reward estimate for the chosen pair
    exp_reward = router.get_expected_reward("A", chosen)
    print(f"Estimated reward (exploitation only) for A→{chosen}: {exp_reward:.4f}")

    # Dump internal state for debugging / inspection
    emit_json(router.dump_state())


if __name__ == "__main__":
    demonstrate_hybrid_operation()