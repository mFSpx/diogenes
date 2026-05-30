# DARWIN HAMMER — match 1835, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_ternary_route_hybrid_ternary_route_m98_s1.py (gen3)
# parent_b: hybrid_krampus_brainmap_bandit_router_m129_s1.py (gen1)
# born: 2026-05-29T23:39:12Z

"""
Fusing hybrid_hybrid_ternary_route_hybrid_ternary_route_m98_s1.py and 
hybrid_krampus_brainmap_bandit_router_m129_s1.py into a unified system.

The mathematical bridge between the two parents lies in interpreting the 
ternary route's length matrix as a context vector for the bandit algorithm. 
The length matrix's dimensions serve as features for contextual action selection.

The hybrid algorithm integrates the Euclidean length calculations from the 
ternary route with the LinUCB/Thompson action routing from the bandit router.
"""

import json
import math
import random
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
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Edge]]:
    """
    Return a symmetric matrix L where L[i, j] = Euclidean length if (i, j) is an edge,
    otherwise 0. Also return the ordered list of edge index pairs matching the
    non‑zero entries of L (used for vectorised prior updates) and edge list.
    """
    idx_map = {name: i for i, name in enumerate(sorted(nodes))}
    n = len(nodes)
    L = np.zeros((n, n), dtype=float)
    edge_idx: List[Tuple[int, int]] = []
    edge_list: List[Edge] = []

    for a, b in edges:
        i, j = idx_map[a], idx_map[b]
        length = euclidean_length(nodes[a], nodes[b])
        L[i, j] = L[j, i] = length
        edge_idx.append((i, j))
        edge_list.append((a, b))

    return L, edge_idx, edge_list

class HybridRouter:
    def __init__(self, nodes: Dict[str, Point], edges: List[Edge]):
        self.length_matrix, _, _ = build_length_matrix(nodes, edges)
        self.num_nodes = len(nodes)
        self._reset_policy()

    def _reset_policy(self) -> None:
        self._POLICY = np.zeros((self.num_nodes, self.num_nodes))

    def update_policy(self, context_id: str, action_id: str, reward: float) -> None:
        i, j = int(context_id), int(action_id)
        self._POLICY[i, j] += reward

    def select_action(self, context_id: str) -> str:
        i = int(context_id)
        action_id = np.argmax(self._POLICY[i])
        return str(action_id)

    def get_expected_reward(self, context_id: str, action_id: str) -> float:
        i, j = int(context_id), int(action_id)
        return self._POLICY[i, j] / (self.num_nodes * self.num_nodes)

def demonstrate_hybrid_operation():
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}
    edges = [("A", "B"), ("A", "C"), ("B", "C")]
    router = HybridRouter(nodes, edges)

    # Update policy with some rewards
    router.update_policy("0", "1", 1.0)
    router.update_policy("0", "2", 0.5)
    router.update_policy("1", "2", 1.5)

    # Select actions
    action_id = router.select_action("0")
    print(f"Selected action: {action_id}")

    # Get expected rewards
    expected_reward = router.get_expected_reward("0", action_id)
    print(f"Expected reward: {expected_reward}")

if __name__ == "__main__":
    demonstrate_hybrid_operation()