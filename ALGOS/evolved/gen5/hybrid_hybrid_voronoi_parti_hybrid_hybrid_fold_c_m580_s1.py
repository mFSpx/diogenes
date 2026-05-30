# DARWIN HAMMER — match 580, survivor 1
# gen: 5
# parent_a: hybrid_voronoi_partition_hybrid_hybrid_hybrid_m325_s4.py (gen4)
# parent_b: hybrid_hybrid_fold_change_d_hybrid_hybrid_hard_t_m88_s0.py (gen4)
# born: 2026-05-29T23:29:49Z

import numpy as np
import math
from pathlib import Path
from collections import defaultdict

class Point(tuple):
    pass

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

class Sheaf:
    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
        self._restrictions = {}
        self._sections = {}

    def set_restriction(
        self,
        edge: tuple,
        src_map: np.ndarray,
        dst_map: np.ndarray,
    ) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError
        self._restrictions[edge] = (src_map, dst_map)

class BanditPolicy:
    def __init__(self):
        self._policy: dict[str, list[float]] = {}

    def reset(self) -> None:
        """Clear the internal bandit policy."""
        self._policy.clear()

    def _reward(self, action: str) -> float:
        """Average reward observed for *action*."""
        total, n = self._policy.get(action, [0.0, 0.0])
        return total / n if n else 0.0

    def _count(self, action: str) -> float:
        """Number of times *action* has been selected."""
        return self._policy.get(action, [0.0, 0.0])[1]

    def update(self, updates: list[tuple[str, float]]) -> None:
        """Incorporate a batch of (action, reward) observations."""
        for action, reward in updates:
            if action not in self._policy:
                self._policy[action] = [0.0, 0.0]
            self._policy[action][0] += reward
            self._policy[action][1] += 1

    def select_action(self, actions: list, bias: float = 0.0) -> str:
        """Select an action based on the bandit policy and bias."""
        best_action = max(actions, key=lambda action: self._reward(action) + bias)
        return best_action

def hybrid_state_vector(point: Point, seeds: list) -> np.ndarray:
    """Create a hybrid state vector by concatenating the point coordinates 
    with the index of the nearest Voronoi seed."""
    nearest_seed = nearest(point, seeds)
    return np.array([point[0], point[1], nearest_seed])

def assign_region(point: Point, seeds: list) -> int:
    """Assign a region to a point based on the nearest Voronoi seed."""
    return nearest(point, seeds)

def hybrid_select_action(point: Point, seeds: list, actions: list, policy: BanditPolicy) -> str:
    """Select an action based on the hybrid state vector and the bandit policy."""
    state_vector = hybrid_state_vector(point, seeds)
    region = assign_region(point, seeds)
    # Use the region as a bias term to influence the action selection
    bias = region / len(seeds)
    # Select the action with the highest average reward plus bias
    return policy.select_action(actions, bias)

if __name__ == "__main__":
    seeds = [Point(0, 0), Point(1, 1), Point(2, 2)]
    point = Point(0.5, 0.5)
    actions = ["action1", "action2", "action3"]
    policy = BanditPolicy()
    policy.reset()
    policy.update([("action1", 1.0), ("action2", 0.5), ("action3", 0.0)])
    print(hybrid_state_vector(point, seeds))
    print(assign_region(point, seeds))
    print(hybrid_select_action(point, seeds, actions, policy))