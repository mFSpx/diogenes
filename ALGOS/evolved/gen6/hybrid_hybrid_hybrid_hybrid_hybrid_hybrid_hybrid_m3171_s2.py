# DARWIN HAMMER — match 3171, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_hybrid_m310_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2701_s0.py (gen5)
# born: 2026-05-29T23:48:26Z

import math
import random
import sys
import numpy as np

DIM = 10000
_POSITIVE_WEIGHTS = np.array(
    [1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64
)
_NEGATIVE_WEIGHTS = np.array(
    [0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64
)

class TreeNode:
    __slots__ = ("name", "children", "pheromone", "reward")

    def __init__(self, name: str, reward: float = 0.0):
        self.name = name
        self.children: list[TreeNode] = []
        self.pheromone: float = 0.1
        self.reward: float = reward

    def add_child(self, child: "TreeNode") -> None:
        self.children.append(child)

class Multivector:
    def __init__(self, components: dict[tuple[int, ...], float] | None = None):
        self.components: dict[tuple[int, ...], float] = {}
        if components:
            for blade, coeff in components.items():
                if coeff != 0.0:
                    self.components[tuple(blade)] = float(coeff)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = Multivector(self.components)
        for blade, coeff in other.components.items():
            result.components[blade] = result.components.get(blade, 0.0) + coeff
            if abs(result.components[blade]) < 1e-12:
                del result.components[blade]
        return result

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coeff for blade, coeff in self.components.items() if len(blade) == k}
        )

    def scalar_part(self) -> float:
        return self.components.get((), 0.0)

    @staticmethod
    def from_hypervector(hv: np.ndarray, grade: int = 1) -> "Multivector":
        result = Multivector()
        for i in range(len(hv)):
            if hv[i] != 0:
                result.components[(i,) * grade] = hv[i]
        return result

def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

def compute_regex_features(text: str) -> dict[str, int]:
    evidence_re = text.count("evidence") + text.count("proof")
    planning_re = text.count("plan") + text.count("strategy")
    delay_re = text.count("delay") + text.count("wait")
    return {
        "evidence": evidence_re,
        "planning": planning_re,
        "delay": delay_re,
        "support": 0,
        "boundary": 0,
        "outcome": 0,
        "impulsive": 0,
        "scarcity": 0,
        "risk": 0,
    }

def node_to_hypervector(node: TreeNode, context_text: str) -> np.ndarray:
    raw_counts = compute_regex_features(context_text)
    feature_vector = np.array(
        [raw_counts[key] for key in _FEATURE_ORDER], dtype=np.int64
    )
    tau = node.pheromone
    bias = sigmoid(tau)
    w = _POSITIVE_WEIGHTS * bias - _NEGATIVE_WEIGHTS * (1.0 - bias)
    dot = int(np.dot(w, feature_vector))
    v = np.sign(dot) if dot != 0 else 0
    return np.full(DIM, v, dtype=np.int8)

def update_pheromone(node: TreeNode, reward: float, decay: float = 0.9) -> None:
    old = node.pheromone
    node.pheromone = decay * old + (1.0 - decay) * reward
    node.pheromone = max(0.01, min(node.pheromone, 10.0))

def aggregate_multivector(nodes: list[TreeNode], context_text: str) -> Multivector:
    result = Multivector()
    for node in nodes:
        hv = node_to_hypervector(node, context_text)
        mv = Multivector.from_hypervector(hv)
        result = result + mv
    return result