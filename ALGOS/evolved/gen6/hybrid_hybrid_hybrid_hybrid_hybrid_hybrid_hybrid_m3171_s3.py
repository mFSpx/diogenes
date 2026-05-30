# DARWIN HAMMER — match 3171, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_hybrid_m310_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2701_s0.py (gen5)
# born: 2026-05-29T23:48:26Z

import math
import random
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import numpy as np

# ----------------------------------------------------------------------
# Global configuration (Parent B)
# ----------------------------------------------------------------------
DIM = 10000                     # Dimensionality of hypervectors
SEED = 42                       # Global seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------------
# Feature hypervectors (static, random bipolar)
# ----------------------------------------------------------------------
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

_FEATURE_HV = {
    name: np.where(np.random.rand(DIM) < 0.5, 1, -1).astype(np.int8)
    for name in _FEATURE_ORDER
}

# ----------------------------------------------------------------------
# Regex feature extraction (Parent A stub)
# ----------------------------------------------------------------------
_EVIDENCE_RE = re.compile(r"\b(evidence|proof)\b", re.IGNORECASE)
_PLANNING_RE = re.compile(r"\b(plan|strategy)\b", re.IGNORECASE)
_DELAY_RE = re.compile(r"\b(delay|wait)\b", re.IGNORECASE)

def compute_regex_features(text: str) -> dict[str, int]:
    return {
        "evidence": len(_EVIDENCE_RE.findall(text)),
        "planning": len(_PLANNING_RE.findall(text)),
        "delay": len(_DELAY_RE.findall(text)),
        "support": 0,
        "boundary": 0,
        "outcome": 0,
        "impulsive": 0,
        "scarcity": 0,
        "risk": 0,
    }

# ----------------------------------------------------------------------
# Utility functions (Parent A)
# ----------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

# ----------------------------------------------------------------------
# Tree node (Parent A)
# ----------------------------------------------------------------------
class TreeNode:
    __slots__ = ("name", "children", "pheromone", "reward")
    def __init__(self, name: str, reward: float = 0.0):
        self.name = name
        self.children: list[TreeNode] = []
        self.pheromone: float = 0.1
        self.reward: float = reward
    def add_child(self, child: "TreeNode") -> None:
        self.children.append(child)

# ----------------------------------------------------------------------
# Pheromone dynamics (enhanced)
# ----------------------------------------------------------------------
def update_pheromone(node: TreeNode, reward: float, decay: float = 0.85, min_val: float = 0.01, max_val: float = 20.0) -> None:
    """
    Reinforcement learning style update with adaptive decay.
    """
    old = node.pheromone
    node.pheromone = decay * old + (1.0 - decay) * reward
    node.pheromone = max(min_val, min(node.pheromone, max_val))

# ----------------------------------------------------------------------
# Pheromone‑derived hypervector (binding seed)
# ----------------------------------------------------------------------
def pheromone_hypervector(node: TreeNode) -> np.ndarray:
    """
    Produce a deterministic bipolar hypervector from node name and pheromone.
    The pheromone magnitude modulates the random seed via sigmoid.
    """
    base = f"{node.name}:{node.pheromone:.6f}"
    h = hashlib.sha256(base.encode()).digest()
    # Expand hash to required length
    repeats = (DIM * 8 + len(h) - 1) // len(h)
    bits = np.unpackbits(np.frombuffer((h * repeats)[: (DIM // 8)], dtype=np.uint8))
    hv = np.where(bits[:DIM] == 0, -1, 1).astype(np.int8)
    # Modulate polarity by sigmoid to avoid saturation
    scale = sigmoid(node.pheromone) * 2 - 1  # map to (-1,1)
    return hv if scale >= 0 else -hv

# ----------------------------------------------------------------------
# Node → hypervector (deepened fusion)
# ----------------------------------------------------------------------
def node_to_hypervector(node: TreeNode, context_text: str) -> np.ndarray:
    """
    Encode a node as a high‑dimensional hypervector.
    1. Extract feature counts.
    2. Form a weighted sum of static feature hypervectors.
    3. Bind the result with a pheromone‑derived hypervector.
    4. Binarize to bipolar representation.
    """
    # 1. Feature counts
    counts = compute_regex_features(context_text)

    # 2. Weighted sum (preserve magnitude)
    agg = np.zeros(DIM, dtype=np.int32)
    for name in _FEATURE_ORDER:
        cnt = counts[name]
        if cnt != 0:
            agg += cnt * _FEATURE_HV[name].astype(np.int32)

    # Normalization to keep values in [-1,1] before binding
    max_abs = np.max(np.abs(agg)) or 1
    normed = np.where(agg >= 0, 1, -1).astype(np.int8)  # bipolar sign

    # 3. Bind with pheromone hypervector
    phv = pheromone_hypervector(node)
    bound = normed * phv  # element‑wise binding (XOR equivalent for bipolar)

    # 4. Final bipolar hypervector
    return bound

# ----------------------------------------------------------------------
# Multivector algebra (enhanced)
# ----------------------------------------------------------------------
class Multivector:
    """
    Simple geometric algebra implementation for grades 0 (scalar),
    1 (vector) and 2 (bivector). Blades are represented by sorted tuples
    of basis indices (0‑based). The geometric product follows the
    exterior algebra sign rule.
    """
    def __init__(self, components: dict[tuple[int, ...], float] | None = None):
        self.components: dict[tuple[int, ...], float] = {}
        if components:
            for blade, coeff in components.items():
                if abs(coeff) > 1e-12:
                    self.components[tuple(blade)] = float(coeff)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = Multivector(self.components)
        for blade, coeff in other.components.items():
            result.components[blade] = result.components.get(blade, 0.0) + coeff
            if abs(result.components[blade]) < 1e-12:
                del result.components[blade]
        return result

    def __iadd__(self, other: "Multivector") -> "Multivector":
        for blade, coeff in other.components.items():
            self.components[blade] = self.components.get(blade, 0.0) + coeff
            if abs(self.components[blade]) < 1e-12:
                del self.components[blade]
        return self

    def grade(self, k: int) -> "Multivector":
        return Multivector({b: c for b, c in self.components.items() if len(b) == k})

    def scalar_part(self) -> float:
        return self.components.get((), 0.0)

    @staticmethod
    def from_hypervector(hv: np.ndarray, grade: int = 1) -> "Multivector":
        """
        Lift a bipolar hypervector into a multivector.
        For grade‑1 we map each index i to blade (i,).
        For grade‑2 we generate all unordered pairs (i,j) with i<j and bind.
        """
        if grade == 0:
            return Multivector({(): float(np.mean(hv))})
        if grade == 1:
            comps = {(i,): float(val) for i, val in enumerate(hv)}
            return Multivector(comps)
        if grade == 2:
            # Approximate bivector by outer product of vector with itself,
            # keeping only i<j to avoid duplication.
            comps: dict[tuple[int, ...], float] = {}
            vec = hv.astype(np.float32)
            for i in range(DIM):
                vi = vec[i]
                if vi == 0:
                    continue
                for j in range(i + 1, DIM):
                    vj = vec[j]
                    if vj == 0:
                        continue
                    comps[(i, j)] = vi * vj
            return Multivector(comps)
        raise ValueError("Unsupported grade for lifting.")

    def geometric_product(self, other: "Multivector") -> "Multivector":
        """
        Compute geometric product using blade XOR and sign from permutation parity.
        Only works for blades represented as sorted tuples of distinct indices.
        """
        result = Multivector()
        for b1, c1 in self.components.items():
            for b2, c2 in other.components.items():
                # concatenate and sort while tracking swaps for sign
                merged = list(b1) + list(b2)
                sign = 1
                # bubble sort to count swaps (inefficient but clear)
                for i in range(len(merged)):
                    for j in range(i + 1, len(merged)):
                        if merged[i] > merged[j]:
                            merged[i], merged[j] = merged[j], merged[i]
                            sign = -sign
                # Remove duplicate indices (they square to scalar 1)
                uniq = []
                dup_count = {}
                for idx in merged:
                    dup_count[idx] = dup_count.get(idx, 0) + 1
                for idx, cnt in dup_count.items():
                    if cnt % 2 == 1:
                        uniq.append(idx)
                blade = tuple(sorted(uniq))
                coeff = sign * c1 * c2
                result.components[blade] = result.components.get(blade, 0.0) + coeff
        # Clean near‑zero entries
        result.components = {b: c for b, c in result.components.items() if abs(c) > 1e-12}
        return result

# ----------------------------------------------------------------------
# Aggregation across a traversal (deep fusion)
# ----------------------------------------------------------------------
def aggregate_multivector(nodes: list[TreeNode], texts: list[str]) -> Multivector:
    """
    For each node/text pair:
        * grade‑0: sum of pheromone levels (scalar)
        * grade‑1: sum of bound hypervectors (vector)
        * grade‑2: sum of bivector from outer product of vector with itself
    Returns a single Multivector encoding the whole traversal.
    """
    total = Multivector()
    for node, txt in zip(nodes, texts):
        hv = node_to_hypervector(node, txt)

        # scalar contribution
        scalar_mv = Multivector.from_hypervector(np.array([node.pheromone]), grade=0)
        total += scalar_mv

        # vector contribution
        vec_mv = Multivector.from_hypervector(hv, grade=1)
        total += vec_mv

        # bivector contribution (deeper interaction)
        biv_mv = Multivector.from_hypervector(hv, grade=2)
        total += biv_mv
    return total

# ----------------------------------------------------------------------
# Example usage (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = TreeNode("root")
    child1 = TreeNode("A", reward=2.0)
    child2 = TreeNode("B", reward=0.5)
    root.add_child(child1)
    root.add_child(child2)

    # Simulate updates
    update_pheromone(child1, reward=child1.reward)
    update_pheromone(child2, reward=child2.reward)

    texts = [
        "The evidence supports the plan despite delay.",
        "Risk and scarcity are evident in the strategy."
    ]

    mv = aggregate_multivector([child1, child2], texts)
    print("Scalar part:", mv.scalar_part())
    print("Number of vector blades:", len(mv.grade(1).components))
    print("Number of bivector blades:", len(mv.grade(2).components))