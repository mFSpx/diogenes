# DARWIN HAMMER — match 3171, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_hybrid_m310_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2701_s0.py (gen5)
# born: 2026-05-29T23:48:26Z

"""Hybrid Algorithm combining pheromone‑guided tree search (Parent A) with
high‑dimensional computing and multivector algebra (Parent B).

Mathematical Bridge
-------------------
Parent A builds a search tree where each node carries a *pheromone* value that
reflects the quality of past traversals.  Parent B represents discrete feature
vectors as high‑dimensional hypervectors (HDC) and aggregates them with a
multivector whose grades encode different semantic layers.

The fusion treats the pheromone intensity of a node as a *dynamic weight* that
modulates the contribution of each feature when the node is encoded as a
hypervector.  Concretely:

* For a node we extract a raw feature count vector **c** (e.g. depth,
  branching factor, reward).
* The pheromone **τ** of the node is used to bias the global positive/negative
  weight vectors **w⁺**, **w⁻** from Parent B:
  
      wᵢ(τ) = w⁺ᵢ·σ(τ) – w⁻ᵢ·(1‑σ(τ))

  where σ is the logistic sigmoid mapping pheromone to (0,1).
* The weighted dot product `v = sign(w(τ)·c)` yields a scalar that is tiled
  into a DIM‑dimensional hypervector.
* Hypervectors from all nodes of a traversal are lifted into a
  **Multivector**; grade‑0 stores the aggregated scalar part, grade‑1 stores the
  element‑wise sum of hypervectors, etc.  This provides a unified algebraic
  representation of both the search dynamics (pheromones) and the high‑dimensional
  encoding (HDC).

The three core functions below illustrate this hybrid pipeline:
`update_pheromone`, `node_to_hypervector`, and `aggregate_multivector`. """

import math
import random
import sys
from pathlib import Path
import json
import hashlib
import re
from datetime import datetime, timezone
import numpy as np

# ----------------------------------------------------------------------
# Constants from Parent B
# ----------------------------------------------------------------------
DIM = 10000                     # HDC dimensionality
_POSITIVE_WEIGHTS = np.array(
    [1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64
)
_NEGATIVE_WEIGHTS = np.array(
    [0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64
)

# ----------------------------------------------------------------------
# Simple tree node with pheromone (Parent A concept)
# ----------------------------------------------------------------------
class TreeNode:
    """A node in a search tree carrying a pheromone level."""
    __slots__ = ("name", "children", "pheromone", "reward")

    def __init__(self, name: str, reward: float = 0.0):
        self.name = name
        self.children: list[TreeNode] = []
        self.pheromone: float = 0.1          # small initial pheromone
        self.reward: float = reward

    def add_child(self, child: "TreeNode") -> None:
        self.children.append(child)

# ----------------------------------------------------------------------
# Helper functions (shared utilities)
# ----------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sigmoid(x: float) -> float:
    """Logistic sigmoid mapping any real number to (0,1)."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# ----------------------------------------------------------------------
# 1️⃣ Pheromone update (inherits Parent A dynamics)
# ----------------------------------------------------------------------
def update_pheromone(node: TreeNode, reward: float, decay: float = 0.9) -> None:
    """
    Update the pheromone level of a node using a simple reinforcement rule.

    τ_new = decay * τ_old + (1 - decay) * reward
    """
    old = node.pheromone
    node.pheromone = decay * old + (1.0 - decay) * reward
    # Clamp to a reasonable positive range
    node.pheromone = max(0.01, min(node.pheromone, 10.0))


# ----------------------------------------------------------------------
# 2️⃣ Feature extraction (stub of Parent B regex features)
# ----------------------------------------------------------------------
# Minimal regex patterns to illustrate the idea
EVIDENCE_RE = re.compile(r"\b(evidence|proof)\b", re.IGNORECASE)
PLANNING_RE = re.compile(r"\b(plan|strategy)\b", re.IGNORECASE)
DELAY_RE = re.compile(r"\b(delay|wait)\b", re.IGNORECASE)

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
    """
    Very lightweight extraction of three demo features.
    Real implementation would include all nine features.
    """
    return {
        "evidence": len(EVIDENCE_RE.findall(text)),
        "planning": len(PLANNING_RE.findall(text)),
        "delay": len(DELAY_RE.findall(text)),
        # placeholders for the remaining features
        "support": 0,
        "boundary": 0,
        "outcome": 0,
        "impulsive": 0,
        "scarcity": 0,
        "risk": 0,
    }


# ----------------------------------------------------------------------
# 3️⃣ Hypervector construction using pheromone‑biased weights (core bridge)
# ----------------------------------------------------------------------
def node_to_hypervector(node: TreeNode, context_text: str) -> np.ndarray:
    """
    Encode a tree node as a high‑dimensional hypervector.

    Steps:
    1. Extract regex feature counts from `context_text`.
    2. Bias the global weight vector with the node's pheromone via a sigmoid.
    3. Compute a signed scalar `v = sign(w(τ) · c)`.
    4. Tile `v` into a DIM‑dimensional vector (as in Parent B).
    """
    # 1. Feature counts
    raw_counts = compute_regex_features(context_text)
    # Preserve order defined by _FEATURE_ORDER
    feature_vector = np.array(
        [raw_counts[key] for key in _FEATURE_ORDER], dtype=np.int64
    )

    # 2. Pheromone‑biased weights
    tau = node.pheromone
    bias = sigmoid(tau)                     # maps pheromone to (0,1)
    w = _POSITIVE_WEIGHTS * bias - _NEGATIVE_WEIGHTS * (1.0 - bias)

    # 3. Signed scalar
    dot = int(np.dot(w, feature_vector))
    v = np.sign(dot) if dot != 0 else 0

    # 4. Tile into hypervector
    hypervector = np.full(DIM, v, dtype=np.int8)
    return hypervector


# ----------------------------------------------------------------------
# 4️⃣ Multivector algebra (Parent B)
# ----------------------------------------------------------------------
class Multivector:
    """Element of a geometric algebra Cl(n,0) represented as a dict of blades."""

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
        """Return a new Multivector containing only blades of grade k."""
        return Multivector(
            {blade: coeff for blade, coeff in self.components.items() if len(blade) == k}
        )

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) component."""
        return self.components.get((), 0.0)

    @staticmethod
    def from_hypervector(hv: np.ndarray, grade: int = 1) -> "Multivector":
        """
        Lift a hypervector into a multivector of specified grade.
        For simplicity we treat each index as a basis blade.
        """
        comp = { (i,): float(val) for i, val in enumerate(hv) if val != 0 }
        return Multivector(comp)


# ----------------------------------------------------------------------
# 5️⃣ Aggregation of a traversal into a single Multivector
# ----------------------------------------------------------------------
def aggregate_multivector(nodes: list[TreeNode], texts: list[str]) -> Multivector:
    """
    Convert a list of nodes (with accompanying context texts) into hypervectors,
    then combine them into a single Multivector.  Grade‑0 accumulates the summed
    rewards, while grade‑1 holds the superposition of hypervectors.
    """
    if len(nodes) != len(texts):
        raise ValueError("nodes and texts must have the same length")

    total_reward = sum(node.reward for node in nodes)
    mv = Multivector({(): total_reward})          # scalar part

    for node, txt in zip(nodes, texts):
        hv = node_to_hypervector(node, txt)
        mv = mv + Multivector.from_hypervector(hv, grade=1)

    return mv


# ----------------------------------------------------------------------
# 6️⃣ Demonstration of hybrid operation
# ----------------------------------------------------------------------
def demo_hybrid_process() -> None:
    """Build a tiny tree, run pheromone updates, and produce a multivector."""
    # Build tree
    root = TreeNode("root", reward=1.0)
    child_a = TreeNode("A", reward=2.0)
    child_b = TreeNode("B", reward=-1.0)
    root.add_child(child_a)
    root.add_child(child_b)
    child_a.add_child(TreeNode("A1", reward=0.5))
    child_b.add_child(TreeNode("B1", reward=1.5))

    # Simulated traversal (depth‑first)
    traversal = [root, child_a, child_a.children[0], child_b, child_b.children[0]]

    # Dummy context texts (could be generated from node data)
    texts = [
        "Evidence of planning was found.",
        "The plan was delayed due to lack of support.",
        "No evidence, just speculation.",
        "Strategic evidence supports the outcome.",
        "Risk and scarcity were evident."
    ]

    # Update pheromones based on reward
    for node in traversal:
        update_pheromone(node, reward=node.reward)

    # Aggregate into a multivector
    mv = aggregate_multivector(traversal, texts)

    # Simple inspection
    print("Scalar (total reward):", mv.scalar_part())
    print("Number of hypervector blades (grade‑1):", len(mv.grade(1).components))


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_hybrid_process()