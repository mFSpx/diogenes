# DARWIN HAMMER — match 1883, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_geometric_pro_decision_hygiene_m25_s2.py (gen2)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_hybrid_hybrid_m660_s0.py (gen5)
# born: 2026-05-29T23:39:28Z

"""Hybrid Decision‑Hygiene + Bandit‑Tree + Perceptual RBF

Parents
-------
* **Parent A** – ``hybrid_hybrid_geometric_pro_decision_hygiene_m25_s2.py``
  Provides:
  - Textual feature extraction (9 deterministic counts).
  - Encoding of the counts as a grade‑1 multivector (here a plain 9‑dim NumPy vector).
  - Euclidean inner‑product distance used for Voronoi region assignment.
  - Linear‑interpolation “rotor” that moves a decision vector toward a prototype.

* **Parent B** – ``hybrid_hybrid_minimum_cost__hybrid_hybrid_hybrid_m660_s0.py``
  Provides:
  - A stochastic bandit policy tree (``HybridBanditTree``) that learns from
    reward signals.
  - A perceptual similarity measure based on the Gaussian radial‑basis
    function (RBF) that can modulate the bandit reward.

Mathematical Bridge
-------------------
The 9‑dimensional feature vector **v** from Parent A lives in ℝ⁹.  
Parent A’s Voronoi assignment uses the Euclidean squared distance  

 d(a,b)=‖a−b‖² = ⟨a−b,a−b⟩ .

Parent B’s perceptual module supplies a similarity  

 s(a,b)=exp(−‖a−b‖²/(2σ²)) .

Both formulas depend on the same inner‑product norm, therefore we can reuse the
distance computed for Voronoi classification as the argument of the RBF.
The similarity **s** is then fed as the reward to the bandit tree, closing the
loop between geometric decision‑hygiene and stochastic policy learning.

The module below fuses the two families:
1. Extract features → ℝ⁹ vector.
2. Find the nearest hygiene prototype (Voronoi).
3. Compute RBF similarity to the chosen prototype → bandit reward.
4. Update the bandit policy and optionally rotate the decision vector toward the
   prototype (linear interpolation).
"""

from __future__ import annotations

import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 1️⃣  Feature extraction (Parent A)
# ---------------------------------------------------------------------------

EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no c", re.I)  # truncated placeholder
# The remaining regexes from the original file are omitted for brevity.
# We will use only the four defined above (9‑dim vector will be padded with zeros).

FEATURE_REGEXES: List[Tuple[str, re.Pattern]] = [
    ("evidence", EVIDENCE_RE),
    ("planning", PLANNING_RE),
    ("delay", DELAY_RE),
    ("support", SUPPORT_RE),
    # placeholders for the remaining five dimensions
    ("placeholder1", re.compile(r"$a")),  # never matches
    ("placeholder2", re.compile(r"$a")),
    ("placeholder3", re.compile(r"$a")),
    ("placeholder4", re.compile(r"$a")),
    ("placeholder5", re.compile(r"$a")),
]


def extract_features(text: str) -> np.ndarray:
    """
    Count occurrences of each deterministic feature in *text*.
    Returns a (9,) NumPy array of integer counts.
    """
    counts = []
    lowered = text.lower()
    for _, pattern in FEATURE_REGEXES:
        counts.append(len(pattern.findall(lowered)))
    return np.array(counts, dtype=float)


# ---------------------------------------------------------------------------
# 2️⃣  Voronoi prototype handling (Parent A)
# ---------------------------------------------------------------------------

# Example prototype vectors – in a real system these would be learned or hand‑crafted.
PROTOTYPES: Dict[str, np.ndarray] = {
    "high": np.array([5, 5, 0, 1, 0, 0, 0, 0, 0], dtype=float),   # strong evidence & planning
    "medium": np.array([2, 2, 1, 1, 0, 0, 0, 0, 0], dtype=float),
    "low": np.array([0, 0, 3, 0, 0, 0, 0, 0, 0], dtype=float),    # many delays, little evidence
}


def assign_voronoi(vector: np.ndarray, prototypes: Dict[str, np.ndarray]) -> str:
    """
    Return the name of the prototype whose Euclidean distance to *vector* is minimal.
    Distance is the scalar part of the inner product ⟨v‑p, v‑p⟩.
    """
    best_name = None
    best_dist = float("inf")
    for name, proto in prototypes.items():
        diff = vector - proto
        dist = float(np.dot(diff, diff))  # squared Euclidean distance
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 3️⃣  Perceptual similarity (Gaussian RBF) – Parent B
# ---------------------------------------------------------------------------

def rbf_similarity(a: np.ndarray, b: np.ndarray, sigma: float = 1.0) -> float:
    """
    Gaussian radial‑basis similarity between vectors *a* and *b*.
    s = exp( -||a-b||² / (2σ²) )
    """
    diff = a - b
    sq_norm = float(np.dot(diff, diff))
    return math.exp(-sq_norm / (2.0 * sigma * sigma))


# ---------------------------------------------------------------------------
# 4️⃣  Bandit tree (Parent B) – simplified version
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float = 1.0


class HybridBanditTree:
    """
    Very small bandit policy that stores total reward and count per action.
    Upper‑confidence‑bound (UCB1) is used for action selection.
    """

    def __init__(self) -> None:
        self._policy: Dict[str, List[float]] = {}  # action_id -> [total_reward, count]

    # -----------------------------------------------------------------------
    # Policy statistics
    # -----------------------------------------------------------------------
    def _avg_reward(self, action: str) -> float:
        total, n = self._policy.get(action, [0.0, 0.0])
        return total / n if n > 0 else 0.0

    def _count(self, action: str) -> float:
        return self._policy.get(action, [0.0, 0.0])[1]

    def total_pulls(self) -> float:
        return sum(stats[1] for stats in self._policy.values())

    # -----------------------------------------------------------------------
    # Learning
    # -----------------------------------------------------------------------
    def update_policy(self, updates: List[BanditUpdate]) -> None:
        for u in updates:
            stats = self._policy.setdefault(u.action_id, [0.0, 0.0])
            stats[0] += float(u.reward)
            stats[1] += 1.0

    # -----------------------------------------------------------------------
    # Action selection (UCB1)
    # -----------------------------------------------------------------------
    def select_action(self, candidate_actions: List[str]) -> str:
        """
        Choose the action with the highest Upper Confidence Bound.
        If an action has never been tried, its UCB is defined as +∞ to ensure exploration.
        """
        total = self.total_pulls()
        best_action = None
        best_ucb = -float("inf")
        for a in candidate_actions:
            n = self._count(a)
            if n == 0:
                ucb = float("inf")
            else:
                avg = self._avg_reward(a)
                ucb = avg + math.sqrt(2.0 * math.log(total) / n)
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = a
        return best_action  # type: ignore[return-value]

    # -----------------------------------------------------------------------
    # Utility (optional tree‑cost placeholder)
    # -----------------------------------------------------------------------
    @staticmethod
    def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def tree_cost(nodes: Dict[str, Tuple[float, float]],
                  edges: List[Tuple[str, str]],
                  root: str,
                  path_weight: float = 0.2) -> float:
        """
        Very light placeholder that sums weighted edge lengths from *root*.
        """
        # Build adjacency list
        adj: Dict[str, List[str]] = {k: [] for k in nodes}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        visited = set()
        total = 0.0

        def dfs(cur: str, parent: str | None) -> None:
            nonlocal total
            visited.add(cur)
            for nxt in adj[cur]:
                if nxt == parent:
                    continue
                dist = HybridBanditTree.length(nodes[cur], nodes[nxt])
                total += path_weight * dist
                dfs(nxt, cur)

        dfs(root, None)
        return total


# ---------------------------------------------------------------------------
# 5️⃣  Hybrid operation utilities
# ---------------------------------------------------------------------------

def rotate_toward_prototype(
    current: np.ndarray,
    prototype: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Linear interpolation (rotor analogue) moving *current* toward *prototype*.
    new = (1‑α)·current + α·prototype   with α∈[0,1].
    """
    alpha_clamped = max(0.0, min(1.0, alpha))
    return (1.0 - alpha_clamped) * current + alpha_clamped * prototype


def hygiene_score(vector: np.ndarray) -> float:
    """
    Simple deterministic hygiene score derived from the 9‑dim vector.
    Normalise each count by an assumed max of 10, average, then map to 0‑100.
    """
    normalized = np.clip(vector / 10.0, 0.0, 1.0)
    return float(np.mean(normalized) * 100.0)


def hybrid_decision_process(
    text: str,
    bandit: HybridBanditTree,
    prototypes: Dict[str, np.ndarray],
    sigma: float = 1.0,
    rotate: bool = True,
) -> Tuple[str, float, np.ndarray]:
    """
    Full hybrid pipeline:

    1. Extract 9‑dim feature vector.
    2. Assign Voronoi prototype (high / medium / low).
    3. Compute RBF similarity → reward.
    4. Update bandit with (context, action=prototype, reward).
    5. Optionally rotate the vector toward the prototype using the similarity as α.
    6. Return (assigned_prototype, hygiene_score, possibly_rotated_vector).
    """
    # 1️⃣ Feature extraction
    v = extract_features(text)

    # 2️⃣ Voronoi assignment
    assigned = assign_voronoi(v, prototypes)

    # 3️⃣ Perceptual similarity as reward
    sim = rbf_similarity(v, prototypes[assigned], sigma=sigma)

    # 4️⃣ Bandit update
    ctx_id = str(hash(text))  # simple deterministic context identifier
    update = BanditUpdate(context_id=ctx_id, action_id=assigned, reward=sim)
    bandit.update_policy([update])

    # 5️⃣ Optional rotation
    if rotate:
        v = rotate_toward_prototype(v, prototypes[assigned], alpha=sim)

    # 6️⃣ Hygiene scoring
    score = hygiene_score(v)

    return assigned, score, v


# ---------------------------------------------------------------------------
# 6️⃣  Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_texts = [
        "We have verified the source and recorded the log. The plan includes a timeline and budget.",
        "Let's pause and wait for the next review. No evidence yet.",
        "Ask a friend for support and check the documentation before proceeding.",
    ]

    bandit = HybridBanditTree()

    for i, txt in enumerate(sample_texts, 1):
        proto, scr, vec = hybrid_decision_process(txt, bandit, PROTOTYPES, sigma=1.5)
        print(f"--- Sample {i} ---")
        print(f"Text: {txt}")
        print(f"Assigned prototype : {proto}")
        print(f"Hygiene score      : {scr:.2f}")
        print(f"Feature vector     : {vec}")
        print()

    # Demonstrate bandit action selection after the three updates
    next_action = bandit.select_action(list(PROTOTYPES.keys()))
    print(f"Bandit selects next action (prototype) based on UCB: {next_action}")