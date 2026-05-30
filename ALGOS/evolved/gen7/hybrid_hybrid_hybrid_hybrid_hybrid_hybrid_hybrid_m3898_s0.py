# DARWIN HAMMER — match 3898, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_caputo_fracti_m1436_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1915_s3.py (gen5)
# born: 2026-05-29T23:52:27Z

"""Hybrid Algorithm integrating Sheaf‑Caputo topology (Parent A) with Linguistic Style Matching (Parent B)

Mathematical Bridge
-------------------
* **Node sections** of the `Sheaf` are instantiated with the LSM feature vector
  `v ∈ ℝ⁹` derived from a textual context.  
* The **Caputo fractional weight** `γ(t,α)=t^{α‑1}/Γ(α)·Σ_jγ_j` (parent A) scales the
  Euclidean norm of each edge restriction matrix, producing a hybrid edge weight
  `w_{uv}=γ(t,α)·‖R_{uv}‖`.  
* A **Radial‑Basis‑Function surrogate** receives the concatenated vector
  `[v, w_{uv}]` (node features + edge weight) and predicts a scalar reward.
* A simple **contextual bandit** then combines the surrogate prediction with
  bandit‑specific statistics (`propensity`, `expected_reward`, `confidence_bound`)
  to select an action.

The code below fuses these components into three demonstrative functions:
`lsm_vector`, `compute_hybrid_edge_weight`, and `select_bandit_action`. """

import math
import re
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent B – Linguistic Style Matching (LSM) utilities
# ----------------------------------------------------------------------
_EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
_PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|"
    r"criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
_DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|"
    r"before i|first|after|review)\b",
    re.I,
)
_SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|"
    r"support|help|community|team|handoff|delegate)\b",
    re.I,
)
_BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact)\b", re.I)

_FEATURE_ORDER: List[str] = [
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

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.float64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.float64)

def _safe_divide(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0

def lsm_vector(text: str) -> np.ndarray:
    """Return a 9‑dimensional LSM feature vector for *text*."""
    counts = np.zeros(len(_FEATURE_ORDER), dtype=np.float64)

    # evidence, planning, delay, support, boundary are detected via regexes
    regex_map = {
        0: _EVIDENCE_RE,
        1: _PLANNING_RE,
        2: _DELAY_RE,
        3: _SUPPORT_RE,
        4: _BOUNDARY_RE,
    }
    for idx, regex in regex_map.items():
        counts[idx] = len(regex.findall(text))

    # outcome, impulsive, scarcity, risk are approximated by keyword heuristics
    outcome_keywords = r"\b(?:success|win|achieve|result|outcome|gain)\b"
    impulsive_keywords = r"\b(?:impulse|rash|spontaneous|quickly|now|immediate)\b"
    scarcity_keywords = r"\b(?:limited|rare|exclusive|only|few|once)\b"
    risk_keywords = r"\b(?:danger|risk|threat|hazard|uncertain|dangerous)\b"

    counts[5] = len(re.findall(outcome_keywords, text, re.I))
    counts[6] = len(re.findall(impulsive_keywords, text, re.I))
    counts[7] = len(re.findall(scarcity_keywords, text, re.I))
    counts[8] = len(re.findall(risk_keywords, text, re.I))

    # Weighted combination: positive – negative
    weighted = _POSITIVE_WEIGHTS * counts - _NEGATIVE_WEIGHTS * counts
    return weighted

# ----------------------------------------------------------------------
# Parent A – Sheaf, Caputo weight, and RBF surrogate
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: List[float]) -> float:
        return sum(
            w * math.exp(-((self.epsilon * self.euclidean(x, list(c))) ** 2))
            for w, c in zip(self.weights, self.centers)
        )

    @staticmethod
    def euclidean(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            raise ValueError("vectors must have same dimension")
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def gamma_term(t: float, alpha: float, sum_j_gamma: float) -> float:
    """Caputo‑type fractional coefficient."""
    if t <= 0 or alpha <= 0:
        return 0.0
    return (t ** (alpha - 1)) / math.gamma(alpha) * sum_j_gamma

class Sheaf:
    """Topological data structure with Caputo‑weighted edges."""
    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]]):
        self.node_dims = dict(node_dims)          # node → dimension
        self.edges = list(edge_list)              # list of (u,v)
        self._restrictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[int, np.ndarray] = {}
        self._caputo_weights: Dict[Tuple[int, int], float] = {}

    def set_restriction(self, edge: Tuple[int, int], src_map: List[List[float]], dst_map: List[List[float]]) -> None:
        u, v = edge
        src = np.array(src_map, dtype=float)
        dst = np.array(dst_map, dtype=float)
        self._restrictions[(u, v)] = (src, dst)

    def set_section(self, node: int, value: List[float]) -> None:
        self._sections[node] = np.array(value, dtype=float)

    def set_caputo_weight(self, edge: Tuple[int, int], t: float, alpha: float, sum_j_gamma: float) -> None:
        self._caputo_weights[edge] = gamma_term(t, alpha, sum_j_gamma)

    def get_section(self, node: int) -> np.ndarray:
        return self._sections[node]

    def hybrid_edge_weight(self, edge: Tuple[int, int]) -> float:
        """γ(t,α)·‖R_{uv}‖ where R is the source‑map of the restriction."""
        if edge not in self._caputo_weights:
            raise KeyError(f"Caputo weight not defined for edge {edge}")
        caputo = self._caputo_weights[edge]
        if edge in self._restrictions:
            src_map = self._restrictions[edge][0]
            norm = np.linalg.norm(src_map)
            return caputo * norm
        # if restriction defined in opposite direction, use that source map
        rev = (edge[1], edge[0])
        if rev in self._restrictions:
            src_map = self._restrictions[rev][0]
            norm = np.linalg.norm(src_map)
            return caputo * norm
        # No restriction → weight equals caputo term alone
        return caputo

# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def compute_node_section(sheaf: Sheaf, node: int, text: str) -> None:
    """Populate a node's section with the LSM feature vector derived from *text*."""
    vec = lsm_vector(text)
    sheaf.set_section(node, vec.tolist())

def compute_hybrid_edge_weight(sheaf: Sheaf, edge: Tuple[int, int], t: float, alpha: float, sum_j_gamma: float) -> float:
    """Create/refresh the Caputo weight and return the hybrid edge weight."""
    sheaf.set_caputo_weight(edge, t, alpha, sum_j_gamma)
    return sheaf.hybrid_edge_weight(edge)

def select_bandit_action(
    actions: List[BanditAction],
    context_vec: np.ndarray,
    surrogate: RBFSurrogate,
    sheaf: Sheaf,
    edge: Tuple[int, int],
) -> BanditAction:
    """
    Combine bandit statistics with the surrogate's prediction.
    The surrogate receives the concatenation of the context vector and the hybrid edge weight.
    """
    edge_weight = sheaf.hybrid_edge_weight(edge)
    input_vec = np.concatenate([context_vec, np.array([edge_weight])]).tolist()
    pred = surrogate.predict(input_vec)

    # Score = propensity·expected_reward + confidence + α·prediction
    alpha = 0.5  # blending factor
    best = None
    best_score = -math.inf
    for act in actions:
        score = act.propensity * act.expected_reward + act.confidence_bound + alpha * pred
        if score > best_score:
            best_score = score
            best = act
    return best

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny sheaf with two nodes and one edge
    node_dims = {0: 9, 1: 9}
    edges = [(0, 1)]
    sheaf = Sheaf(node_dims, edges)

    # Populate node sections using LSM on example sentences
    txt0 = "I have evidence that the plan succeeded despite the risk."
    txt1 = "We need to pause, gather more support and verify the source."
    compute_node_section(sheaf, 0, txt0)
    compute_node_section(sheaf, 1, txt1)

    # Define a simple restriction (identity) between the nodes
    identity = np.eye(9).tolist()
    sheaf.set_restriction((0, 1), identity, identity)

    # Compute hybrid edge weight
    t, alpha, sum_j_gamma = 2.0, 0.8, 1.5
    w = compute_hybrid_edge_weight(sheaf, (0, 1), t, alpha, sum_j_gamma)
    print(f"Hybrid edge weight: {w:.4f}")

    # Build a surrogate with two random centers
    rng = np.random.default_rng(42)
    centers = [tuple(rng.normal(size=10)) for _ in range(2)]  # 9 dims + 1 edge weight = 10
    weights = list(rng.uniform(-1, 1, size=2))
    surrogate = RBFSurrogate(centers=centers, weights=weights, epsilon=0.5)

    # Define three mock bandit actions
    actions = [
        BanditAction("A", propensity=0.6, expected_reward=1.2, confidence_bound=0.1),
        BanditAction("B", propensity=0.3, expected_reward=2.0, confidence_bound=0.2),
        BanditAction("C", propensity=0.9, expected_reward=0.5, confidence_bound=0.05),
    ]

    # Choose action based on node 0 context
    ctx_vec = sheaf.get_section(0)
    chosen = select_bandit_action(actions, ctx_vec, surrogate, sheaf, (0, 1))
    print(f"Chosen action: {chosen.action_id}")