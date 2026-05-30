# DARWIN HAMMER — match 5362, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_decreasing_pr_m1229_s2.py (gen5)
# born: 2026-05-30T00:02:54Z

"""Hybrid Textual‑Geometric Minimum‑Cost Tree

Parents:
- hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s4.py (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_decreasing_pr_m1229_s2.py (Algorithm B)

Mathematical bridge:
Algorithm A extracts posterior‑edge‑belief scalars from free‑form text via
regular‑expression counts (evidence, planning, delay, …).  Algorithm B
provides a fractional‑calculus Caputo kernel kα(t)=t^{α‑1}/Γ(α) that
modifies Euclidean distances for a minimum‑cost tree.

The hybrid defines a *belief‑scaled* Caputo distance

    d̂(i,j) = kα(‖x_i‑x_j‖) · b_{ij}

where b_{ij}=σ(Δ_{ij}) is a sigmoid of the difference between the
evidence‑richness and the delay‑richness of the two nodes (Δ_{ij} is the
pairwise belief derived from the textual features).  The resulting
distance matrix is fed to a Kruskal MST, yielding a tree that respects
both textual confidence and fractional geometric scaling.

The module implements:
1. `extract_text_features` – regex based count extraction (A).
2. `belief_factor` – conversion of feature vectors into a scalar belief
   (bridge).
3. `hybrid_distance` – Caputo‑scaled distance using the belief factor
   (fusion of A & B).
4. `minimum_cost_spanning_tree` – Kruskal algorithm on the hybrid
   distances (B’s tree logic enriched by A’s beliefs).

All functions are pure NumPy / std‑lib and can be used independently.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import re

# ----------------------------------------------------------------------
# Regex feature extraction – mirrors Algorithm A
# ----------------------------------------------------------------------
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
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicki)\b",
    re.I,
)


def extract_text_features(text: str) -> Dict[str, int]:
    """Count occurrences of each semantic class in *text*.

    Returns a dictionary with keys:
    evidence, planning, delay, support, boundary, outcome, impulsive
    """
    return {
        "evidence": len(EVIDENCE_RE.findall(text)),
        "planning": len(PLANNING_RE.findall(text)),
        "delay": len(DELAY_RE.findall(text)),
        "support": len(SUPPORT_RE.findall(text)),
        "boundary": len(BOUNDARY_RE.findall(text)),
        "outcome": len(OUTCOME_RE.findall(text)),
        "impulsive": len(IMPULSIVE_RE.findall(text)),
    }


# ----------------------------------------------------------------------
# Fractional calculus utilities – mirrors Algorithm B
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array(
    [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857,
    ]
)


def _gamma(z: float) -> float:
    """Lanczos approximation of the Gamma function."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * _gamma(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x


def caputo_kernel(alpha: float, t: np.ndarray) -> np.ndarray:
    """Caputo kernel k_α(t) = t^{α‑1} / Γ(α) for α>0."""
    if alpha <= 0:
        raise ValueError("Fractional order alpha must be positive.")
    # avoid division by zero
    t = np.where(t == 0, 1e-12, t)
    return t ** (alpha - 1) / _gamma(alpha)


def euclidean_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Standard ℓ₂ distance."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ----------------------------------------------------------------------
# Bridge: textual belief → scalar scaling factor
# ----------------------------------------------------------------------
def belief_factor(feat_i: Dict[str, int], feat_j: Dict[str, int]) -> float:
    """Compute a symmetric belief scaling between two feature dicts.

    The belief is higher when evidence & planning dominate delay &
    impulsive signals.  A sigmoid squashes the raw difference to (0,1).

    Parameters
    ----------
    feat_i, feat_j : dict
        Feature dictionaries as returned by ``extract_text_features``.

    Returns
    -------
    float
        Scaling factor in (0,1].
    """
    # aggregate positive vs negative signals
    pos_i = feat_i["evidence"] + feat_i["planning"]
    neg_i = feat_i["delay"] + feat_i["impulsive"] + 1e-9  # avoid zero
    pos_j = feat_j["evidence"] + feat_j["planning"]
    neg_j = feat_j["delay"] + feat_j["impulsive"] + 1e-9

    # pairwise belief difference (symmetric)
    delta = (pos_i / neg_i + pos_j / neg_j) / 2.0
    # sigmoid with temperature to keep scaling moderate
    temperature = 0.5
    return 1.0 / (1.0 + math.exp(-temperature * (delta - 1.0)))


# ----------------------------------------------------------------------
# Hybrid distance integrating Caputo kernel and belief factor
# ----------------------------------------------------------------------
def hybrid_distance(
    a: Tuple[float, ...],
    b: Tuple[float, ...],
    alpha: float,
    feat_a: Dict[str, int],
    feat_b: Dict[str, int],
) -> float:
    """Caputo‑scaled Euclidean distance weighted by textual belief.

    d̂ = kα(‖a‑b‖) · b_{ab}

    Parameters
    ----------
    a, b : tuple of floats
        Coordinate vectors of the two nodes.
    alpha : float
        Fractional order for the Caputo kernel (α>0).
    feat_a, feat_b : dict
        Textual feature dictionaries for the two nodes.

    Returns
    -------
    float
        Hybrid distance.
    """
    raw = euclidean_distance(a, b)
    kernel_val = caputo_kernel(alpha, np.array([raw]))[0]
    belief = belief_factor(feat_a, feat_b)
    return kernel_val * belief


# ----------------------------------------------------------------------
# Minimum‑Cost Spanning Tree on hybrid distances (Kruskal)
# ----------------------------------------------------------------------
class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True


def minimum_cost_spanning_tree(
    positions: List[Tuple[float, ...]],
    texts: List[str],
    alpha: float = 0.8,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Compute MST total cost and edge list using hybrid distances.

    Parameters
    ----------
    positions : list of coordinate tuples
        Spatial embedding of each node.
    texts : list of str
        Raw textual data associated with each node.
    alpha : float, optional
        Fractional order for the Caputo kernel (default 0.8).

    Returns
    -------
    total_cost : float
        Sum of hybrid distances of the selected edges.
    mst_edges : list of (int, int)
        Edge list (indices into ``positions``) of the MST.
    """
    if len(positions) != len(texts):
        raise ValueError("positions and texts must have the same length")

    n = len(positions)
    feats = [extract_text_features(t) for t in texts]

    # Build all possible edges with their hybrid weights
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = hybrid_distance(
                positions[i],
                positions[j],
                alpha,
                feats[i],
                feats[j],
            )
            edges.append((w, i, j))

    # Kruskal
    edges.sort(key=lambda e: e[0])
    uf = _UnionFind(n)
    mst_edges = []
    total_cost = 0.0
    for w, i, j in edges:
        if uf.union(i, j):
            mst_edges.append((i, j))
            total_cost += w
            if len(mst_edges) == n - 1:
                break

    return total_cost, mst_edges


# ----------------------------------------------------------------------
# Additional helper demonstrating a composite hybrid score
# ----------------------------------------------------------------------
def hybrid_score(
    positions: List[Tuple[float, ...]],
    texts: List[str],
    alpha: float = 0.8,
) -> float:
    """Composite score = MST cost + Shannon entropy of evidence counts.

    This showcases the full fusion: geometry (MST), fractional calculus
    (Caputo kernel), and information‑theoretic entropy from textual
    evidence (Algorithm A’s stylometry side).
    """
    # MST component
    mst_cost, _ = minimum_cost_spanning_tree(positions, texts, alpha)

    # Entropy component (only evidence counts)
    evidence_counts = np.array(
        [extract_text_features(t)["evidence"] for t in texts], dtype=float
    )
    total = evidence_counts.sum()
    if total == 0:
        entropy = 0.0
    else:
        probs = evidence_counts / total
        # avoid log(0)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))

    return mst_cost + entropy


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic example with 5 nodes
    positions = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.5, 0.5),
    ]

    texts = [
        "Evidence confirmed. Plan steps. No delay.",
        "Verified source. Checklist ready. Wait for approval.",
        "No evidence. Delay expected. Impulsive reaction.",
        "Proof logged. Schedule set. Support team ready.",
        "Audit record. Boundary respected. Outcome successful.",
    ]

    total_cost, edges = minimum_cost_spanning_tree(positions, texts, alpha=0.9)
    print("Hybrid MST total cost:", total_cost)
    print("Edges (index pairs):", edges)

    score = hybrid_score(positions, texts, alpha=0.9)
    print("Composite hybrid score (MST + entropy):", score)