# DARWIN HAMMER — match 4418, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_pheromone_hyb_hybrid_hybrid_hybrid_m1483_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1257_s3.py (gen6)
# born: 2026-05-29T23:55:37Z

"""Hybrid MIS‑Bandit‑RBF Algorithm
Parent A: hybrid_pheromone_hybrid_distributed_l_m41_s1.py
Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1257_s3.py

Mathematical bridge:
- Nodes emit a pheromone level τ_i computed from the broadcast probability
  schedule of the MIS routine (Parent A).
- Each node is described by a feature vector ϕ_i (e.g. degree, τ_i) that feeds
  a radial‑basis‑function (RBF) surrogate model (Parent B) producing a predicted
  utility s_i.
- The utility is combined with epistemic certainty c_i (derived from a
  CertaintyFlag) to obtain a modulation factor μ_i = τ_i·s_i·(c_i/10000).
- μ_i acts as a propensity for a multi‑armed bandit policy that selects the
  next MIS leader. The bandit update is thus driven by pheromone‑informed
  RBF scores, fusing both topologies into a single selection rule.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types shared by both parents
# ----------------------------------------------------------------------
Node = Hashable = int  # simple integer identifiers for this example
Graph = Mapping[Node, set[Node]]

EPISTEMIC_FLAGS: Tuple[str, ...] = (
    "FACT",
    "PROBABLE",
    "POSSIBLE",
    "BULLSHIT",
    "SURE_MAYBE",
)

# ----------------------------------------------------------------------
# Parent‑A utilities (pheromone & MIS)
# ----------------------------------------------------------------------
def broadcast_probability(phase: int, step: int) -> float:
    """Return p = 1 / 2^(phase‑step), clamped to [0, 1]."""
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


def compute_pheromone_levels(
    graph: Graph, phases: int = 8, seed: int | str | None = None
) -> Dict[Node, float]:
    """Simulate pheromone emission for each node over *phases* broadcast rounds.

    The pheromone τ_i of node i is the sum of broadcast probabilities in which i
    was elected a temporary leader. A deterministic RNG makes the process
    reproducible.
    """
    rng = random.Random(seed)
    pheromones: Dict[Node, float] = {v: 0.0 for v in graph}
    undecided = set(graph)

    for phase in range(1, phases + 1):
        # each node draws a random number; the smallest wins the broadcast slot
        draws = {v: rng.random() for v in undecided}
        winner = min(draws, key=draws.get)
        p = broadcast_probability(phase, step=1)
        pheromones[winner] += p
        # remove winner and its neighbours from the current round (they listen)
        undecided -= {winner} | graph[winner]

        # restart round if all nodes have been considered
        if not undecided:
            undecided = set(graph)

    return pheromones


def maximal_independent_set(
    graph: Graph,
    propensities: Dict[Node, float],
    seed: int | str | None = None,
) -> set[Node]:
    """Greedy MIS using externally supplied propensities as selection scores.

    At each iteration the node with maximal propensity is added to the set,
    and it as well as its neighbours are removed from further consideration.
    """
    rng = random.Random(seed)
    remaining = set(graph)
    mis: set[Node] = set()

    while remaining:
        # break ties randomly but deterministically
        max_score = max(propensities[n] for n in remaining)
        candidates = [n for n in remaining if propensities[n] == max_score]
        chosen = rng.choice(candidates)
        mis.add(chosen)
        remaining -= {chosen} | graph[chosen]

    return mis


# ----------------------------------------------------------------------
# Parent‑B utilities (RBF surrogate & epistemic certainty)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(
                self,
                "generated_at",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def rbf_predict(
    features: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Radial‑basis‑function surrogate.

    Returns ϕ(x) = Σ_k w_k · exp(-‖x‑c_k‖² / (2σ²)).
    """
    if centers.shape[0] != weights.shape[0]:
        raise ValueError("centers and weights must have the same number of rows")
    diff = features[:, None, :] - centers[None, :, :]  # (n, m, d)
    sq_norm = np.einsum("nmd,nmd->nm", diff, diff)  # (n, m)
    kernels = np.exp(-sq_norm / (2.0 * sigma ** 2))  # (n, m)
    return kernels @ weights  # (n,)


# ----------------------------------------------------------------------
# Hybrid bridge: combine pheromone, RBF score, and epistemic certainty
# ----------------------------------------------------------------------
def compute_modulation_factors(
    graph: Graph,
    pheromones: Dict[Node, float],
    centers: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    certainty_map: Dict[Node, CertaintyFlag],
) -> Dict[Node, float]:
    """Return μ_i = τ_i · s_i · (c_i / 10000) for each node i.

    - τ_i : pheromone level (Parent A)
    - s_i : RBF‑predicted utility (Parent B)
    - c_i : confidence basis points from CertaintyFlag (0‑10000)
    """
    # Build feature matrix: [degree, pheromone]
    node_list = list(graph)
    degrees = np.array([len(graph[n]) for n in node_list], dtype=float)
    tau = np.array([pheromones[n] for n in node_list], dtype=float)
    features = np.column_stack((degrees, tau))

    # RBF prediction
    scores = rbf_predict(features, centers, weights, sigma)

    # Assemble modulation factors
    mu: Dict[Node, float] = {}
    for idx, n in enumerate(node_list):
        confidence = certainty_map.get(n, CertaintyFlag("POSSIBLE", 0, "N/A", "missing")).confidence_bps
        mu[n] = tau[idx] * scores[idx] * (confidence / 10000.0)
    return mu


# ----------------------------------------------------------------------
# High‑level hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_mis_bandit_rbf(
    graph: Graph,
    phases: int = 8,
    rbf_centers: np.ndarray | None = None,
    rbf_weights: np.ndarray | None = None,
    rbf_sigma: float = 1.0,
    seed: int | str | None = None,
) -> set[Node]:
    """Unified algorithm that fuses pheromone‑MIS, RBF surrogate, and epistemic certainty.

    1. Compute pheromone levels τ_i via broadcast rounds.
    2. Assign a default CertaintyFlag to every node (could be replaced by external data).
    3. Use τ_i together with node degree as RBF features → predicted utility s_i.
    4. Form modulation factors μ_i = τ_i·s_i·(c_i/10000) which act as bandit propensities.
    5. Run a greedy MIS selection guided by μ_i.
    """
    rng = random.Random(seed)

    # 1. Pheromones
    pheromones = compute_pheromone_levels(graph, phases=phases, seed=seed)

    # 2. Epistemic certainty (simple uniform placeholder)
    certainty_map: Dict[Node, CertaintyFlag] = {
        n: CertaintyFlag(
            label="FACT",
            confidence_bps=8000,
            authority_class="auto",
            rationale="default certainty",
        )
        for n in graph
    }

    # 3‑4. RBF prediction & modulation
    n_nodes = len(graph)
    if rbf_centers is None:
        # random centers in the 2‑D feature space
        rbf_centers = rng.random() * np.ones((4, 2))
    if rbf_weights is None:
        rbf_weights = rng.random(4)

    mu = compute_modulation_factors(
        graph,
        pheromones,
        centers=rbf_centers,
        weights=rbf_weights,
        sigma=rbf_sigma,
        certainty_map=certainty_map,
    )

    # 5. Greedy MIS using μ as propensities
    mis = maximal_independent_set(graph, propensities=mu, seed=seed)
    return mis


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple undirected graph (5‑node cycle)
    example_graph: Dict[int, set[int]] = {
        0: {1, 4},
        1: {0, 2},
        2: {1, 3},
        3: {2, 4},
        4: {3, 0},
    }

    mis_set = hybrid_mis_bandit_rbf(example_graph, phases=6, seed=42)
    print("Hybrid MIS result:", mis_set)
    # Verify that the result is indeed independent
    for v in mis_set:
        if any(nei in mis_set for nei in example_graph[v]):
            sys.exit("Error: result is not an independent set")
    print("Smoke test passed.")