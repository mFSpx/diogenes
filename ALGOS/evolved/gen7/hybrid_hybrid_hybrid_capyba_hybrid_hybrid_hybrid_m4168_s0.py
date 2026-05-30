# DARWIN HAMMER — match 4168, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_capybara_opti_hybrid_hybrid_minimu_m751_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bayes__m1736_s3.py (gen6)
# born: 2026-05-29T23:53:51Z

"""Hybrid Capybara‑Fisher Epistemic Optimization (HCFO)

This module fuses the core topologies of:

* **Parent A** – Capybara Optimization with social interaction,
  predator‑evasion and epistemic certainty flags.
* **Parent B** – Bayesian surrogate model enhanced by Fisher‑information
  derived likelihoods.

**Mathematical bridge**

The bridge is the *epistemic certainty* which in Parent A modulates edge
weights, and in Parent B is quantified by the Fisher information of a
Gaussian beam.  We therefore:

1. Use the Capybara social‑interaction and evasion dynamics to move the
   node vectors of a minimum‑cost tree.
2. Compute a Fisher‑information based certainty `F(θ)` for each edge
   (θ being the geometric angle of the edge).  This `F` replaces the
   discrete epistemic flags and multiplies the Euclidean distance,
   yielding a *certainty‑weighted* edge cost.
3. Perform a Bayesian update of hypothesis posteriors where the likelihood
   ratio is the normalized Fisher information of the supporting evidence.

The resulting algorithm jointly optimises node positions and edge costs
while maintaining a probabilistic belief over candidate solutions.

"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Vector = List[float]
Point = Tuple[float, float]
Edge = Tuple[int, int]               # indices of nodes in the node list
HypothesisID = str
EvidenceID = str

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------
def social_interaction(x: Vector, g_best: Vector, k: int = 1,
                       r1: float | None = None,
                       seed: int | str | None = None) -> Vector:
    """Capybara‑style attraction toward the global best."""
    if len(x) != len(g_best):
        raise ValueError("x and g_best must have same dimension")
    if k not in (1, 2):
        raise ValueError("k must be 1 or 2")
    rng = random.Random(seed)
    r = rng.random() if r1 is None else r1
    if not (0.0 <= r <= 1.0):
        raise ValueError("r1 must be in [0, 1]")
    return [xi + r * (gj - k * xi) for xi, gj in zip(x, g_best)]

def evasion_delta(t: int, t_max: int,
                  delta_max: float = 1.0,
                  alpha: float = 3.0) -> float:
    """Exponential decay schedule for predator evasion."""
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion parameters")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)

def predator_evasion(x: Vector, delta: float,
                     r2: float | None = None,
                     seed: int | str | None = None) -> Vector:
    """Random jitter scaled by the evasion factor."""
    rng = random.Random(seed)
    r = rng.random() if r2 is None else r2
    if not (0.0 <= r <= 1.0):
        raise ValueError("r2 must be in [0, 1]")
    return [xi + delta * (r - 0.5) for xi in x]

# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------
def euclidean(a: Vector, b: Vector) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile for a beam."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float,
                 eps: float = 1e-12) -> float:
    """Fisher information of the Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Hybrid components
# ----------------------------------------------------------------------
def angle_between(p1: Vector, p2: Vector) -> float:
    """Angle of the vector from p1 to p2 in radians."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(dy, dx)

def certainty_weighted_cost(p1: Vector, p2: Vector,
                           center: float = 0.0,
                           width: float = math.pi / 4) -> float:
    """
    Euclidean distance multiplied by a normalized Fisher‑information
    certainty factor derived from the edge orientation.
    """
    dist = euclidean(p1, p2)
    theta = angle_between(p1, p2)
    fisher = fisher_score(theta, center, width)
    # Normalise Fisher to [0,1] using a simple logistic map
    certainty = 1.0 / (1.0 + math.exp(-fisher))
    return dist * certainty

def update_node_positions(nodes: List[Vector],
                          g_best: Vector,
                          t: int,
                          t_max: int,
                          k: int = 1,
                          seed: int | str | None = None) -> List[Vector]:
    """
    Apply Capybara social interaction followed by predator evasion.
    The evasion factor decays with iteration `t`.
    """
    delta = evasion_delta(t, t_max)
    new_nodes = []
    for i, x in enumerate(nodes):
        # social pull
        y = social_interaction(x, g_best, k=k, seed=seed)
        # evasion jitter
        y = predator_evasion(y, delta, seed=seed)
        new_nodes.append(y)
    return new_nodes

def reweight_tree_edges(nodes: List[Vector],
                       edges: List[Edge],
                       center: float = 0.0,
                       width: float = math.pi / 4) -> Dict[Edge, float]:
    """
    Compute certainty‑weighted edge costs for the whole tree.
    Returns a mapping edge -> cost.
    """
    costs: Dict[Edge, float] = {}
    for (i, j) in edges:
        p1, p2 = nodes[i], nodes[j]
        costs[(i, j)] = certainty_weighted_cost(p1, p2,
                                                center=center,
                                                width=width)
    return costs

# ----------------------------------------------------------------------
# Bayesian layer using Fisher information as likelihood
# ----------------------------------------------------------------------
class MathHypothesis:
    def __init__(self, id: HypothesisID, prior: float,
                 evidence_ids: List[EvidenceID]):
        self.id = id
        self.prior = prior
        self.posterior = prior
        self.evidence_ids = evidence_ids

def bayesian_update(hypotheses: List[MathHypothesis],
                    evidence_angles: Dict[EvidenceID, float],
                    center: float = 0.0,
                    width: float = math.pi / 4) -> None:
    """
    Update each hypothesis posterior using Fisher‑information based likelihood.
    For a hypothesis, the likelihood is the product of normalized Fisher scores
    of its supporting evidence angles.
    """
    # compute normalising constant
    total_likelihood = 0.0
    likelihoods: Dict[HypothesisID, float] = {}
    for h in hypotheses:
        prod = 1.0
        for ev_id in h.evidence_ids:
            theta = evidence_angles.get(ev_id, 0.0)
            fisher = fisher_score(theta, center, width)
            # map Fisher to a probability‑like factor in (0,1]
            prob = 1.0 / (1.0 + math.exp(-fisher))
            prod *= prob
        likelihoods[h.id] = prod
        total_likelihood += h.prior * prod

    if total_likelihood == 0:
        raise RuntimeError("All likelihoods are zero; cannot update.")

    for h in hypotheses:
        num = h.prior * likelihoods[h.id]
        h.posterior = num / total_likelihood

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
def _smoke_test() -> None:
    # 1. initialise a tiny tree (4 nodes, 3 edges)
    rng = random.Random(42)
    nodes: List[Vector] = [[rng.random(), rng.random()] for _ in range(4)]
    edges: List[Edge] = [(0, 1), (1, 2), (2, 3)]

    # global best is the centroid of the initial nodes
    g_best = [sum(coord[i] for coord in nodes) / len(nodes) for i in range(2)]

    # 2. run hybrid position update
    updated_nodes = update_node_positions(nodes, g_best,
                                          t=5, t_max=20,
                                          k=1, seed=42)

    # 3. recompute edge costs with Fisher‑certainty weighting
    costs = reweight_tree_edges(updated_nodes, edges)

    # 4. set up a tiny Bayesian problem
    hypotheses = [
        MathHypothesis("H1", prior=0.6, evidence_ids=["e1", "e2"]),
        MathHypothesis("H2", prior=0.4, evidence_ids=["e3"])
    ]
    evidence_angles = {
        "e1": angle_between(updated_nodes[0], updated_nodes[1]),
        "e2": angle_between(updated_nodes[1], updated_nodes[2]),
        "e3": angle_between(updated_nodes[2], updated_nodes[3])
    }

    bayesian_update(hypotheses, evidence_angles)

    # 5. simple sanity prints (no assertions, just ensure no crash)
    print("Updated node positions:")
    for i, v in enumerate(updated_nodes):
        print(f"  Node {i}: {v}")
    print("\nEdge costs (certainty‑weighted):")
    for e, c in costs.items():
        print(f"  Edge {e}: {c:.4f}")
    print("\nPosterior probabilities:")
    for h in hypotheses:
        print(f"  {h.id}: {h.posterior:.4f}")

if __name__ == "__main__":
    _smoke_test()