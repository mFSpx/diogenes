# DARWIN HAMMER — match 3953, survivor 0
# gen: 5
# parent_a: hybrid_infotaxis_hybrid_semantic_neig_m739_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_bayes_claim_k_m9_s2.py (gen4)
# born: 2026-05-29T23:52:45Z

"""Hybrid Infotaxis–Semantic–Pheromone–Bayesian System
Parents:
- hybrid_infotaxis_hybrid_semantic_neig_m739_s4.py (entropy‑based infotaxis with morphology‑scaled semantic affinity)
- hybrid_hybrid_hybrid_pherom_hybrid_bayes_claim_k_m9_s2.py (pheromone probability, decision hygiene scores, Shannon entropy, Bayesian update, tree cost)

Mathematical Bridge:
Both parents treat a set of scalar scores as a probability‑like mass and evaluate
uncertainty with Shannon entropy.  The bridge is therefore the *probability
distribution* over candidate actions.  In the fused algorithm we:

1. Build a hybrid affinity `h = c * p` where `c` is cosine similarity and `p`
   is the recovery priority derived from morphology.
2. Interpret the affinities as a prior distribution over actions.
3. Use pheromone‑derived probabilities as likelihoods and decision‑hygiene
   scores as additional evidence, performing a Bayesian update:
   `posterior ∝ prior * likelihood * evidence`.
4. Normalize the posterior to obtain a proper distribution and compute its
   Shannon entropy, which drives the infotaxis‑style action ranking.
5. The posterior also weights a minimum‑cost tree evaluation, linking the
   routing cost to the same probabilistic view.

The following module implements this unified pipeline."""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Morphology & Recovery Priority (Parent A core)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """A simple shape descriptor; higher means more spherical."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (length+width)/(2*height)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Physical proxy for how long a body needs to right itself."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """
    Normalized priority p∈[0,1] derived from righting‑time index.
    The larger the index, the lower the priority (harder to recover).
    """
    idx = righting_time_index(m)
    # Clip to [0, max_index] then invert and normalise.
    idx = max(0.0, min(max_index, idx))
    return 1.0 - (idx / max_index)


# ----------------------------------------------------------------------
# Hybrid Affinity & Entropy (Parent A + Parent B core)
# ----------------------------------------------------------------------
def hybrid_affinity(cosine_similarity: float, priority_other: float) -> float:
    """Morphology‑modulated semantic affinity h = c * p."""
    return cosine_similarity * priority_other


def normalize(vec: np.ndarray) -> np.ndarray:
    """L1‑normalize a non‑negative vector; returns zeros if sum==0."""
    total = vec.sum()
    if total == 0:
        return np.zeros_like(vec)
    return vec / total


def shannon_entropy(probabilities: np.ndarray) -> float:
    """Standard Shannon entropy H = -∑ p log2 p (ignoring zero entries)."""
    mask = probabilities > 0
    return -float(np.sum(probabilities[mask] * np.log2(probabilities[mask])))


def bayesian_update_distribution(prior: np.ndarray,
                                 likelihood: np.ndarray,
                                 evidence: np.ndarray) -> np.ndarray:
    """
    Element‑wise Bayesian update:
        posterior ∝ prior * likelihood * evidence
    The `evidence` argument plays the role of an additional likelihood
    (e.g., decision‑hygiene scores normalised to probabilities).
    Returns a normalized posterior distribution.
    """
    unnorm = prior * likelihood * evidence
    return normalize(unnorm)


# ----------------------------------------------------------------------
# Pheromone & Decision Hygiene (Parent B core)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def calculate_pheromone_probabilities(surface_key: str,
                                      limit: int,
                                      db_url: str | None = None) -> np.ndarray:
    """Stub: return a random probability vector of length `limit`."""
    raw = np.array([random.random() for _ in range(limit)], dtype=float)
    return normalize(raw)


def decision_hygiene_scores(text: str) -> np.ndarray:
    """
    Stub: produce a probability vector from mock hygiene scores.
    The real implementation would parse `text`; here we use a deterministic
    random seed for reproducibility.
    """
    random.seed(hash(text) & 0xffffffff)
    scores = np.array([random.randint(1, 5) for _ in range(5)], dtype=float)
    return normalize(scores)


# ----------------------------------------------------------------------
# Tree Cost Evaluation
# ----------------------------------------------------------------------
def tree_cost(nodes: Dict[str, Point],
              edges: List[Edge],
              root: str,
              path_weight: float = 0.2) -> float:
    """
    Simple cost: sum of Euclidean distances for each edge multiplied by a
    constant weight.  No cycle detection – assumes a tree structure.
    """
    cost = 0.0
    for u, v in edges:
        if u not in nodes or v not in nodes:
            continue
        pu, pv = nodes[u], nodes[v]
        dist = math.hypot(pu[0] - pv[0], pu[1] - pv[1])
        cost += path_weight * dist
    return cost


# ----------------------------------------------------------------------
# Unified Hybrid Operations (≥3 functions)
# ----------------------------------------------------------------------
def compute_action_distribution(cosine_similarities: List[float],
                                morphology_self: Morphology,
                                morphologies_others: List[Morphology]) -> np.ndarray:
    """
    Build a prior distribution over actions using hybrid affinities.
    Each action corresponds to a neighbour; its affinity is
        h_i = c_i * p_i
    where p_i is the recovery priority of the neighbour.
    """
    if len(cosine_similarities) != len(morphologies_others):
        raise ValueError("Lengths of similarities and neighbour morphologies must match")
    affinities = []
    for c, m in zip(cosine_similarities, morphologies_others):
        p = recovery_priority(m)
        affinities.append(hybrid_affinity(c, p))
    return normalize(np.array(affinities, dtype=float))


def fuse_with_pheromone_and_hygiene(prior: np.ndarray,
                                    surface_key: str,
                                    text_evidence: str) -> np.ndarray:
    """
    Apply Bayesian fusion:
      posterior ∝ prior * pheromone_likelihood * hygiene_evidence
    """
    pheromone_likelihood = calculate_pheromone_probabilities(surface_key,
                                                             limit=len(prior))
    hygiene_evidence = decision_hygiene_scores(text_evidence)
    # Resize evidence if dimensions differ (simple repeat/truncate)
    if len(hygiene_evidence) != len(prior):
        # repeat to match length
        repeats = (len(prior) + len(hygiene_evidence) - 1) // len(hygiene_evidence)
        hygiene_evidence = np.tile(hygiene_evidence, repeats)[:len(prior)]
    return bayesian_update_distribution(prior, pheromone_likelihood, hygiene_evidence)


def select_optimal_action(cosine_similarities: List[float],
                          self_morph: Morphology,
                          neighbour_morphs: List[Morphology],
                          surface_key: str,
                          text_evidence: str,
                          nodes: Dict[str, Point],
                          edges: List[Edge],
                          root: str) -> Tuple[int, float]:
    """
    End‑to‑end pipeline:
      1. Build prior from hybrid affinities.
      2. Fuse with pheromone & hygiene via Bayesian update.
      3. Compute Shannon entropy of the posterior (infotaxis metric).
      4. Weight the tree cost by the posterior probability of each action.
      5. Return the index of the action minimising a combined score:
           score = entropy + λ * expected_tree_cost
    """
    λ = 0.5  # trade‑off coefficient between uncertainty and routing cost

    prior = compute_action_distribution(cosine_similarities, self_morph, neighbour_morphs)
    posterior = fuse_with_pheromone_and_hygiene(prior, surface_key, text_evidence)

    entropy = shannon_entropy(posterior)

    # Expected tree cost: sum_i posterior_i * cost_i where each cost_i is
    # the cost of a tree rooted at neighbour i (here we reuse the same tree
    # for simplicity and weight it by posterior).
    base_cost = tree_cost(nodes, edges, root)
    expected_cost = posterior.mean() * base_cost  # simplistic expectation

    combined_score = entropy + λ * expected_cost
    best_idx = int(np.argmin(posterior))  # choose most certain (highest posterior)
    return best_idx, combined_score


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Mock data
    self_morph = Morphology(length=2.0, width=1.5, height=1.0, mass=3.0)

    neighbour_morphs = [
        Morphology(2.1, 1.4, 1.0, 2.9),
        Morphology(1.9, 1.6, 1.2, 3.1),
        Morphology(2.0, 1.5, 0.9, 2.8),
    ]

    cosine_similarities = [0.8, 0.4, 0.6]

    surface_key = "arena_floor"
    text_evidence = "sample decision context"

    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, 0.8),
    }

    edges = [("A", "B"), ("A", "C")]

    best_action, score = select_optimal_action(
        cosine_similarities,
        self_morph,
        neighbour_morphs,
        surface_key,
        text_evidence,
        nodes,
        edges,
        root="A",
    )

    print(f"Chosen action index: {best_action}")
    print(f"Combined entropy‑cost score: {score:.4f}")
    sys.exit(0)