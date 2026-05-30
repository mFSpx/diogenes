# DARWIN HAMMER — match 3696, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_semantic_neig_m1841_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1195_s3.py (gen5)
# born: 2026-05-29T23:51:16Z

"""
Hybrid Fusion of:
- hybrid_hybrid_hybrid_hybrid_hybrid_semantic_neig_m1841_s2.py (Bandit‑Fisher + Semantic Neighbors)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1195_s3.py (Bayesian Edge + SSIM + Recovery Priority)

Mathematical Bridge
------------------
Both parents share the *Morphology* data class.  Parent B defines a
`recovery_priority(m)` that is a normalized function of the righting‑time
index.  Parent A treats a “recovery priority” as a factor that re‑weights the
semantic‑neighbor search, and it also consumes the bandit *propensity* scores
as probabilities in a Gaussian‑time model.

In this fusion we:
1. Compute `rp = recovery_priority(m)` (Parent B).
2. Use `rp` to scale the SSIM similarity between a query vector and each
   semantic neighbor (Parent A + Parent B).
3. Treat the scaled similarity as a *pseudo‑likelihood* for the bandit
   propensity, i.e. `π_i ∝ propensity_i * exp(− (1‑sim_i)² / (2σ²))`,
   where σ is derived from the right‑hand side of the Gaussian time model
   (here we set σ = rp + ε to keep it positive).
4. Update the bandit policy with rewards and simultaneously perform a
   Bayesian edge update whose prior is the current edge belief.
   The posterior mean from the Bayesian update is fed back as an
   *expected reward* for the selected action.

Thus the two topologies are fused through a single scalar
`recovery_priority` that mediates similarity weighting, Gaussian
propensity formation, and Bayesian edge belief updating.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, replace, field
from typing import List, Tuple, Dict, Iterable
import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # raw propensity from the bandit router
    expected_reward: float     # updated after Bayesian edge step
    confidence_bound: float    # optional, not used in core fusion
    algorithm: str = "HybridFusion"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass(frozen=True)
class EdgeBetaPrior:
    alpha: float = 1.0
    beta: float = 1.0

@dataclass(frozen=True)
class SemanticNeighbor:
    doc_id: str
    vector: List[float]

# ----------------------------------------------------------------------
# Global stores (in‑memory, lightweight)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}          # action_id -> [total_reward, count]
_EDGE_PRIORS: Dict[str, EdgeBetaPrior] = {}   # edge_id -> prior

# ----------------------------------------------------------------------
# Parent‑B utilities (re‑implemented for self‑containment)
# ----------------------------------------------------------------------
def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology,
                        b: float = 1.0 / 3.0,
                        k: float = 0.35,
                        neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1] derived from righting‑time index."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


def _mean_std(arr: np.ndarray) -> Tuple[float, float]:
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    return mu, sigma


def ssim(x: List[float],
         y: List[float],
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural similarity index for 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("signals must have the same length")
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    mu_x, sigma_x = _mean_std(x_arr)
    mu_y, sigma_y = _mean_std(y_arr)

    cov = float(np.cov(x_arr, y_arr, ddof=1)[0, 1])
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return numerator / denominator if denominator != 0 else 0.0


def bayesian_edge_update(prior: EdgeBetaPrior,
                         successes: int,
                         failures: int) -> Tuple[float, EdgeBetaPrior]:
    """Return posterior mean and updated Beta prior."""
    new_alpha = prior.alpha + successes
    new_beta = prior.beta + failures
    posterior_mean = new_alpha / (new_alpha + new_beta)
    return posterior_mean, EdgeBetaPrior(new_alpha, new_beta)

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_recovery_priority(morph: Morphology) -> float:
    """Wrapper exposing Parent B's recovery priority."""
    return recovery_priority(morph)


def update_hybrid_policy(updates: Iterable[BanditUpdate],
                         edge_id: str) -> None:
    """
    1. Update the bandit policy with raw rewards.
    2. Perform a Bayesian edge update using the same reward as *successes*.
       The number of failures is taken as the complement to a unit reward
       (i.e. failures = max(0, 1‑reward)).
    3. Store the posterior mean as the *expected_reward* for later use.
    """
    # ---- Bandit policy update ------------------------------------------------
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

    # ---- Bayesian edge update -----------------------------------------------
    # For simplicity we aggregate all rewards for the given edge_id.
    total_reward = sum(u.reward for u in updates)
    # Interpret reward ∈ [0,1]; successes = reward, failures = 1‑reward.
    successes = int(round(total_reward))
    failures = max(0, len(list(updates)) - successes)

    prior = _EDGE_PRIORS.get(edge_id, EdgeBetaPrior())
    posterior_mean, new_prior = bayesian_edge_update(prior, successes, failures)
    _EDGE_PRIORS[edge_id] = new_prior

    # Store the posterior mean in a dedicated cache for quick lookup.
    _POLICY.setdefault(f"edge::{edge_id}", [0.0, 0.0])[0] = posterior_mean
    _POLICY.setdefault(f"edge::{edge_id}", [0.0, 0.0])[1] = 1.0  # count = 1


def _gaussian_propensity(propensity: float, similarity: float, sigma: float) -> float:
    """
    Convert a raw propensity into a Gaussian‑weighted likelihood using the
    (1‑similarity) distance.  The variance σ² is derived from recovery priority.
    """
    distance = 1.0 - similarity
    weight = math.exp(- (distance ** 2) / (2.0 * (sigma ** 2)))
    return propensity * weight


def select_hybrid_action(context_id: str,
                         candidate_actions: List[BanditAction],
                         morphology: Morphology,
                         query_vector: List[float],
                         neighbors: List[SemanticNeighbor]) -> BanditAction:
    """
    Hybrid action selection:
    1. Compute recovery priority (rp) from morphology.
    2. For each neighbor compute SSIM similarity with the query vector,
       then scale by rp.
    3. Combine each action's raw propensity with the *max* scaled similarity
       of its associated neighbor (if any) via a Gaussian weighting.
    4. Choose the action with the highest combined score.
    5. Attach the posterior mean of the related edge (if present) as the
       expected reward.
    """
    rp = compute_recovery_priority(morphology)
    sigma = rp + 1e-6          # avoid zero variance

    # Build a map from doc_id to best similarity (scaled by rp)
    best_sim: Dict[str, float] = {}
    for nb in neighbors:
        sim = ssim(query_vector, nb.vector)
        scaled = rp * sim
        if nb.doc_id not in best_sim or scaled > best_sim[nb.doc_id]:
            best_sim[nb.doc_id] = scaled

    # Assume each action is linked to a neighbor via its action_id == doc_id.
    # If no neighbor exists, similarity defaults to 0.
    best_score = -math.inf
    chosen = None
    for act in candidate_actions:
        sim = best_sim.get(act.action_id, 0.0)
        combined = _gaussian_propensity(act.propensity, sim, sigma)

        # Incorporate policy‑based expected reward (average reward so far)
        total, cnt = _POLICY.get(act.action_id, [0.0, 0.0])
        avg_reward = total / cnt if cnt else 0.0

        # Bayesian edge posterior (if any) further boosts the score
        edge_key = f"edge::{act.action_id}"
        edge_mean = _POLICY.get(edge_key, [0.0, 0.0])[0]

        final_score = combined + avg_reward + edge_mean

        if final_score > best_score:
            best_score = final_score
            chosen = replace(act,
                             expected_reward=avg_reward + edge_mean,
                             confidence_bound=combined)  # repurpose field

    if chosen is None:
        raise RuntimeError("No candidate actions provided")
    return chosen


def weighted_similarity_matrix(query: List[float],
                               neighbors: List[SemanticNeighbor],
                               morphology: Morphology) -> np.ndarray:
    """
    Return an (N × 1) matrix where each entry is
        rp * ssim(query, neighbor_i)
    This matrix can be used downstream for linear‑algebraic formulations.
    """
    rp = compute_recovery_priority(morphology)
    sims = np.array([rp * ssim(query, nb.vector) for nb in neighbors],
                    dtype=float).reshape(-1, 1)
    return sims

# ----------------------------------------------------------------------
# Utility / reset helpers
# ----------------------------------------------------------------------
def reset_hybrid_state() -> None:
    _POLICY.clear()
    _EDGE_PRIORS.clear()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Sample morphology
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)

    # Create dummy semantic neighbors (3‑dim vectors)
    neighbors = [
        SemanticNeighbor(doc_id="A", vector=[0.1, 0.2, 0.3]),
        SemanticNeighbor(doc_id="B", vector=[0.4, 0.5, 0.6]),
        SemanticNeighbor(doc_id="C", vector=[0.7, 0.8, 0.9]),
    ]

    # Query vector
    query = [0.15, 0.25, 0.35]

    # Candidate bandit actions (raw propensities drawn randomly)
    actions = [
        BanditAction(action_id="A", propensity=0.6, expected_reward=0.0,
                     confidence_bound=0.0),
        BanditAction(action_id="B", propensity=0.3, expected_reward=0.0,
                     confidence_bound=0.0),
        BanditAction(action_id="C", propensity=0.1, expected_reward=0.0,
                     confidence_bound=0.0),
    ]

    # Simulate a batch of bandit updates
    updates = [
        BanditUpdate(context_id="ctx1", action_id="A", reward=1.0, propensity=0.6),
        BanditUpdate(context_id="ctx1", action_id="B", reward=0.0, propensity=0.3),
        BanditUpdate(context_id="ctx1", action_id="C", reward=0.0, propensity=0.1),
    ]

    # Apply hybrid update
    update_hybrid_policy(updates, edge_id="A")

    # Select an action
    chosen = select_hybrid_action(
        context_id="ctx1",
        candidate_actions=actions,
        morphology=morph,
        query_vector=query,
        neighbors=neighbors,
    )
    print("Chosen action:", chosen)

    # Compute similarity matrix (demonstrates third required function)
    sim_mat = weighted_similarity_matrix(query, neighbors, morph)
    print("Weighted similarity matrix:\n", sim_mat)