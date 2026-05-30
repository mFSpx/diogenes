# DARWIN HAMMER — match 1682, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_bayes_claim_k_m261_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s2.py (gen4)
# born: 2026-05-29T23:38:11Z

"""Hybrid Fusion of Bayesian Fisher‑Krampus & Decision‑Bandit Topologies.

Parents:
- hybrid_hybrid_hybrid_fisher_hybrid_bayes_claim_k_m261_s0.py (Algorithm A)
- hybrid_hybrid_hybrid_decisi_hybrid_hybrid_bandit_m14_s2.py (Algorithm B)

Mathematical Bridge:
Algorithm A produces a posterior probability `p_i` for each candidate via a Bayesian
update whose likelihood ratio is scaled by a Fisher‑Krampus information density.
Algorithm B builds a resource matrix `A` whose rows correspond to entities or models
and whose columns are spatial, privacy and decision scores.  

The fusion treats the posterior `p_i` as a *resource weight* that modulates the
bandit‑style selection in Algorithm B.  Concretely, for each row `r` of `A` we
compute a weighted utility  

    u_r = p_r * (w_spatial·r_spatial + w_privacy·r_privacy + w_score·r_score)

where the weights `w_*` are hyper‑parameters.  The greedy selector then chooses a
subset `x` that respects the linear budget constraints while maximizing the sum
of `u_r`.  This unifies the Bayesian information‑theoretic update with the
resource‑constrained bandit decision process.

The module implements:
1. Core Gaussian / Fisher‑Krampus utilities (A).
2. Resource‑vector construction for entities (B).
3. `bayesian_update_with_resources` – updates posteriors and recomputes weighted
   utilities.
4. `greedy_select` – a budget‑aware selector that uses the fused utilities.
5. A small smoke‑test exercising the full pipeline.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

# ----------------------------------------------------------------------
# Algorithm A – Fisher‑Krampus & Bayesian helpers
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Standard Gaussian beam intensity."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam (derivative² / intensity)."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Monotonically decreasing pruning probability."""
    if t < 0:
        raise ValueError("time t must be non‑negative")
    return math.exp(-lam * t) * (1.0 - alpha)


def bayesian_update(hypothesis: dict, evidence: dict, likelihood_ratio: float) -> dict:
    """Standard Bayesian odds update with clipping."""
    if likelihood_ratio < 0:
        raise ValueError("likelihood_ratio must be non‑negative")
    p = max(0.0, min(1.0, hypothesis['posterior']))
    if p <= 0.0 or likelihood_ratio == 0.0:
        posterior = 0.0
    elif p >= 1.0:
        posterior = 1.0
    else:
        odds = p / (1.0 - p)
        new_odds = odds * likelihood_ratio
        posterior = new_odds / (1.0 + new_odds)
    posterior = max(0.0, min(1.0, posterior))
    return {
        'id': hypothesis['id'],
        'prior': hypothesis['posterior'],
        'posterior': posterior,
        'evidence_ids': hypothesis['evidence_ids'] + [evidence['id']],
    }

# ----------------------------------------------------------------------
# Algorithm B – Resource vector & bandit‑style selection
# ----------------------------------------------------------------------
def haversine_distance(loc1: tuple[float, float], loc2: tuple[float, float]) -> float:
    """Distance in metres between two (lat, lon) points."""
    R = 6371000.0  # Earth radius in metres
    lat1, lon1 = map(math.radians, loc1)
    lat2, lon2 = map(math.radians, loc2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def calculate_resource_vector(entity: dict,
                              reference_location: tuple[float, float],
                              beta: float = 1.0,
                              alpha: float = 1.0) -> np.ndarray:
    """
    Compute e_i = [d_i, p_i, s_i] for an entity.

    - d_i : haversine distance to reference_location.
    - p_i : beta * sigma_i, where sigma_i = 1 if the signature collides with any
            other entity in the global list (collision detection is deferred to
            the caller; here we accept a pre‑computed bool flag).
    - s_i : decision‑hygiene score (provided in the entity dict).
    """
    d_i = haversine_distance(entity['location'], reference_location)
    sigma_i = 1.0 if entity.get('collision', False) else 0.0
    p_i = beta * sigma_i
    s_i = alpha * entity.get('decision_score', 0.0)
    return np.array([d_i, p_i, s_i], dtype=float)


def assemble_resource_matrix(entities: list[dict],
                             reference_location: tuple[float, float],
                             beta: float = 1.0,
                             alpha: float = 1.0) -> np.ndarray:
    """Stack resource vectors of all entities into matrix A (rows = entities)."""
    rows = [calculate_resource_vector(e, reference_location, beta, alpha) for e in entities]
    return np.vstack(rows) if rows else np.empty((0, 3))


# ----------------------------------------------------------------------
# Fusion Layer – Bayesian posterior as resource weight
# ----------------------------------------------------------------------
def bayesian_update_with_resources(hypotheses: list[dict],
                                   evidences: list[dict],
                                   theta: float,
                                   center: float,
                                   width: float,
                                   t: float) -> list[dict]:
    """
    Perform a Bayesian update for each hypothesis using a likelihood ratio that
    incorporates Fisher information and a time‑dependent pruning factor.

    Returns the updated list of hypotheses.
    """
    pr = prune_probability(t)
    updated = []
    for hyp, ev in zip(hypotheses, evidences):
        # Fisher information acts as an "information density" factor
        info = fisher_score(theta, center, width)
        # Likelihood ratio = info * pr (both >=0)
        lr = info * pr
        updated.append(bayesian_update(hyp, ev, lr))
    return updated


def compute_weighted_utilities(resource_matrix: np.ndarray,
                               posteriors: np.ndarray,
                               w_spatial: float = 1.0,
                               w_privacy: float = 1.0,
                               w_score: float = 1.0) -> np.ndarray:
    """
    Fuse posteriors with the resource matrix.

    For each row r = [d, p, s]:
        utility = posterior * (w_spatial*d + w_privacy*p + w_score*s)

    Returns a 1‑D array of utilities.
    """
    if resource_matrix.shape[0] != posteriors.shape[0]:
        raise ValueError("Row count of resource matrix must match number of posteriors")
    linear_combo = (w_spatial * resource_matrix[:, 0] +
                    w_privacy * resource_matrix[:, 1] +
                    w_score * resource_matrix[:, 2])
    return posteriors * linear_combo


def greedy_select(resource_matrix: np.ndarray,
                  utilities: np.ndarray,
                  budgets: tuple[float, float, float]) -> np.ndarray:
    """
    Greedy budget‑aware selector.

    Parameters
    ----------
    resource_matrix : (n,3) array  – rows = [d, p, s].
    utilities       : (n,) array   – fused utility for each row.
    budgets         : (spatial, privacy, decision) max allowed totals.

    Returns
    -------
    x : (n,) binary ndarray where 1 indicates selection.
    """
    n = resource_matrix.shape[0]
    selected = np.zeros(n, dtype=int)

    # Sort indices by descending utility density (utility per unit of budget consumption)
    # To avoid division by zero we add a tiny epsilon to the denominator.
    eps = 1e-12
    budget_consumption = (resource_matrix[:, 0] / (budgets[0] + eps) +
                         resource_matrix[:, 1] / (budgets[1] + eps) +
                         resource_matrix[:, 2] / (budgets[2] + eps))
    density = utilities / (budget_consumption + eps)
    order = np.argsort(-density)  # descending

    remaining = np.array(budgets, dtype=float)

    for idx in order:
        cost = resource_matrix[idx]
        if np.all(cost <= remaining + eps):
            selected[idx] = 1
            remaining -= cost
    return selected

# ----------------------------------------------------------------------
# End‑to‑end demo
# ----------------------------------------------------------------------
def _demo():
    # ---- Synthetic entities ------------------------------------------------
    reference_loc = (37.7749, -122.4194)  # San Francisco
    entities = [
        {
            'id': f'e{i}',
            'location': (37.7749 + random.uniform(-0.1, 0.1),
                         -122.4194 + random.uniform(-0.1, 0.1)),
            'signature': f'sig{i}',
            'decision_score': random.uniform(0, 10),
        }
        for i in range(8)
    ]

    # Simple collision flag: mark any duplicate signature as colliding
    signatures = {}
    for e in entities:
        sig = e['signature']
        if sig in signatures:
            e['collision'] = True
            signatures[sig]['collision'] = True
        else:
            e['collision'] = False
            signatures[sig] = e

    # ---- Bayesian hypotheses (one per entity) -----------------------------
    hypotheses = [{
        'id': e['id'],
        'posterior': random.uniform(0.2, 0.8),
        'evidence_ids': []
    } for e in entities]

    evidences = [{'id': f'ev{i}'} for i in range(len(entities))]

    # ---- Perform Bayesian update -----------------------------------------
    theta = 0.5
    center = 0.0
    width = 1.0
    t = 0.3
    updated_hyps = bayesian_update_with_resources(
        hypotheses, evidences, theta, center, width, t)

    posteriors = np.array([h['posterior'] for h in updated_hyps], dtype=float)

    # ---- Build resource matrix --------------------------------------------
    A = assemble_resource_matrix(entities, reference_loc, beta=2.0, alpha=0.5)

    # ---- Fuse and select --------------------------------------------------
    utilities = compute_weighted_utilities(A, posteriors,
                                            w_spatial=0.001,   # distance is large, scale down
                                            w_privacy=1.0,
                                            w_score=1.5)

    budgets = (5000.0, 5.0, 30.0)   # metres, privacy load, decision score budget
    selection = greedy_select(A, utilities, budgets)

    # ---- Report ------------------------------------------------------------
    print("=== Fusion Demo ===")
    for i, e in enumerate(entities):
        sel = 'SELECTED' if selection[i] else 'skip'
        print(f"{e['id']}: posterior={posteriors[i]:.3f}, utility={utilities[i]:.2f}, {sel}")

if __name__ == "__main__":
    _demo()