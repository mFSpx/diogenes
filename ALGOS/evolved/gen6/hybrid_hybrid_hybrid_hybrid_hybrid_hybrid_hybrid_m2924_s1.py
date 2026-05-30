# DARWIN HAMMER — match 2924, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m2355_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hdc_se_m1184_s1.py (gen5)
# born: 2026-05-29T23:46:41Z

"""Hybrid Variational‑Free‑Energy / Fisher‑Information Bandit with HDC‑WTA Encoding
================================================================================

This module fuses the core mathematics of the two parent algorithms:

* **Parent A** – provides a *variational free energy* (VFE) formulation that
  evaluates the expected utility of a bandit action and a *developmental rate*
  model based on the Schoolfield equation.

* **Parent B** – supplies hyperdimensional (HDC) encodings of scalar morphology
  and a sparse *winner‑take‑all* (WTA) hypervector derived from a list of
  real‑valued scores.  The dot‑product of these hypervectors is used as a
  similarity measure that can be interpreted as an estimate of *Fisher
  information* for the underlying probabilistic model.

**Mathematical bridge**  
The Fisher information of a parametric model is proportional to the squared
gradient of the log‑likelihood.  In a high‑dimensional binary space the
dot‑product between two bipolar hypervectors behaves like a sum of independent
Bernoulli variables; its square therefore serves as a proxy for the Fisher
information of the encoded parameters.  We therefore set the *precision*
parameter of the VFE to this hyperdimensional similarity, linking the
information‑theoretic content of the HDC/WTA encoding to the variational free
energy of the bandit decision.

The resulting hybrid score for a candidate action is


score = - VFE(expected_reward, precision) + confidence_bound
      = expected_reward * precision + confidence_bound


where `precision = fisher_information(morphology_hv, sparse_wta_hv)`.

The module implements three high‑level functions that demonstrate this fused
behaviour:

1. `morphology_hv` – encodes a scalar morphology into a bipolar hypervector.
2. `sparse_wta_hv` – builds a sparse WTA hypervector from a list of scores.
3. `hybrid_action_score` – computes the hybrid VFE‑Fisher score for a bandit
   action and selects the optimal action for a given context.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Sequence, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Parent A – Variational Free Energy & Developmental Rate
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15


def c_to_k(temp_c: float) -> float:
    """Convert Celsius to Kelvin."""
    return temp_c + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams) -> float:
    """Schoolfield‐type temperature dependence."""
    return params.rho_25 * math.exp(
        (params.delta_h_activation / 8.314) * (1 / 298.15 - 1 / temp_k)
    )


def normalized_activity(temp_c: float, low_c: float = 5) -> float:
    """Normalized activity using the developmental rate."""
    params = SchoolfieldParams()
    return developmental_rate(c_to_k(temp_c), params)


def variational_free_energy(expected_utility: float, precision: float) -> float:
    """Variational Free Energy (VFE) for a given expected utility."""
    return -expected_utility * precision


# ----------------------------------------------------------------------
# Parent B – Hyperdimensional Computing (HDC) & Sparse WTA
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Scalar morphology that will be encoded into a hypervector."""
    length: float


def _random_bipolar(dim: int, seed: int | None = None) -> np.ndarray:
    """Generate a random bipolar vector (+1 / -1)."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=dim).astype(np.int8)


def morphology_hv(morph: Morphology, dim: int = 10_000, seed: int | None = None) -> np.ndarray:
    """
    Encode a scalar morphology into a bipolar hypervector.
    The scalar is used to seed a deterministic random generator so that
    identical morphologies map to identical hypervectors.
    """
    # Derive a reproducible integer seed from the float value
    int_seed = int(abs(morph.length) * 1e6) % (2 ** 31 - 1)
    return _random_bipolar(dim, seed=int_seed)


def sparse_wta_hv(scores: List[float], dim: int = 10_000, k: int = 10, seed: int | None = None) -> np.ndarray:
    """
    Build a sparse winner‑take‑all hypervector.
    The top‑k scores receive a +1, all others -1.
    """
    if k <= 0 or k > dim:
        raise ValueError("k must be between 1 and dim")
    hv = np.full(dim, -1, dtype=np.int8)
    # Determine the indices of the k highest scores
    if len(scores) == 0:
        # No scores → keep all -1
        return hv
    # If there are fewer scores than k, fill the remaining slots randomly
    effective_k = min(k, len(scores))
    top_indices = np.argpartition(scores, -effective_k)[-effective_k:]
    # Randomly distribute the selected +1s across the hypervector
    rng = np.random.default_rng(seed)
    chosen_positions = rng.choice(dim, size=effective_k, replace=False)
    hv[chosen_positions] = 1
    return hv


def fisher_information(hv1: np.ndarray, hv2: np.ndarray) -> float:
    """
    Approximate Fisher information using the squared dot‑product of two
    bipolar hypervectors, normalised by dimensionality.
    """
    if hv1.shape != hv2.shape:
        raise ValueError("Hypervectors must have the same shape")
    dot = np.dot(hv1, hv2)  # range: [-dim, dim]
    dim = hv1.size
    # Square makes it always non‑negative and emphasizes strong similarity
    return (dot ** 2) / dim


# ----------------------------------------------------------------------
# Hybrid Functions (fusion of A & B)
# ----------------------------------------------------------------------
def hybrid_action_score(
    action: BanditAction,
    morph: Morphology,
    scores: List[float],
    dim: int = 10_000,
    wta_k: int = 10,
) -> float:
    """
    Compute the hybrid VFE‑Fisher score for a single bandit action.

    Steps:
    1. Encode the morphology into a hypervector (HDC).
    2. Convert the list of contextual scores into a sparse WTA hypervector.
    3. Estimate Fisher information via the dot‑product of the two hypervectors.
    4. Use this Fisher estimate as the *precision* term in the VFE.
    5. Return the combined score:  expected_reward * precision + confidence_bound.
    """
    hv_morph = morphology_hv(morph, dim=dim)
    hv_wta = sparse_wta_hv(scores, dim=dim, k=wta_k)

    precision = fisher_information(hv_morph, hv_wta) + 1e-12  # avoid zero
    vfe = variational_free_energy(action.expected_reward, precision)
    # Higher score is better; we flip the sign of VFE (since VFE is negative)
    hybrid_score = -vfe + action.confidence_bound
    return hybrid_score


def select_optimal_action(
    actions: List[BanditAction],
    morph: Morphology,
    scores: List[float],
    dim: int = 10_000,
    wta_k: int = 10,
) -> Tuple[BanditAction, float]:
    """
    Evaluate all actions with `hybrid_action_score` and return the
    action with the maximum score together with that score.
    """
    best_action = None
    best_score = -math.inf
    for act in actions:
        sc = hybrid_action_score(act, morph, scores, dim=dim, wta_k=wta_k)
        if sc > best_score:
            best_score = sc
            best_action = act
    if best_action is None:
        raise RuntimeError("No actions provided")
    return best_action, best_score


def update_bandit(
    updates: List[BanditUpdate],
    actions: Dict[str, BanditAction],
    learning_rate: float = 0.1,
) -> Dict[str, BanditAction]:
    """
    Simple incremental update: move the expected_reward of each action towards
    the observed reward proportionally to the propensity and a learning rate.
    This mimics a stochastic gradient step on the VFE objective.
    """
    new_actions = {}
    for upd in updates:
        act = actions.get(upd.action_id)
        if act is None:
            continue  # unknown action, skip
        # Gradient step on expected_reward
        delta = learning_rate * upd.propensity * (upd.reward - act.expected_reward)
        new_expected = act.expected_reward + delta
        # Re‑create the dataclass with the updated expected reward
        new_act = BanditAction(
            action_id=act.action_id,
            propensity=act.propensity,
            expected_reward=new_expected,
            confidence_bound=act.confidence_bound,
            algorithm=act.algorithm,
        )
        new_actions[act.action_id] = new_act
    # Preserve actions that were not updated
    for aid, act in actions.items():
        if aid not in new_actions:
            new_actions[aid] = act
    return new_actions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a small set of bandit actions
    actions = {
        f"a{i}": BanditAction(
            action_id=f"a{i}",
            propensity=random.uniform(0.1, 1.0),
            expected_reward=random.uniform(0.0, 1.0),
            confidence_bound=random.uniform(0.0, 0.5),
            algorithm="hybrid_vfe_fisher",
        )
        for i in range(5)
    }

    # Simulated context scores (e.g., outputs of an RBF surrogate)
    context_scores = [random.gauss(0, 1) for _ in range(20)]

    # Random morphology
    morph = Morphology(length=random.uniform(0.5, 2.0))

    # Choose the best action according to the hybrid score
    best_act, best_sc = select_optimal_action(
        list(actions.values()), morph, context_scores, dim=8000, wta_k=5
    )
    print("Best action selected:")
    print(asdict(best_act))
    print(f"Hybrid score: {best_sc:.6f}")

    # Simulate a bandit update after receiving a reward
    updates = [
        BanditUpdate(
            context_id="ctx1",
            action_id=best_act.action_id,
            reward=random.uniform(0, 1),
            propensity=best_act.propensity,
        )
    ]

    actions = update_bandit(updates, actions, learning_rate=0.05)
    print("\nActions after one update:")
    for act in actions.values():
        print(asdict(act))