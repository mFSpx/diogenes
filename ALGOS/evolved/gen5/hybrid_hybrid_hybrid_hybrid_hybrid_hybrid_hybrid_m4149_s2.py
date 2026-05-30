# DARWIN HAMMER — match 4149, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_pheromone_inf_m495_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1086_s2.py (gen4)
# born: 2026-05-29T23:53:46Z

"""HybridBanditRegretPheromone
================================

Parent algorithms:
- **hybrid_hybrid_hybrid_bandit_hybrid_pheromone_inf_m495_s0.py** – Bandit‑Router
  with pheromone‑based infotaxis.  Pheromone concentrations are turned into an
  entropy‑based bonus that augments each bandit action’s expected reward and
  influences the action propensity through variational free‑energy minimisation.

- **hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1086_s2.py** – Regret‑Weighted
  MinHash selector built on top of a tropical (max‑plus) network fed by health
  scores from a linear state‑space model (SSM).

**Mathematical bridge**
-----------------------

1.  For each action *i* a pheromone concentration ϕᵢ is observed.  
    An entropy‑based bonus  

        Bᵢ = - Σₖ pₖ log pₖ ,   pₖ = ϕᵢ / Σⱼ ϕⱼ

    is added to the bandit expected reward *r̂ᵢ*.

2.  The health‑score vector **y** from the SSM is transformed by a tropical
    (max‑plus) linear layer  

        τᵢ = maxⱼ ( Wᵢⱼ + yⱼ ) ,

    where **W** is a fixed weight matrix.  τᵢ plays the role of the “impurity‑gain”
    candidate in the regret engine.

3.  A regret term  

        Rᵢ = (r̂ᵢ + Bᵢ) – costᵢ – riskᵢ + cfᵢ

    is passed through a sigmoid weighting *g(Rᵢ) = 1/(1+e^{-Rᵢ})*.

4.  MinHash signatures of the action’s token set are compared with a reference
    signature *σ_ref* to obtain a Jaccard‑like similarity *sᵢ*.

5.  The final hybrid score for action *i* is  

        Sᵢ = g(Rᵢ) · (1 + sᵢ) · τᵢ .

    Propensities are updated by a softmax over *S* and a variational free‑energy
    term *F = -log(propensity)*, yielding a new policy that simultaneously
    respects pheromone information, tropical health scores, and regret‑weighted
    similarity.

The functions below implement these equations and expose a minimal API for a
hybrid bandit‑regret‑pheromone agent.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Iterable, Tuple

import numpy as np
import hashlib

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class HybridAction:
    """Unified representation of an action in the hybrid system."""
    action_id: str
    tokens: List[str]                     # tokens for MinHash signature
    expected_reward: float = 0.0
    cost: float = 0.0
    risk: float = 0.0
    counterfactual: float = 0.0           # cfᵢ term
    pheromone: float = 0.0                # raw pheromone concentration ϕᵢ
    propensity: float = 0.0               # policy propensity πᵢ
    health_score: float = 0.0             # τᵢ after tropical transform
    signature: List[int] = field(default_factory=list)  # MinHash signature


# ----------------------------------------------------------------------
# MinHash utilities (from Parent B)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 128, seed: int = 0) -> List[int]:
    """Return a MinHash signature of length *k* for the given token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2**64 - 1] * k
    sig = []
    for i in range(k):
        mins = min(_hash(seed + i, t) for t in toks)
        sig.append(mins)
    return sig


def jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Approximate Jaccard similarity using MinHash signatures."""
    if not sig1 or not sig2 or len(sig1) != len(sig2):
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


# ----------------------------------------------------------------------
# Pheromone entropy bonus (from Parent A)
# ----------------------------------------------------------------------
def pheromone_entropy_bonus(actions: List[HybridAction]) -> Dict[str, float]:
    """Compute entropy‑based bonus Bᵢ for each action from pheromone levels."""
    total = sum(a.pheromone for a in actions)
    if total == 0.0:
        # No pheromone information → zero bonus
        return {a.action_id: 0.0 for a in actions}
    probs = {a.action_id: a.pheromone / total for a in actions}
    entropy = -sum(p * math.log(p + 1e-12) for p in probs.values())
    # Distribute the same entropy value as a bonus to all actions
    return {aid: entropy for aid in probs}


# ----------------------------------------------------------------------
# Tropical (max‑plus) linear layer (from Parent B)
# ----------------------------------------------------------------------
def tropical_transform(health_vec: np.ndarray, weight_mat: np.ndarray) -> np.ndarray:
    """
    Apply a max‑plus linear transformation:
        τᵢ = maxⱼ ( Wᵢⱼ + yⱼ )
    where *y* is the health‑score vector.
    """
    if weight_mat.shape[1] != health_vec.shape[0]:
        raise ValueError("Weight matrix and health vector dimensions mismatch")
    # broadcasting addition then max over columns (j)
    return np.max(weight_mat + health_vec, axis=1)


# ----------------------------------------------------------------------
# Core hybrid score computation
# ----------------------------------------------------------------------
def compute_hybrid_scores(
    actions: List[HybridAction],
    ref_signature: List[int],
    weight_mat: np.ndarray,
    health_vec: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute the hybrid score vector S and the updated propensity distribution.
    Returns (propensity_softmax, scores_dict).
    """
    # 1. Pheromone entropy bonus
    bonus_map = pheromone_entropy_bonus(actions)

    # 2. Tropical health scores τᵢ
    tropical_scores = tropical_transform(health_vec, weight_mat)
    for a, τ in zip(actions, tropical_scores):
        a.health_score = float(τ)

    # 3. Regret term Rᵢ and sigmoid weighting g(Rᵢ)
    scores = {}
    for a in actions:
        R = (a.expected_reward + bonus_map[a.action_id]) - a.cost - a.risk + a.counterfactual
        g = 1.0 / (1.0 + math.exp(-R))  # sigmoid
        # 4. MinHash similarity term
        if not a.signature:
            a.signature = minhash_signature(a.tokens)
        sim = jaccard_similarity(a.signature, ref_signature)
        # 5. Hybrid score Sᵢ = g * (1 + sim) * τᵢ
        S = g * (1.0 + sim) * a.health_score
        scores[a.action_id] = S

    # 6. Softmax over scores to obtain propensities
    score_vec = np.array(list(scores.values()))
    max_s = np.max(score_vec)  # for numerical stability
    exp_vals = np.exp(score_vec - max_s)
    propensities = exp_vals / exp_vals.sum()

    # 7. Attach propensities back to actions
    for a, π in zip(actions, propensities):
        a.propensity = float(π)

    return propensities, scores


# ----------------------------------------------------------------------
# Policy update (variational free‑energy style)
# ----------------------------------------------------------------------
def update_policy(actions: List[HybridAction]) -> None:
    """
    Perform a simple variational free‑energy update:
        Fᵢ = -log πᵢ
    The policy dictionary stores cumulative free‑energy per action.
    """
    for a in actions:
        F = -math.log(a.propensity + 1e-12)
        # Store in a global dict for inspection (mimics parent A's _POLICY)
        _POLICY.setdefault(a.action_id, []).append(F)


_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()


def average_free_energy(action_id: str) -> float:
    """Return the average free‑energy accumulated for the given action."""
    vals = _POLICY.get(action_id, [])
    return sum(vals) / len(vals) if vals else float('inf')


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def initialise_actions() -> List[HybridAction]:
    """Create a small set of example actions with random attributes."""
    actions = []
    for i in range(5):
        aid = f"a{i}"
        tokens = [f"token{j}" for j in random.sample(range(10), 4)]
        act = HybridAction(
            action_id=aid,
            tokens=tokens,
            expected_reward=random.uniform(0, 1),
            cost=random.uniform(0, 0.3),
            risk=random.uniform(0, 0.2),
            counterfactual=random.uniform(-0.1, 0.1),
            pheromone=random.uniform(0, 5),
        )
        actions.append(act)
    return actions


def run_hybrid_step(actions: List[HybridAction]) -> None:
    """Execute one hybrid decision step."""
    # Reference signature (could be a goal description)
    ref_sig = minhash_signature(["goal", "target", "optimal"])

    # Random tropical weight matrix and health vector
    dim = len(actions)
    weight_mat = np.random.randn(dim, dim)  # Wᵢⱼ
    health_vec = np.random.rand(dim) * 2.0  # yⱼ in [0,2]

    prop, scores = compute_hybrid_scores(
        actions, ref_sig, weight_mat, health_vec
    )
    update_policy(actions)

    # Simple diagnostics
    print("Propensities:", prop.round(3))
    print("Hybrid scores:", {k: round(v, 3) for k, v in scores.items()})
    for a in actions:
        print(
            f"{a.action_id}: π={a.propensity:.3f}, F_avg={average_free_energy(a.action_id):.3f}"
        )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    reset_policy()
    acts = initialise_actions()
    for step in range(3):
        print(f"\n--- Hybrid step {step + 1} ---")
        run_hybrid_step(acts)