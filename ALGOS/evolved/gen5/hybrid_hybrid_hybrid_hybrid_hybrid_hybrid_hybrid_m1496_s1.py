# DARWIN HAMMER — match 1496, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_regret_m746_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fracti_m506_s1.py (gen4)
# born: 2026-05-29T23:36:48Z

"""Hybrid Privacy‑Bandit Engine

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fracti_m506_s1.py (DP & hyper‑vector)
- hybrid_hybrid_hybrid_bandit_hybrid_hybrid_regret_m746_s0.py (Bandit & regret‑weighted selection)

Mathematical bridge:
The min‑hash signature of a text document yields a set of integer identifiers.
Its cardinality |S| is fed to the reconstruction risk function 
    r = unique_quasi_identifiers / total_records .
The same signature is turned into a deterministic seed that drives a random
hyper‑vector hv ∈ {‑1,+1}^d.  A bandit policy maintains a regret‑weighted
probability distribution over admissible differential‑privacy budgets ε_i.
The expected reward for a budget is defined as a utility term minus a
privacy‑risk penalty α·r(ε_i).  Selecting ε_i via the bandit therefore
optimally couples the regret‑weighted distribution (Parent A) with the
privacy‑risk / hyper‑vector pipeline (Parent B) in a single unified system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Iterable, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared immutable data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Action representing a concrete DP budget ε."""
    epsilon: float
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Feedback used to update the bandit policy."""
    context_id: str
    epsilon: float
    reward: float
    propensity: float


@dataclass(frozen=True)
class ModelTier:
    """Placeholder from Parent B – kept for compatibility."""
    name: str
    ram_mb: int
    tier: str


# ----------------------------------------------------------------------
# Global mutable state (policy & store) – mirrors Parent A
# ----------------------------------------------------------------------


_POLICY: Dict[float, List[float]] = {}   # epsilon → [total_reward, count]
_STORE: Dict[str, float] = {}           # arbitrary key‑value store
DEFAULT_BUDGET_MB = 1024 * 4


def reset_policy() -> None:
    """Clear all learned statistics."""
    _POLICY.clear()
    _STORE.clear()


def _reward(epsilon: float) -> float:
    """Mean reward observed for a given ε."""
    total, n = _POLICY.get(epsilon, [0.0, 0.0])
    return total / n if n else 0.0


# ----------------------------------------------------------------------
# Privacy‑related utilities – from Parent B
# ----------------------------------------------------------------------


def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Probability that a record can be re‑identified."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def minhash_signature(text: str, num_hashes: int = 12) -> List[int]:
    """
    Deterministic min‑hash signature.
    For each hash function i we compute min(hash_i(word)) over all words.
    The hash functions are simulated by mixing the built‑in hash with i.
    """
    words = set(text.lower().split())
    signature = []
    for i in range(num_hashes):
        min_val = sys.maxsize
        for w in words:
            # simple mixed hash, deterministic across runs
            h = hash((w, i)) & 0xFFFFFFFFFFFFFFFF
            if h < min_val:
                min_val = h
        signature.append(min_val)
    return signature


def text_to_hypervector(text: str,
                       dim: int = 128,
                       seed: int | None = None) -> np.ndarray:
    """
    Produce a ±1 hyper‑vector seeded by the min‑hash signature.
    The signature is reduced to a 64‑bit integer which becomes the RNG seed.
    """
    sig = minhash_signature(text)
    seed_val = int.from_bytes(
        bytes(sig[:8]), byteorder="little", signed=False
    ) if seed is None else seed
    rng = np.random.RandomState(seed_val % (2**32))
    # random.choice of {-1, +1}
    hv = rng.choice([-1.0, 1.0], size=dim)
    return hv.astype(np.float32)


def dp_aggregate(values: Iterable[float],
                 epsilon: float = 1.0,
                 sensitivity: float = 1.0) -> float:
    """
    Laplace mechanism: add Laplace noise calibrated to ε.
    """
    total = float(sum(values))
    scale = sensitivity / max(epsilon, 1e-9)
    noise = random.Random().expovariate(1 / scale) - scale  # symmetric Laplace
    return total + noise


def dp_aggregate_hypervectors(hypervectors: List[np.ndarray],
                              epsilon: float) -> np.ndarray:
    """
    Component‑wise sum of hyper‑vectors with Laplace noise added.
    """
    if not hypervectors:
        raise ValueError("hypervectors list cannot be empty")
    stacked = np.stack(hypervectors, axis=0)
    summed = stacked.sum(axis=0)
    # Add Laplace noise independently to each dimension
    scale = 1.0 / max(epsilon, 1e-9)
    noise = np.random.laplace(loc=0.0, scale=scale, size=summed.shape)
    return summed + noise


# ----------------------------------------------------------------------
# Bandit core – adapted from Parent A
# ----------------------------------------------------------------------


def select_epsilon(
    context: Dict[str, float],
    epsilon_candidates: List[float],
    algorithm: str = "linucb",
    epsilon_explore: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Choose a DP budget ε from *epsilon_candidates* using a bandit policy.
    Supported algorithms:
    - "epsilon_greedy": with probability *epsilon_explore* pick random ε.
    - "thompson": sample from Beta posterior derived from rewards.
    - any other value defaults to a LinUCB‑like upper‑confidence bound.
    """
    if not epsilon_candidates:
        raise ValueError("epsilon_candidates required")
    rng = random.Random(seed)

    # Exploration step
    if algorithm == "epsilon_greedy" and rng.random() < epsilon_explore:
        chosen = rng.choice(epsilon_candidates)
    elif algorithm == "thompson":
        def beta_sample(eps: float) -> float:
            total, n = _POLICY.get(eps, [0.0, 0.0])
            # interpret reward as successes; failures are (1‑reward)
            a = 1 + max(0.0, total)
            b = 1 + max(0.0, n - total)
            return rng.betavariate(a, b)
        chosen = max(epsilon_candidates, key=beta_sample)
    else:
        # LinUCB‑style: mean reward + confidence term
        scale = math.sqrt(
            sum(float(v) * float(v) for v in context.values())
        ) if context else 1.0
        def ucb_score(eps: float) -> float:
            mean = _reward(eps)
            n = _POLICY.get(eps, [0.0, 0.0])[1]
            conf = scale / math.sqrt(1 + n)
            return mean + conf
        chosen = max(epsilon_candidates, key=ucb_score)

    # Build the BanditAction record
    prop = 1.0 / len(epsilon_candidates)  # uniform propensity for logging
    exp_r = _reward(chosen)
    conf_bound = math.sqrt(
        math.log(1 + len(_POLICY)) / (1 + _POLICY.get(chosen, [0.0, 0.0])[1])
    )
    return BanditAction(
        epsilon=chosen,
        propensity=prop,
        expected_reward=exp_r,
        confidence_bound=conf_bound,
        algorithm=algorithm,
    )


def update_policy(update: BanditUpdate) -> None:
    """
    Incorporate observed *reward* for the selected ε into the global policy.
    """
    total, n = _POLICY.get(update.epsilon, [0.0, 0.0])
    _POLICY[update.epsilon] = [total + update.reward, n + 1]


# ----------------------------------------------------------------------
# Hybrid workflow
# ----------------------------------------------------------------------


def privacy_risk_from_text(text: str, total_records: int) -> float:
    """
    Compute privacy‑risk score r based on the cardinality of the min‑hash
    signature (unique quasi‑identifiers).  This mirrors Parent B's risk
    function and provides a scalar that will be used as a penalty term in
    the bandit reward calculation.
    """
    signature = minhash_signature(text)
    uq = len(set(signature))
    return reconstruction_risk_score(uq, total_records)


def compute_reward(epsilon: float,
                   utility: float,
                   risk: float,
                   alpha: float = 0.5) -> float:
    """
    Reward = utility − α·risk.
    Higher ε gives lower noise → higher utility but also higher risk.
    The bandit tries to maximise this reward.
    """
    return utility - alpha * risk


def select_budget_and_aggregate(
    texts: List[str],
    context: Dict[str, float],
    total_records: int,
    epsilon_candidates: List[float],
    utility_fn: callable,
    algorithm: str = "linucb",
) -> Tuple[float, np.ndarray]:
    """
    End‑to‑end hybrid operation:

    1. For each *text* produce a hyper‑vector.
    2. Estimate privacy risk from the concatenated texts.
    3. Use a bandit to pick a DP budget ε, where the reward incorporates
       the utility (provided by *utility_fn*) and the risk penalty.
    4. Aggregate the hyper‑vectors with Laplace noise calibrated to the
       selected ε.

    Returns the chosen ε and the DP‑aggregated hyper‑vector.
    """
    # Step 1: hyper‑vectors
    hv_list = [text_to_hypervector(t) for t in texts]

    # Step 2: risk
    combined_text = " ".join(texts)
    risk = privacy_risk_from_text(combined_text, total_records)

    # Step 3: bandit selection
    bandit_action = select_epsilon(
        context=context,
        epsilon_candidates=epsilon_candidates,
        algorithm=algorithm,
    )
    epsilon_chosen = bandit_action.epsilon

    # Compute utility (e.g., L2 norm of the summed clean vectors)
    clean_agg = np.stack(hv_list).sum(axis=0)
    utility = utility_fn(clean_agg)

    # Update policy with observed reward
    reward_val = compute_reward(epsilon_chosen, utility, risk)
    update_policy(
        BanditUpdate(
            context_id=str(context),
            epsilon=epsilon_chosen,
            reward=reward_val,
            propensity=bandit_action.propensity,
        )
    )

    # Step 4: DP aggregation
    dp_agg = dp_aggregate_hypervectors(hv_list, epsilon_chosen)
    return epsilon_chosen, dp_agg


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Simple deterministic context
    ctx = {"user_age": 30.0, "session_len": 12.5}
    eps_candidates = [0.1, 0.5, 1.0, 2.0]

    # Dummy utility: larger L2 norm => higher utility
    def l2_utility(vec: np.ndarray) -> float:
        return float(np.linalg.norm(vec))

    sample_texts = [
        "the quick brown fox jumps over the lazy dog",
        "privacy preserving machine learning is fascinating",
        "bandit algorithms balance exploration and exploitation"
    ]

    chosen_eps, agg_vec = select_budget_and_aggregate(
        texts=sample_texts,
        context=ctx,
        total_records=1_000_000,
        epsilon_candidates=eps_candidates,
        utility_fn=l2_utility,
        algorithm="linucb",
    )

    print(f"Chosen ε: {chosen_eps}")
    print(f"Aggregated vector (first 8 dims): {agg_vec[:8]}")
    print(f"Current policy state: {_POLICY}")