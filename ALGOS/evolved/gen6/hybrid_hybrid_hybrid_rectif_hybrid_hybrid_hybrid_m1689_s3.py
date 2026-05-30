# DARWIN HAMMER — match 1689, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_rectified_flo_hybrid_hybrid_endpoi_m519_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decrea_m527_s6.py (gen5)
# born: 2026-05-29T23:38:15Z

"""Hybrid Algorithm combining Stylometric LSM vectors with Regret‑Epistemic Bayesian pruning.

Parents:
- hybrid_hybrid_rectified_flo_hybrid_hybrid_endpoi_m519_s0.py (stylometry, LSM vectors, morphology health factor)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decrea_m527_s6.py (regret‑weighted cost, Bayesian update, Gini‑adjusted pruning, Fisher‑score localization)

Mathematical Bridge:
The LSM vector provides a *certainty* scalar `c ∈ [0,1]` derived from the proportion of
high‑certainty word categories (e.g., adverbs, auxiliary verbs).  The regret vector `r`
produced by the decision‑engine acts as a prior risk `p`.  By treating `c` as a likelihood
and `p` as a prior we compute a Bayesian posterior weight  

    w_i = (c * p_i) / (c * p_i + fp * (1‑p_i))

where `fp` is a tiny false‑positive rate.  The posterior weight vector `w` is then fed
to the decreasing‑rate pruning schedule; surviving actions are localized by a
Fisher‑score weighted centroid (the “optimal angle”).  Finally a Gini‑coefficient
adjustment rescales the expected value of the remaining actions.

The functions below implement this unified pipeline."""
import math
import random
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Stylometry utilities (from Parent A)
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> List[str]:
    """Tokenise a string into alphabetic lower‑case words."""
    return [word for word in (text or "").lower().split() if word.isalpha()]


def lsm_vector(text: str) -> Dict[str, float]:
    """
    Compute a simplified LSM (Latent Semantic Morphology) vector.
    For each functional category we calculate the relative frequency of its words.
    """
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    vec: Dict[str, float] = {}
    for cat, vocab in FUNCTION_CATS.items():
        cat_count = sum(cnt[w] for w in vocab)
        vec[cat] = cat_count / total
    return vec


# ----------------------------------------------------------------------
# Regret‑Epistemic utilities (from Parent B)
# ----------------------------------------------------------------------


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a 1‑D array."""
    if values.ndim != 1:
        raise ValueError("Gini coefficient requires a 1‑D array")
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals, dtype=float)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


def bayes_marginal(prior: float, likelihood: float, false_positive: float = 1e-6) -> float:
    """Marginal probability for a simple Bayesian update."""
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, false_positive: float = 1e-6) -> float:
    """Posterior weight using Bayes rule with a tiny false‑positive term."""
    marginal = bayes_marginal(prior, likelihood, false_positive)
    if marginal == 0:
        return 0.0
    return (likelihood * prior) / marginal


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def compute_certainty_factor(lsm_vec: Dict[str, float]) -> float:
    """
    Derive a single certainty scalar from the LSM vector.
    We treat categories that convey confidence (auxiliary, adverb_common, quantifier)
    as contributing positively; all others are ignored.
    The result is clipped to [0,1].
    """
    confidence_cats = ["auxiliary", "adverb_common", "quantifier"]
    c = sum(lsm_vec.get(cat, 0.0) for cat in confidence_cats)
    return max(0.0, min(1.0, c))


def generate_regret_vector(num_actions: int, seed: int | None = None) -> np.ndarray:
    """
    Simulate a regret (risk) vector for a set of actions.
    Larger values indicate higher expected regret.
    """
    rng = np.random.default_rng(seed)
    # Use a log‑normal distribution to obtain strictly positive regrets
    regrets = rng.lognormal(mean=0.0, sigma=1.0, size=num_actions)
    # Normalise to a probability‑like prior in [0,1]
    regrets = regrets / regrets.max()
    return regrets.astype(float)


def posterior_weights_from_regret(regret_vec: np.ndarray, certainty: float) -> np.ndarray:
    """
    Apply the Bayesian bridge: treat each regret entry as a prior risk `p`,
    and the certainty scalar as a likelihood `c`.  Return the posterior weight
    vector `w`.
    """
    if regret_vec.ndim != 1:
        raise ValueError("regret_vec must be 1‑D")
    posterior = np.vectorize(bayes_update)(regret_vec, certainty)
    # Normalise posterior to sum to 1 for downstream expectation calculations
    total = posterior.sum()
    return posterior / total if total > 0 else posterior


def decreasing_prune_schedule(weights: np.ndarray, step: int, decay_rate: float = 0.05) -> np.ndarray:
    """
    Compute a time‑dependent pruning probability for each action.
    The probability grows with `step` and is modulated by the weight magnitude.
    Actions with a random draw below the probability are pruned (set to 0).
    """
    if step < 0:
        raise ValueError("step must be non‑negative")
    # Base pruning probability grows exponentially with step
    base_prob = 1.0 - math.exp(-decay_rate * step)
    # Weight‑dependent adjustment (higher weight → lower chance of removal)
    probs = base_prob * (1.0 - weights)  # weights close to 1 survive more
    random_vals = np.random.random(weights.shape)
    mask = random_vals >= probs  # True means keep
    return weights * mask.astype(float)


def fisher_score_centroid(actions: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute a Fisher‑score weighted centroid angle.
    Actions are assumed to be 1‑D positions on a line (e.g., indices).
    The centroid angle is arctan2 of the weighted sum.
    """
    if actions.ndim != 1 or weights.ndim != 1:
        raise ValueError("actions and weights must be 1‑D")
    if actions.size != weights.size:
        raise ValueError("actions and weights must have the same length")
    weighted_sum = np.sum(actions * weights)
    total_weight = np.sum(weights)
    if total_weight == 0:
        return 0.0
    centroid = weighted_sum / total_weight
    # Map centroid (a scalar) to an angle in [0, 2π)
    return (centroid % len(actions)) / len(actions) * 2 * math.pi


def gini_adjusted_expectation(weights: np.ndarray, values: np.ndarray) -> float:
    """
    Compute the expected value of `values` under `weights`,
    then scale it by (1 - Gini) to favour more equitable distributions.
    """
    if weights.shape != values.shape:
        raise ValueError("weights and values must share shape")
    exp_val = np.sum(weights * values)
    gini = gini_coefficient(weights)
    return exp_val * (1.0 - gini)


# ----------------------------------------------------------------------
# Public hybrid interface (three demonstrative functions)
# ----------------------------------------------------------------------


def hybrid_lsm_regret_weights(text: str, num_actions: int, step: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    End‑to‑end pipeline:
    1. Compute LSM vector from `text`.
    2. Derive certainty factor `c`.
    3. Generate regret prior `r` for `num_actions`.
    4. Obtain posterior weights `w` via Bayesian update.
    5. Apply one step of the decreasing‑rate pruning schedule.

    Returns:
        (final_weights, posterior_before_prune)
    """
    lsm = lsm_vector(text)
    certainty = compute_certainty_factor(lsm)
    regret = generate_regret_vector(num_actions)
    posterior = posterior_weights_from_regret(regret, certainty)
    pruned = decreasing_prune_schedule(posterior, step)
    return pruned, posterior


def hybrid_prune_and_localize(text: str, actions: List[int], step: int) -> Tuple[List[int], float]:
    """
    Uses the hybrid weight vector to prune actions and then computes
    a Fisher‑score centroid angle.

    Returns:
        (list_of_surviving_action_ids, centroid_angle_radians)
    """
    num_actions = len(actions)
    final_weights, _ = hybrid_lsm_regret_weights(text, num_actions, step)
    # Keep actions whose weight survived (non‑zero)
    surviving = [act for act, w in zip(actions, final_weights) if w > 0]
    # For centroid we need numeric positions; use the original integer ids
    angle = fisher_score_centroid(np.array(actions, dtype=float), final_weights)
    return surviving, angle


def hybrid_gini_adjusted_expectation(text: str, values: List[float], step: int) -> float:
    """
    Computes a Gini‑adjusted expected value of arbitrary per‑action `values`
    (e.g., rewards) using the hybrid posterior/pruned weight distribution.
    """
    num_actions = len(values)
    final_weights, _ = hybrid_lsm_regret_weights(text, num_actions, step)
    return gini_adjusted_expectation(final_weights, np.array(values, dtype=float))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The quick brown fox jumps over the lazy dog while it "
        "really enjoys very bright sunshine and constantly "
        "asks why the world is so unpredictable."
    )
    actions = list(range(10))  # dummy action identifiers 0‑9
    step = 3

    # Test hybrid_lsm_regret_weights
    weights, posterior = hybrid_lsm_regret_weights(sample_text, len(actions), step)
    print("Posterior (pre‑prune):", posterior)
    print("Final weights (post‑prune):", weights)

    # Test prune and localize
    surviving, angle = hybrid_prune_and_localize(sample_text, actions, step)
    print("Surviving actions:", surviving)
    print(f"Centroid angle (radians): {angle:.4f}")

    # Test Gini‑adjusted expectation with dummy reward values
    rewards = [random.uniform(0, 10) for _ in actions]
    exp_val = hybrid_gini_adjusted_expectation(sample_text, rewards, step)
    print("Gini‑adjusted expected reward:", exp_val)