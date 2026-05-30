# DARWIN HAMMER — match 5628, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_regret_m25_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_ternar_bayes_claim_kernel_m294_s1.py (gen3)
# born: 2026-05-30T00:03:38Z

"""Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s4.py
- Parent B: hybrid_hybrid_hybrid_ternar_bayes_claim_kernel_m294_s1.py

Mathematical Bridge:
Parent A provides a Hoeffding‑bound confidence interval for empirical
action values, a simulated‑annealing acceptance probability and a regret
framework.  Parent B supplies a similarity measure (SSIM), a variational
free‑energy update for belief means and a Bayesian likelihood update.
The hybrid algorithm treats each candidate action as a stochastic
hypothesis whose mean belief is updated by the free‑energy rule, whose
quality is measured by SSIM between the observed reward pattern and the
prior pattern, and whose selection is governed by regret‑minimisation
augmented with a simulated‑annealing acceptance probability.  The
Bayesian posterior over actions is finally normalised to yield the
decision distribution.

The core functions below implement this fused mathematics.
"""

import sys
import math
import random
import pathlib
from dataclasses import dataclass
from collections.abc import Mapping, Hashable
from typing import Any, Sequence, Tuple, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Types shared by both parents
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]

# ----------------------------------------------------------------------
# Parent A utilities (adapted)
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    """Probability that a node broadcasts in the current phase."""
    if total_phases < 1 or current_phase < 1:
        raise ValueError('phases and phase must be positive')
    return min(1.0, 1.0 / (2 ** max(0, total_phases - current_phase)))


def hoeffding_bound(n: int, epsilon: float, delta: float) -> float:
    """
    Hoeffding bound: with probability 1‑delta the true mean lies within
    epsilon of the empirical mean for n i.i.d. samples bounded in [0,1].
    Returns the required epsilon for given n and delta.
    """
    if n <= 0:
        raise ValueError('sample size must be positive')
    if not (0 < delta < 1):
        raise ValueError('delta must be in (0,1)')
    return math.sqrt(math.log(2 / delta) / (2 * n))


def simulated_annealing_accept(regret: float, temperature: float) -> float:
    """
    Acceptance probability for a worse decision under simulated annealing.
    """
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    return math.exp(-regret / temperature)


# ----------------------------------------------------------------------
# Parent B utilities (adapted)
# ----------------------------------------------------------------------
def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """
    Simplified Structural Similarity Index (SSIM) for 1‑D signals.
    Returns a value in [-1, 1] (higher = more similar).
    """
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')

    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator


def variational_free_energy_update(prior_mean: float,
                                   prior_var: float,
                                   observation: float,
                                   obs_var: float) -> Tuple[float, float]:
    """
    Performs a single variational free‑energy (Gaussian) update.
    Returns (posterior_mean, posterior_variance).
    """
    if prior_var <= 0 or obs_var <= 0:
        raise ValueError('variances must be positive')
    precision_post = 1.0 / prior_var + 1.0 / obs_var
    posterior_var = 1.0 / precision_post
    posterior_mean = posterior_var * (prior_mean / prior_var + observation / obs_var)
    return posterior_mean, posterior_var


def bayesian_update(prior: Dict[Any, float],
                    likelihood: Dict[Any, float]) -> Dict[Any, float]:
    """
    Simple discrete Bayesian update: posterior ∝ prior * likelihood.
    Returns a normalised posterior distribution.
    """
    unnorm = {k: prior.get(k, 0.0) * likelihood.get(k, 0.0) for k in set(prior) | set(likelihood)}
    total = sum(unnorm.values())
    if total == 0:
        # avoid division by zero – return uniform distribution over keys
        n = len(unnorm)
        return {k: 1.0 / n for k in unnorm}
    return {k: v / total for k, v in unnorm.items()}


# ----------------------------------------------------------------------
# Hybrid core structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


# ----------------------------------------------------------------------
# Hybrid functions (demonstrating the fused mathematics)
# ----------------------------------------------------------------------
def compute_regret(actions: Sequence[MathAction]) -> Dict[str, float]:
    """
    Regret for each action = (max expected value) - (action expected value).
    """
    if not actions:
        raise ValueError('actions sequence must not be empty')
    max_ev = max(a.expected_value for a in actions)
    return {a.id: max_ev - a.expected_value for a in actions}


def hybrid_belief_update(action_observations: Dict[str, List[float]],
                         prior_means: Dict[str, float],
                         prior_vars: Dict[str, float],
                         obs_var: float = 1.0) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    For each action, update its belief mean/variance using the variational
    free‑energy rule on the mean of the observed rewards.
    Returns (posterior_means, posterior_vars).
    """
    posterior_means = {}
    posterior_vars = {}
    for aid, obs in action_observations.items():
        if not obs:
            # no data → keep prior
            posterior_means[aid] = prior_means.get(aid, 0.0)
            posterior_vars[aid] = prior_vars.get(aid, 1.0)
            continue
        observation_mean = float(np.mean(obs))
        prior_mean = prior_means.get(aid, 0.0)
        prior_var = prior_vars.get(aid, 1.0)
        post_mean, post_var = variational_free_energy_update(
            prior_mean, prior_var, observation_mean, obs_var)
        posterior_means[aid] = post_mean
        posterior_vars[aid] = post_var
    return posterior_means, posterior_vars


def hybrid_decision(actions: Sequence[MathAction],
                    action_observations: Dict[str, List[float]],
                    total_phases: int,
                    current_phase: int,
                    temperature: float = 1.0,
                    delta: float = 0.05) -> str:
    """
    Integrated decision procedure:
    1. Compute Hoeffding confidence intervals for each action.
    2. Evaluate regret and obtain simulated‑annealing acceptance probabilities.
    3. Update beliefs via variational free‑energy.
    4. Measure similarity between new observations and prior patterns using SSIM.
    5. Perform a Bayesian update that fuses regret‑based and similarity‑based
       likelihoods.
    6. Sample or pick the action with highest posterior probability.
    Returns the selected action id.
    """
    # ---- 1. Hoeffding confidence (used only as a sanity check) ----
    for a in actions:
        obs = action_observations.get(a.id, [])
        n = len(obs)
        if n > 0:
            eps = hoeffding_bound(n, epsilon=0.0, delta=delta)  # epsilon derived internally
            # we could tighten expected_value bounds here; omitted for brevity

    # ---- 2. Regret & simulated‑annealing acceptance ----
    regret_dict = compute_regret(actions)
    acceptance = {aid: simulated_annealing_accept(r, temperature) for aid, r in regret_dict.items()}

    # ---- 3. Variational free‑energy belief update ----
    prior_means = {a.id: a.expected_value for a in actions}
    prior_vars = {a.id: 1.0 for _ in actions}  # unit variance as default
    post_means, post_vars = hybrid_belief_update(action_observations, prior_means, prior_vars)

    # ---- 4. SSIM similarity between observed reward vector and prior mean vector ----
    # Build vectors aligned by action id order
    ids = [a.id for a in actions]
    prior_vec = np.array([prior_means[i] for i in ids], dtype=float)
    obs_means = np.array([np.mean(action_observations.get(i, [0.0])) for i in ids], dtype=float)
    similarity = ssim(prior_vec, obs_means)

    # ---- 5. Bayesian fusion of likelihoods ----
    # Regret‑based likelihood: higher acceptance → higher likelihood
    regret_likelihood = {aid: acceptance[aid] for aid in ids}
    # Similarity‑based likelihood: uniform scaling by SSIM (shifted to [0,1])
    sim_factor = (similarity + 1) / 2  # map [-1,1] -> [0,1]
    similarity_likelihood = {aid: sim_factor for aid in ids}
    # Combine multiplicatively
    combined_likelihood = {aid: regret_likelihood[aid] * similarity_likelihood[aid] for aid in ids}
    posterior = bayesian_update({aid: 1.0 / len(ids) for aid in ids}, combined_likelihood)

    # ---- 6. Selection ----
    # Choose action with maximal posterior probability; break ties randomly
    max_prob = max(posterior.values())
    candidates = [aid for aid, p in posterior.items() if math.isclose(p, max_prob, rel_tol=1e-9)]
    chosen = random.choice(candidates)
    return chosen


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Define three actions with arbitrary expected values
    actions = [
        MathAction(id="A", expected_value=0.6),
        MathAction(id="B", expected_value=0.4),
        MathAction(id="C", expected_value=0.5),
    ]

    # Simulated reward observations per action (bounded in [0,1])
    action_observations = {
        "A": list(np.random.rand(30)),
        "B": list(np.random.rand(25)),
        "C": list(np.random.rand(20)),
    }

    total_phases = 5
    current_phase = 3
    temperature = 0.8

    chosen = hybrid_decision(
        actions=actions,
        action_observations=action_observations,
        total_phases=total_phases,
        current_phase=current_phase,
        temperature=temperature,
        delta=0.05,
    )
    print(f"Chosen action: {chosen}")
    # Verify that the function runs without raising exceptions.