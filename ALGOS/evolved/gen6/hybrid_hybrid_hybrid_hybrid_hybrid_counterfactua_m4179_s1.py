# DARWIN HAMMER — match 4179, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1396_s0.py (gen4)
# parent_b: hybrid_counterfactual_effec_hybrid_hybrid_hybrid_m948_s1.py (gen5)
# born: 2026-05-29T23:53:56Z

"""Hybrid algorithm combining semantic‑neighbor Bayesian updates (Parent A) with
causal‑effect priors and Gaussian radial‑basis likelihoods (Parent B).

Mathematical bridge:
- The causal average‑treatment‑effect (ATE) from Parent B is interpreted as a
  prior probability `π` for a hypothesis about a target document.
- Semantic neighbour distances `d_i` (Parent A) are transformed by the Gaussian
  kernel `g(d_i)=exp(-(ε·d_i)²)` (Parent B) to obtain likelihoods `ℓ_i`.
- Bayesian marginal `P(E)=ℓ_i·π + (1-π)·(1-ℓ_i)` and posterior
  `P(H|E)=π·ℓ_i / P(E)` are computed with the same formulas used in Parent A.
- The posterior modulates a liquid‑time‑constant (LTC) integrator:
  `x_{t+1}=x_t + (Δt/τ_i)·(ℓ_i - x_t)`,
  where the time‑constant `τ_i` is inversely proportional to the posterior,
  creating a dynamic system that reacts faster to neighbours with higher
  posterior probability.

The module provides three core functions that demonstrate this fused
behaviour and a small smoke test.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Sequence, List, Tuple, Dict, Iterable

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Parent A utilities (Bayesian update, label scoring, distance)
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability P(E) = ℓ·π + fp·(1-π)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior probability P(H|E) = π·ℓ / P(E)."""
    if marginal <= 0.0:
        raise ValueError("Marginal must be > 0")
    return prior * likelihood / marginal

def label_score(text: str, label: str) -> float:
    """Simple literal fallback: count occurrences of label."""
    return float(text.count(label))

# ----------------------------------------------------------------------
# Parent B utilities (causal effect, Gaussian kernel, Euclidean)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CausalEffect:
    effect_id: str
    treatment: str
    outcome: str
    confounders: Tuple[str, ...]
    ate_estimate: float | None
    ate_confidence_interval: Tuple[float, float] | None
    refutation_passed: bool
    refutation_methods: Tuple[str, ...]
    heterogeneous_effects: Dict[str, float]

def estimate_causal_effect(
    treatment: str,
    outcome: str,
    confounders: List[str],
    data: Dict[str, List[float]],
) -> CausalEffect:
    """Very lightweight ATE estimator (identical to Parent B)."""
    t = list(map(float, data.get(treatment, [])))
    y = list(map(float, data.get(outcome, [])))
    if not t or len(t) != len(y):
        ate = None
        ci = None
    else:
        yt = [yy for tt, yy in zip(t, y) if tt >= 0.5]
        yc = [yy for tt, yy in zip(t, y) if tt < 0.5]
        ate = (sum(yt) / len(yt) - sum(yc) / len(yc)) if yt and yc else None
        spread = (
            math.sqrt(sum((yy - sum(y) / len(y)) ** 2 for yy in y) / len(y))
            if len(y) > 1
            else 0.0
        )
        ci = None if ate is None else (ate - spread, ate + spread)
    return CausalEffect(
        effect_id=str(random.getrandbits(128)),
        treatment=treatment,
        outcome=outcome,
        confounders=tuple(confounders),
        ate_estimate=ate,
        ate_confidence_interval=ci,
        refutation_passed=ate is not None,
        refutation_methods=("placebo_treatment", "data_subset", "random_common_cause"),
        heterogeneous_effects={},
    )

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Radial‑basis Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance for arbitrary‑dimensional vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# ----------------------------------------------------------------------
# Hybrid components
# ----------------------------------------------------------------------
def semantic_neighbors(doc_id: str, k: int = 5) -> List[Tuple[str, float]]:
    """
    Mocked semantic neighbour retrieval.
    Returns `k` neighbour identifiers together with a synthetic distance in [0, 2].
    """
    random.seed(hash(doc_id) & 0xFFFFFFFF)
    neighbours = []
    for i in range(k):
        neighbour_id = f"{doc_id}_nbr_{i}"
        # distance simulated as a uniform random number; smaller = more similar
        distance = random.uniform(0.0, 2.0)
        neighbours.append((neighbour_id, distance))
    return neighbours

def _ate_to_prior(ate: float | None) -> float:
    """
    Convert an ATE (which may be any real number) into a prior probability in [0,1]
    using a logistic squashing function.
    """
    if ate is None:
        return 0.5  # uninformed prior
    return 1.0 / (1.0 + math.exp(-ate))

def hybrid_posterior_for_neighbors(
    doc_id: str,
    treatment: str,
    outcome: str,
    confounders: List[str],
    data: Dict[str, List[float]],
    epsilon: float = 1.0,
    false_positive: float = 0.01,
) -> List[Tuple[str, float, float]]:
    """
    Compute posterior probabilities for each semantic neighbour.

    Returns a list of tuples:
        (neighbour_id, likelihood, posterior)
    """
    # 1️⃣ Prior from causal effect
    ce = estimate_causal_effect(treatment, outcome, confounders, data)
    prior = _ate_to_prior(ce.ate_estimate)

    # 2️⃣ Likelihoods from semantic distances via Gaussian kernel
    neighbours = semantic_neighbors(doc_id)
    results = []
    for nid, dist in neighbours:
        likelihood = gaussian(dist, epsilon)  # ℓ_i ∈ (0,1]
        marginal = bayes_marginal(prior, likelihood, false_positive)
        posterior = bayes_update(prior, likelihood, marginal)
        results.append((nid, likelihood, posterior))
    return results

def liquid_time_constant_update(
    state: float,
    input_signal: float,
    posterior: float,
    base_tau: float = 1.0,
    dt: float = 1.0,
) -> float:
    """
    Single LTC integration step.

    The effective time‑constant τ_i = base_tau / (posterior + 1e-6) makes the
    system react faster when the posterior is high.
    """
    tau = base_tau / (posterior + 1e-6)
    return state + (dt / tau) * (input_signal - state)

def hybrid_resource_allocation(
    doc_id: str,
    treatment: str,
    outcome: str,
    confounders: List[str],
    data: Dict[str, List[float]],
    init_state: float = 0.0,
    steps: int = 5,
) -> Dict[str, float]:
    """
    Simulate a dynamic allocation process where each neighbour contributes an
    LTC‑driven signal. The final state per neighbour is returned.

    The process:
        * Compute posterior for each neighbour.
        * Initialise its LTC state to `init_state`.
        * Iterate `steps` times, feeding the neighbour's likelihood as the input
          signal and updating the state with `liquid_time_constant_update`.
    """
    neighbour_info = hybrid_posterior_for_neighbors(
        doc_id, treatment, outcome, confounders, data
    )
    final_states: Dict[str, float] = {}
    for nid, likelihood, posterior in neighbour_info:
        state = init_state
        for _ in range(steps):
            state = liquid_time_constant_update(state, likelihood, posterior)
        final_states[nid] = state
    return final_states

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal synthetic dataset
    synthetic_data = {
        "drug_A": [random.random() for _ in range(100)],
        "recovery": [random.random() for _ in range(100)],
    }
    conf = ["age", "sex"]
    result = hybrid_resource_allocation(
        doc_id="patient_123",
        treatment="drug_A",
        outcome="recovery",
        confounders=conf,
        data=synthetic_data,
        init_state=0.0,
        steps=10,
    )
    print("Final LTC states per neighbour:")
    for nid, val in result.items():
        print(f"  {nid}: {val:.4f}")