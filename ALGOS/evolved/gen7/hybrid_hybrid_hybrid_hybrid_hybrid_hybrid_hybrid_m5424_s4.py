# DARWIN HAMMER — match 5424, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_physar_m1865_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m813_s0.py (gen5)
# born: 2026-05-30T00:02:02Z

import numpy as np
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Iterable, Mapping, Any


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Return a normalized risk score in [0,1] based on the proportion of unique identifiers."""
    if total_records <= 0:
        return 0.0
    return float(np.clip(unique_quasi_identifiers / total_records, 0.0, 1.0))


def _confounder_adjustment(confounders: Mapping[str, float]) -> float:
    """
    Produce a multiplicative adjustment from confounder importance scores.
    The adjustment is in (0,1] – stronger confounding yields a smaller factor.
    """
    if not confounders:
        return 1.0
    # Example: geometric mean of (1 - importance) to keep within (0,1]
    factors = [1.0 - np.clip(v, 0.0, 1.0) for v in confounders.values()]
    return float(np.prod(factors) ** (1.0 / len(factors)))


def weighted_causal_effect(
    ate_estimate: float,
    risk_score: float,
    confounder_importances: Mapping[str, float] | Tuple[str, ...] = (),
) -> float:
    """
    Compute a causal reward that discounts the ATE by both privacy risk and
    confounding strength.

    Parameters
    ----------
    ate_estimate: float
        Average Treatment Effect estimated from data.
    risk_score: float
        Normalised reconstruction risk in [0,1].
    confounder_importances: mapping or tuple
        If a mapping, keys are confounder names and values are importance scores
        in [0,1]. If a tuple, it is interpreted as a list of confounder names with
        unknown importance – in that case a neutral adjustment (1.0) is used.

    Returns
    -------
    float
        The weighted causal effect, guaranteed to be non‑negative.
    """
    if ate_estimate < 0:
        raise ValueError("ATE estimate should be non‑negative for reward computation.")
    risk_factor = 1.0 - np.clip(risk_score, 0.0, 1.0)
    if isinstance(confounder_importances, Mapping):
        conf_factor = _confounder_adjustment(confounder_importances)
    else:
        conf_factor = 1.0
    return float(max(0.0, ate_estimate * risk_factor * conf_factor))


def compute_regret_weighted_strategy(
    actions: List[Dict[str, Any]],
    counterfactuals: List[Dict[str, Any]],
    reward: float,
    temperature: float = 0.1,
) -> Dict[str, float]:
    """
    Produce a probability distribution over actions using regret signals
    and the causal reward.

    The regret for each action is the expected loss relative to its
    counterfactual outcomes, scaled by the causal reward.  The final
    probabilities are obtained via a soft‑max over negative regrets.

    Parameters
    ----------
    actions: list of dicts
        Each dict must contain ``id`` and ``expected_value`` keys.
    counterfactuals: list of dicts
        Each dict must contain ``action_id``, ``outcome_value`` and ``probability``.
    reward: float
        The weighted causal effect to bias the regret.
    temperature: float
        Soft‑max temperature; lower values make the policy more deterministic.

    Returns
    -------
    dict
        Mapping from action id to selection probability.
    """
    if not actions:
        return {}

    # Build lookup tables
    exp_vals = {a["id"]: float(a["expected_value"]) for a in actions}
    regrets = {aid: 0.0 for aid in exp_vals}

    # Accumulate regret contributions from counterfactuals
    for cf in counterfactuals:
        aid = cf.get("action_id")
        if aid not in exp_vals:
            continue
        outcome = float(cf.get("outcome_value", 0.0))
        prob = float(cf.get("probability", 0.0))
        # Regret = (counterfactual outcome - expected) * probability
        regrets[aid] += (outcome - exp_vals[aid]) * prob

    # Incorporate the causal reward as a uniform shift that reduces regret
    for aid in regrets:
        regrets[aid] -= reward  # reward lowers perceived regret

    # Convert regrets to a probability distribution via soft‑max
    # Use negative regrets so lower regret → higher probability
    regret_arr = np.array([regrets[aid] for aid in exp_vals])
    scaled = -regret_arr / max(temperature, 1e-8)
    max_shift = np.max(scaled)  # for numerical stability
    exp_vals_soft = np.exp(scaled - max_shift)
    probs = exp_vals_soft / exp_vals_soft.sum()

    return {aid: float(p) for aid, p in zip(exp_vals.keys(), probs)}


def hybrid_conductance_update(
    conductance: float,
    weighted_effect: float,
    span: Span,
    dt: float = 1.0,
    gain: float = 1.0,
    decay: float = 0.05,
) -> float:
    """
    Update a conductance variable using a leaky integrator that also
    respects the semantic richness of the processed span.

    The span contributes via its length and confidence score; longer,
    higher‑scoring spans amplify the effect of the causal reward.
    """
    if conductance < 0:
        raise ValueError("Conductance must be non‑negative.")
    # Normalise span influence to [0,1]
    length_norm = np.tanh(span.length / 100.0)  # saturates for very long spans
    score_norm = np.clip(span.score, 0.0, 1.0)
    span_factor = (length_norm + score_norm) / 2.0

    q = weighted_effect * span_factor
    new_conductance = conductance + dt * (gain * abs(q) - decay * conductance)
    return float(max(0.0, new_conductance))


def broadcast_probability(total_phases: int, current_phase: int, alpha: float = 0.7) -> float:
    """
    Compute a broadcast probability that decays smoothly across phases.
    Uses a logistic schedule controlled by ``alpha`` (0 < alpha < 1).

    Early phases have higher probability; later phases taper off.
    """
    if total_phases <= 0 or current_phase <= 0:
        raise ValueError("Phase counters must be positive integers.")
    if current_phase > total_phases:
        return 0.0
    # Normalised phase index in [0,1]
    t = (current_phase - 1) / (total_phases - 1)
    # Logistic decay
    prob = 1.0 / (1.0 + np.exp(alpha * (t - 0.5) * 12))  # 12 ≈ steepness factor
    return float(np.clip(prob, 0.0, 1.0))


def hybrid_algorithm(
    actions: List[Dict[str, Any]],
    counterfactuals: List[Dict[str, Any]],
    ate_estimate: float,
    risk_score: float,
    confounder_importances: Mapping[str, float] | Tuple[str, ...],
    conductance: float,
    span: Span,
    total_phases: int,
    current_phase: int,
    temperature: float = 0.1,
) -> Tuple[Dict[str, float], float, float]:
    """
    End‑to‑end hybrid computation.

    Returns
    -------
    (policy, updated_conductance, broadcast_probability)
    """
    weighted_effect = weighted_causal_effect(
        ate_estimate, risk_score, confounder_importances
    )
    policy = compute_regret_weighted_strategy(
        actions, counterfactuals, weighted_effect, temperature=temperature
    )
    updated_conductance = hybrid_conductance_update(
        conductance, weighted_effect, span
    )
    broadcast_prob = broadcast_probability(total_phases, current_phase)
    return policy, updated_conductance, broadcast_prob


if __name__ == "__main__":
    # Example usage with richer confounder information
    actions = [
        {"id": "a1", "expected_value": 0.5},
        {"id": "a2", "expected_value": 0.3},
        {"id": "a3", "expected_value": 0.4},
    ]
    counterfactuals = [
        {"action_id": "a1", "outcome_value": 0.7, "probability": 0.2},
        {"action_id": "a2", "outcome_value": 0.4, "probability": 0.1},
        {"action_id": "a3", "outcome_value": 0.6, "probability": 0.15},
    ]
    ate_estimate = 0.22
    risk_score = reconstruction_risk_score(12, 150)
    confounder_importances = {"c1": 0.3, "c2": 0.6}
    conductance = 0.12
    span = Span(0, 45, "sample text", "entity", 0.78)
    total_phases = 6
    current_phase = 2

    policy, new_cond, prob = hybrid_algorithm(
        actions,
        counterfactuals,
        ate_estimate,
        risk_score,
        confounder_importances,
        conductance,
        span,
        total_phases,
        current_phase,
    )
    print("Policy:", policy)
    print("Updated conductance:", new_cond)
    print("Broadcast probability:", prob)