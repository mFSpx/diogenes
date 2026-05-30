# DARWIN HAMMER — match 449, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_bandit_router_m202_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_voronoi_parti_m381_s0.py (gen3)
# born: 2026-05-29T23:29:02Z

"""Hybrid Thompson‑Bandit / Ollivier‑Ricci Curvature Algorithm

Parents:
- hybrid_hybrid_hybrid_ternar_hybrid_bandit_router_m202_s3.py (Thompson‑sampling bandit)
- hybrid_hybrid_hybrid_krampu_hybrid_voronoi_parti_m381_s0.py (feature extraction + Ollivier‑Ricci curvature)

Mathematical bridge:
The curvature vector 𝜅∈ℝⁿ computed from the raw feature map is interpreted as a
context‑dependent prior shift Δα for the Beta posteriors of the bandit:
    α_i ← α_i + η·κ_î ,
where κ_î is a normalized projection of 𝜅 onto the action space and η>0 is a
tuning coefficient. This couples the geometric information of the data
distribution (via Ollivier‑Ricci curvature) to the Bayesian exploration‑exploitation
mechanism of Thompson sampling, yielding a unified decision‑making system.
"""

import numpy as np
import random
import json
import math
import sys
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 with trailing “Z”."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_context(text: Optional[str]) -> Dict[str, Any]:
    """Parse a JSON string into a dict, returning an empty dict on ``None``."""
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("context must be a JSON object")
    return value


# ----------------------------------------------------------------------
# Bandit core – a minimal Thompson‑sampling Bernoulli bandit
# ----------------------------------------------------------------------
@dataclass
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "thompson_sampling"


@dataclass
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float = 1.0


class ThompsonBandit:
    """A lightweight Thompson‑sampling bandit for continuous rewards."""

    def __init__(self, actions: List[str], prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self._alpha: Dict[str, float] = {a: prior_alpha for a in actions}
        self._beta: Dict[str, float] = {a: prior_beta for a in actions}
        self._actions = actions

    def sample(self) -> str:
        """Draw a sample from each Beta posterior and return the best action."""
        samples = {a: np.random.beta(self._alpha[a], self._beta[a]) for a in self._actions}
        return max(samples, key=samples.get)

    def update(self, upd: BanditUpdate) -> None:
        """Update the posterior for the given action."""
        # Clip reward to [0,1] because Beta expects a probability‑like observation.
        r = max(0.0, min(1.0, upd.reward))
        self._alpha[upd.action_id] += r
        self._beta[upd.action_id] += (1.0 - r)

    # ------------------------------------------------------------------
    # Hybrid extensions
    # ------------------------------------------------------------------
    def adjust_priors(self, weights: Dict[str, float], eta: float = 0.5) -> None:
        """
        Shift the α‑parameters according to external weights (e.g. curvature).
        α_i ← α_i + η·w_i   (β unchanged)
        """
        for a, w in weights.items():
            if a in self._alpha:
                self._alpha[a] += eta * w

    def current_means(self) -> Dict[str, float]:
        """Return the current mean of each Beta posterior (α/(α+β))."""
        return {a: self._alpha[a] / (self._alpha[a] + self._beta[a]) for a in self._actions}


# ----------------------------------------------------------------------
# Feature extraction & Ollivier‑Ricci curvature (parent B)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> dict[str, float]:
    """Generate a synthetic feature vector from free‑form text."""
    # In a real system this would parse *text*; here we synthesize random values.
    features: dict[str, float] = {}
    features.update({
        "operator_visceral_ratio": random.random(),
        "operator_tech_ratio": random.random(),
        "operator_legal_osint_ratio": random.random(),
    })
    features.update({
        "psyche_forensic_shield_ratio": random.random(),
        "psyche_poetic_entropy": random.random(),
        "psyche_dissociative_index": random.random(),
    })
    features.update({
        "resilience_bureaucratic_weaponization_index": random.random(),
        "resilience_resource_exhaustion_metric": random.random(),
        "resilience_swarm_orchestration_density": random.random(),
    })
    features.update({
        "rainmaker_corporate_grit_tension": random.random(),
        "rainmaker_countdown_density": random.random(),
        "rainmaker_asset_structuring_weight": random.random(),
    })
    features.update({
        "telemetry_agent_symmetry_ratio": random.random(),
        "telemetry_protocol_discipline": random.random(),
        "telemetry_manic_velocity": random.random(),
    })
    return features


def calculate_oric_curvature(features: dict[str, float]) -> dict[str, float]:
    """
    Calculates a toy Ollivier‑Ricci curvature for each feature.
    The real Ollivier‑Ricci curvature is far more involved; here we use
    simple scaling factors to emulate a curvature field.
    """
    oric_features: dict[str, float] = {}
    for name, value in features.items():
        if name.startswith("operator"):
            oric_features[name] = value * 0.1
        elif name.startswith("psyche"):
            oric_features[name] = value * 0.2
        elif name.startswith("resilience"):
            oric_features[name] = value * -0.05
        elif name.startswith("rainmaker"):
            oric_features[name] = value * 0.15
        elif name.startswith("telemetry"):
            oric_features[name] = value * -0.1
        else:
            oric_features[name] = value * 0.0
    return oric_features


def curvature_to_action_weights(
    curvatures: dict[str, float],
    actions: List[str],
) -> dict[str, float]:
    """
    Project the high‑dimensional curvature vector onto the discrete action set.
    For demonstration we compute the mean absolute curvature per action by
    hashing the feature name onto an action index.
    """
    # Initialise accumulators
    sums = {a: 0.0 for a in actions}
    counts = {a: 0 for a in actions}

    for fname, curv in curvatures.items():
        # Simple deterministic hash → action mapping
        idx = abs(hash(fname)) % len(actions)
        act = actions[idx]
        sums[act] += abs(curv)
        counts[act] += 1

    # Normalise to obtain a weight per action
    weights: dict[str, float] = {}
    for a in actions:
        if counts[a] > 0:
            weights[a] = sums[a] / counts[a]
        else:
            weights[a] = 0.0
    return weights


# ----------------------------------------------------------------------
# Hybrid operations (three demonstrative functions)
# ----------------------------------------------------------------------
def hybrid_initialize(actions: List[str]) -> ThompsonBandit:
    """Create a ThompsonBandit ready for curvature‑aware updates."""
    return ThompsonBandit(actions=actions, prior_alpha=1.0, prior_beta=1.0)


def hybrid_observe_and_update(
    bandit: ThompsonBandit,
    raw_text: str,
    reward: float,
    context_id: str = "default",
    eta: float = 0.5,
) -> None:
    """
    Full pipeline:
    1. Extract features from *raw_text*.
    2. Compute curvature.
    3. Map curvature to action‑specific prior shifts.
    4. Adjust bandit priors.
    5. Sample an action.
    6. Update the bandit with the observed *reward*.
    """
    # 1‑2
    feats = extract_full_features(raw_text)
    curv = calculate_oric_curvature(feats)

    # 3‑4
    weights = curvature_to_action_weights(curv, bandit._actions)
    bandit.adjust_priors(weights, eta=eta)

    # 5
    chosen = bandit.sample()

    # 6
    upd = BanditUpdate(context_id=context_id, action_id=chosen, reward=reward)
    bandit.update(upd)


def hybrid_predict(
    bandit: ThompsonBandit,
    raw_text: str,
    eta: float = 0.5,
) -> Tuple[str, Dict[str, float]]:
    """
    Return a predicted action together with the current posterior means,
    after a curvature‑driven prior adjustment (non‑destructive).
    """
    # Compute curvature‑based weights
    curv = calculate_oric_curvature(extract_full_features(raw_text))
    weights = curvature_to_action_weights(curv, bandit._actions)

    # Preserve original alphas
    original_alpha = bandit._alpha.copy()

    # Temporary adjustment
    bandit.adjust_priors(weights, eta=eta)
    action = bandit.sample()
    means = bandit.current_means()

    # Restore original alphas to keep the adjustment purely predictive
    bandit._alpha = original_alpha
    return action, means


def hybrid_batch_process(
    bandit: ThompsonBandit,
    texts: List[str],
    rewards: List[float],
    eta: float = 0.5,
) -> List[BanditAction]:
    """
    Process a batch of (text, reward) pairs, returning a list of BanditAction
    objects that capture the decision made for each entry.
    """
    results: List[BanditAction] = []
    for txt, rew in zip(texts, rewards):
        hybrid_observe_and_update(bandit, txt, rew, eta=eta)
        # After the update we can report the current best action and its stats.
        best = bandit.sample()
        mean = bandit.current_means()[best]
        # Confidence bound approximated by standard deviation of Beta
        a = bandit._alpha[best]
        b = bandit._beta[best]
        std = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        results.append(
            BanditAction(
                action_id=best,
                propensity=mean,
                expected_reward=mean,
                confidence_bound=std,
            )
        )
    return results


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    actions = ["alpha", "beta", "gamma"]
    bandit = hybrid_initialize(actions)

    # Simple synthetic texts
    texts = [
        "scenario one: high operator activity",
        "scenario two: psychological stress",
        "scenario three: resilience testing",
    ]
    rewards = [0.7, 0.2, 0.9]

    # Batch processing demonstration
    outcomes = hybrid_batch_process(bandit, texts, rewards, eta=0.3)
    for i, out in enumerate(outcomes):
        print(f"Step {i+1}: action={out.action_id}, mean={out.propensity:.3f}, ci={out.confidence_bound:.3f}")

    # Predictive call without affecting the model
    pred_action, pred_means = hybrid_predict(bandit, "new unseen scenario")
    print(f"Predictive step: chosen={pred_action}, posterior_means={pred_means}")