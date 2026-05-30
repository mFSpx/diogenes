# DARWIN HAMMER — match 82, survivor 1
# gen: 4
# parent_a: hybrid_regret_engine_hybrid_doomsday_cale_m19_s4.py (gen2)
# parent_b: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s1.py (gen3)
# born: 2026-05-29T23:26:48Z

import re
import sys
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    """An action with an expected value and optional cost/risk penalties."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """A counterfactual adjustment for a specific action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Decision‑hygiene cue extraction
# ----------------------------------------------------------------------


EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)

PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)


def extract_decision_hygiene_cues(text: str) -> Dict[str, int]:
    """Count evidence‑ and planning‑related cues in *text*."""
    cues = defaultdict(int)
    cues["evidence"] = len(EVIDENCE_RE.findall(text))
    cues["planning"] = len(PLANNING_RE.findall(text))
    return dict(cues)


# ----------------------------------------------------------------------
# Regret‑weighted core
# ----------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    epsilon: float = 1e-9,
) -> Dict[str, float]:
    """
    Produce a probability distribution over *actions* based on regret.

    Regret for an action is the expected shortfall between the counterfactual
    outcome and the action's nominal expected value.  Negative regrets are set
    to zero (they indicate no loss) and a tiny epsilon guarantees a non‑zero
    denominator.
    """
    # Aggregate regret per action
    regret_map: Dict[str, float] = {a.id: 0.0 for a in actions}
    for cf in counterfactuals:
        if cf.action_id in regret_map:
            regret = (cf.outcome_value - next(a.expected_value for a in actions if a.id == cf.action_id)) * cf.probability
            regret_map[cf.action_id] += max(regret, 0.0)  # discard negative regret

    # Convert regrets to a probability distribution via softmax
    regrets = np.array([regret_map[a.id] for a in actions])
    probs = _softmax(regrets + epsilon)  # epsilon avoids a zero‑vector softmax
    return {a.id: float(p) for a, p in zip(actions, probs)}


# ----------------------------------------------------------------------
# Hybrid integration
# ----------------------------------------------------------------------


def _cue_influence_factor(
    cues: Dict[str, int],
    alpha: float = 0.5,
) -> float:
    """
    Compute a single scalar factor from the extracted cues.

    The factor lies in [1, 1+alpha] and grows with the total cue count.
    """
    total_cues = sum(cues.values())
    if total_cues == 0:
        return 1.0
    # Diminishing returns via a logistic‑like transform
    return 1.0 + alpha * (total_cues / (total_cues + 1.0))


def hybrid_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    text: str,
    beta: float = 0.3,
) -> Dict[str, float]:
    """
    Combine regret‑weighted probabilities with decision‑hygiene cues.

    *beta* controls the blend: 0 → pure regret, 1 → pure cue‑driven uniform
    distribution.  The cue influence is applied as a multiplicative boost to
    the regret distribution, then re‑normalised.
    """
    # 1️⃣ Regret‑based core
    regret_probs = compute_regret_weighted_strategy(actions, counterfactuals)

    # 2️⃣ Extract cues and turn them into a global boost factor
    cues = extract_decision_hygiene_cues(text)
    cue_factor = _cue_influence_factor(cues)

    # 3️⃣ Apply cue factor multiplicatively (preserves ranking)
    boosted = {aid: p * cue_factor for aid, p in regret_probs.items()}

    # 4️⃣ Blend with a uniform distribution to respect *beta*
    uniform_prob = 1.0 / len(actions)
    blended = {
        aid: (1 - beta) * boosted[aid] + beta * uniform_prob for aid in boosted
    }

    # 5️⃣ Final normalisation
    total = sum(blended.values())
    return {aid: prob / total for aid, prob in blended.items()}


# ----------------------------------------------------------------------
# Example usage (executed when run as a script)
# ----------------------------------------------------------------------


if __name__ == "__main__":
    actions = [
        MathAction("action1", 10.0),
        MathAction("action2", 20.0),
        MathAction("action3", 30.0),
    ]

    counterfactuals = [
        MathCounterfactual("action1", 15.0),
        MathCounterfactual("action2", 25.0),
        MathCounterfactual("action3", 35.0),
    ]

    sample_text = "I will verify the evidence and plan the next steps."

    hybrid_dist = hybrid_strategy(actions, counterfactuals, sample_text)
    print("Hybrid probability distribution:")
    for aid, prob in hybrid_dist.items():
        print(f"  {aid}: {prob:.4f}")