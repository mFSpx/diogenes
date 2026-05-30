# DARWIN HAMMER — match 2739, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_label_foundry_m122_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_cockpi_m582_s0.py (gen4)
# born: 2026-05-29T23:43:50Z

"""
Hybrid algorithm merging:
- Parent A: regex‑based feature extraction and labeling (hybrid_hybrid_hybrid_decisi_label_foundry_m122_s0)
- Parent B: bandit router with Schoolfield developmental rate and trust‑weighted scaling (hybrid_hybrid_hybrid_bandit_hybrid_hybrid_cockpi_m582_s0)

Mathematical bridge:
The labeling functions of Parent A produce per‑category counts that are interpreted as *trust scores*.
These trust scores are normalized and then used to scale the Schoolfield developmental rate
from Parent B (rate × trust). The scaled rate modulates the propensity of each BanditAction,
creating a unified decision‑making pipeline that couples textual evidence with adaptive
bandit dynamics.
"""

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regex feature extraction (labels)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b|"
    r"\b(?:kill|die|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)

REGEX_LABELS: List[Tuple[str, re.Pattern]] = [
    ("evidence", EVIDENCE_RE),
    ("planning", PLANNING_RE),
    ("delay", DELAY_RE),
    ("support", SUPPORT_RE),
    ("boundary", BOUNDARY_RE),
    ("outcome", OUTCOME_RE),
    ("impulsive", IMPULSIVE_RE),
    ("scarcity", SCARCITY_RE),
]

def extract_features(text: str) -> Dict[str, int]:
    """Count occurrences of each regex label in *text*."""
    counts: Dict[str, int] = {}
    for label, pattern in REGEX_LABELS:
        matches = pattern.findall(text)
        counts[label] = len(matches)
    return counts

def compute_trust_weights(counts: Dict[str, int]) -> Dict[str, float]:
    """
    Convert raw label counts into a normalized trust vector.
    Trust for a label = count / (total counts + ε)  (ε prevents division by zero).
    """
    epsilon = 1e-9
    total = sum(counts.values()) + epsilon
    return {label: cnt / total for label, cnt in counts.items()}

# ----------------------------------------------------------------------
# Parent B – bandit router with Schoolfield developmental rate
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0          # baseline rate at 25 °C
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15        # 10 °C in Kelvin
    t_high: float = 307.15       # 34 °C in Kelvin
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987         # universal gas constant cal·mol⁻¹·K⁻¹

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def _reward(action_id: str) -> float:
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def schoolfield_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield developmental rate (temperature dependent)."""
    if temp_k <= 0:
        raise ValueError("Temperature in Kelvin must be positive")
    # Arrhenius term
    arr = math.exp(-params.delta_h_activation / (params.r_cal * temp_k))
    # Low‑temperature inhibition
    low = 1.0 + math.exp(params.delta_h_low / (params.r_cal * (params.t_low - temp_k)))
    # High‑temperature inhibition
    high = 1.0 + math.exp(params.delta_h_high / (params.r_cal * (temp_k - params.t_high)))
    return params.rho_25 * arr / (low * high)

# ----------------------------------------------------------------------
# Hybrid functions – bridging the two parents
# ----------------------------------------------------------------------
def scaled_developmental_rate(
    temp_c: float,
    trust: Dict[str, float],
    params: SchoolfieldParams = SchoolfieldParams()
) -> float:
    """
    Compute the Schoolfield rate at *temp_c* and scale it by an aggregated trust factor.
    Aggregated trust = weighted sum of label trusts (simple mean here).
    """
    temp_k = c_to_k(temp_c)
    base_rate = schoolfield_rate(temp_k, params)
    if not trust:
        return base_rate
    aggregated_trust = sum(trust.values()) / len(trust)
    return base_rate * aggregated_trust

def select_action(
    context_id: str,
    actions: List[BanditAction],
    temp_c: float,
    feature_counts: Dict[str, int]
) -> BanditAction:
    """
    Choose an action by combining:
    - expected reward (from policy)
    - propensity adjusted by the scaled developmental rate
    - confidence bound (UCB style)
    """
    trust = compute_trust_weights(feature_counts)
    rate = scaled_developmental_rate(temp_c, trust)

    # Compute a score for each action
    scored = []
    for act in actions:
        reward_est = _reward(act.action_id)
        # propensity is multiplied by the temperature‑trust scaled rate
        adjusted_propensity = act.propensity * rate
        # Upper Confidence Bound component
        ucb = adjusted_propensity + act.confidence_bound
        total_score = reward_est + ucb
        scored.append((total_score, act))

    # Return the action with highest total_score
    best_action = max(scored, key=lambda x: x[0])[1]
    return best_action

def generate_updates(
    selected: BanditAction,
    context_id: str,
    observed_reward: float
) -> List[BanditUpdate]:
    """
    Produce a list containing a single BanditUpdate for the chosen action.
    The propensity recorded is the action's original propensity (pre‑scaling)
    to keep the policy unbiased.
    """
    upd = BanditUpdate(
        context_id=context_id,
        action_id=selected.action_id,
        reward=observed_reward,
        propensity=selected.propensity,
    )
    return [upd]

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "We need to verify the source and citation before we can finalize the plan. "
        "If there is any delay, please let us know. Also, reach out to a friend for support."
    )
    # Feature extraction (Parent A)
    counts = extract_features(sample_text)
    print("Feature counts:", counts)

    # Define a small action set (Parent B)
    actions = [
        BanditAction(action_id="A1", propensity=0.3, expected_reward=0.0, confidence_bound=0.2),
        BanditAction(action_id="A2", propensity=0.5, expected_reward=0.0, confidence_bound=0.1),
        BanditAction(action_id="A3", propensity=0.2, expected_reward=0.0, confidence_bound=0.3),
    ]

    # Simulated temperature
    temperature_c = 22.0

    # Choose action using hybrid logic
    chosen = select_action("ctx-001", actions, temperature_c, counts)
    print("Chosen action:", chosen)

    # Simulate an observed reward
    reward = random.uniform(0, 1)

    # Generate and apply update
    ups = generate_updates(chosen, "ctx-001", reward)
    update_policy(ups)

    # Show updated policy statistics
    for aid in _POLICY:
        print(f"Policy[{aid}] = total_reward={_POLICY[aid][0]:.3f}, count={_POLICY[aid][1]}")
    sys.exit(0)