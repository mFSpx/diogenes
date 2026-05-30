# DARWIN HAMMER — match 3950, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hard_t_m2491_s0.py (gen6)
# parent_b: hybrid_rectified_flow_hybrid_hybrid_hard_t_m184_s0.py (gen3)
# born: 2026-05-29T23:52:44Z

"""HybridRegretFlowBandit
Combines the regret‑weighted bandit core from *hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m410_s3.py* with the
stylometry‑based LSM vector and rectified‑flow matching ideas from
*hybrid_rectified_flow_hybrid_hybrid_hard_t_m184_s0.py*.

Mathematical bridge
-------------------
1. **Context extraction** – the source text is transformed into a
   linguistic‑style‑matrix (LSM) vector `c ∈ ℝ^K` using the categorical
   word counts defined in parent B.
2. **Temperature scaling** – the Schoolfield temperature‑performance curve
   `S(T;θ)` (parent A) provides a scalar temperature factor for each
   decision step.
3. **Regret‑adjusted score** – for an action `a` we compute  

   
   r_a = E_a – λ·risk_a                     # regret‑adjusted value
   s_a = Σ_{cat∈tokens_a} c_cat             # stylometry contribution
   q_a = S(T;θ)·r_a + α·s_a                 # fused quality score
   

   where `λ,α` are hyper‑parameters.
4. **Rectified flow mapping** – the fused scores are interpreted as a
   one‑dimensional “target” and interpolated from a zero source using a
   rectified flow step  

   
   f_a(T,t) = ReLU( (1‑t)·0 + t·q_a )
   

   The resulting vector `f_a` is fed to a soft‑max to obtain the
   selection propensity.

The module implements the above pipeline in three public functions and
provides a minimal smoke‑test."""

import math
import random
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regret‑weighted bandit structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    """Immutable description of an action used by the regret‑weighted component."""
    id: str
    tokens: Tuple[str, ...]          # token set for stylometry weighting
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0                # risk ≥ 0, higher values increase regret non‑linearly


@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters of the Schoolfield temperature‑performance curve."""
    E0: float          # pre‑exponential factor
    Eh: float          # activation energy (J/mol)
    Dh: float          # deactivation energy (J/mol)
    Th: float          # temperature offset (K)
    Td: float          # deactivation midpoint (K)
    Delta: float       # deactivation width (K)
    R: float = 8.314   # universal gas constant (J/(mol·K))


def schoolfield_performance(T: float, p: SchoolfieldParams) -> float:
    """Compute the Schoolfield performance factor at temperature T (°C)."""
    # Convert to Kelvin with offset
    TK = T + 273.15
    # Arrhenius term (activation)
    act = p.E0 * math.exp(-p.Eh / (p.R * (TK + p.Th)))
    # Deactivation term (logistic)
    deact = 1.0 / (1.0 + math.exp((p.Td - TK) / p.Delta))
    return act * deact


# ----------------------------------------------------------------------
# Parent B – stylometry (LSM) and rectified flow utilities
# ----------------------------------------------------------------------


FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}


def words(text: str) -> List[str]:
    """Tokenise a string into lower‑case alphabetic words."""
    return [w for w in (text or "").lower().split() if w.isalpha()]


def lsm_vector(text: str) -> Dict[str, float]:
    """Return the linguistic‑style‑matrix (LSM) vector for *text*."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def rectified_flow(source: np.ndarray, target: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation from *source* to *target* at time *t* with ReLU."""
    interp = (1.0 - t) * source + t * target
    return np.maximum(0.0, interp)


# ----------------------------------------------------------------------
# Hybrid core – three public functions
# ----------------------------------------------------------------------


def compute_fused_quality(
    action: MathAction,
    lsm: Dict[str, float],
    temp: float,
    sf_params: SchoolfieldParams,
    lambda_risk: float = 1.0,
    alpha_styl: float = 1.0,
) -> float:
    """
    Fuse regret‑adjusted value, stylometry contribution and temperature scaling.

    q = S(T)·(E – λ·risk) + α·Σ_{cat∈tokens} lsm[cat]
    """
    # 1. Regret‑adjusted expected value
    regret_val = action.expected_value - lambda_risk * action.risk

    # 2. Stylometry contribution (only categories that appear in tokens)
    styl_score = sum(lsm.get(cat, 0.0) for cat in action.tokens)

    # 3. Temperature factor via Schoolfield curve
    temp_factor = schoolfield_performance(temp, sf_params)

    return temp_factor * regret_val + alpha_styl * styl_score


def select_action_via_flow(
    actions: Iterable[MathAction],
    context_text: str,
    temp: float,
    flow_t: float,
    sf_params: SchoolfieldParams,
    lambda_risk: float = 1.0,
    alpha_styl: float = 1.0,
) -> BanditAction:
    """
    Compute fused qualities, apply a rectified‑flow mapping and return a
    soft‑max propensity BanditAction for the highest‑scoring action.
    """
    lsm = lsm_vector(context_text)

    # Compute fused quality for each action
    qualities = []
    for a in actions:
        q = compute_fused_quality(
            a, lsm, temp, sf_params, lambda_risk=lambda_risk, alpha_styl=alpha_styl
        )
        qualities.append(q)

    # Convert qualities to a vector and apply rectified flow from zero
    source_vec = np.zeros(len(qualities))
    target_vec = np.array(qualities, dtype=float)
    flow_vec = rectified_flow(source_vec, target_vec, flow_t)

    # Soft‑max over the flow‑transformed scores
    max_val = np.max(flow_vec)
    exp_vals = np.exp(flow_vec - max_val)  # numeric stability
    probs = exp_vals / exp_vals.sum()

    # Choose action proportional to its probability (deterministic argmax for demo)
    chosen_idx = int(np.argmax(probs))
    chosen_action = list(actions)[chosen_idx]

    return BanditAction(
        action_id=chosen_action.id,
        propensity=float(probs[chosen_idx]),
        expected_reward=chosen_action.expected_value,
        confidence_bound=chosen_action.risk,  # placeholder
        algorithm="HybridRegretFlowBandit",
    )


def update_bandit_policy(
    updates: List[Tuple[str, float]],
    learning_rate: float = 0.1,
) -> Dict[str, float]:
    """
    Very lightweight policy updater: given a list of (action_id, reward) pairs,
    return a dictionary mapping action_id → updated value estimate using an
    exponential moving average.
    """
    value_estimates: Dict[str, float] = {}
    for aid, reward in updates:
        prev = value_estimates.get(aid, 0.0)
        value_estimates[aid] = (1 - learning_rate) * prev + learning_rate * reward
    return value_estimates


# ----------------------------------------------------------------------
# Supporting dataclass for the returned action (mirrors parent A)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection performed by the hybrid bandit."""
    action_id: str
    propensity: float               # probability of being selected (softmax‑like)
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretFlowBandit"


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny action set
    actions = [
        MathAction(id="A1", tokens=("pronoun", "article"), expected_value=1.2, risk=0.3),
        MathAction(id="A2", tokens=("preposition",), expected_value=0.8, risk=0.1),
        MathAction(id="A3", tokens=("adverb_common", "conjunction"), expected_value=1.0, risk=0.5),
    ]

    # Sample context
    context = "I think that the quick brown fox jumps over the lazy dog while you watch."

    # Schoolfield parameters (reasonable defaults)
    sf_params = SchoolfieldParams(
        E0=1e5,
        Eh=50000.0,
        Dh=80000.0,
        Th=0.0,
        Td=310.0,
        Delta=10.0,
    )

    # Run selection
    chosen = select_action_via_flow(
        actions=actions,
        context_text=context,
        temp=25.0,          # °C
        flow_t=0.6,        # interpolation factor
        sf_params=sf_params,
        lambda_risk=0.8,
        alpha_styl=2.0,
    )

    print("Chosen action:", chosen)

    # Simulate a reward observation and update policy
    updates = [(chosen.action_id, random.uniform(0.0, 2.0))]
    new_vals = update_bandit_policy(updates)
    print("Updated value estimates:", new_vals)